import * as THREE from 'three'
import { getToken } from './api.js'

const SAFE_FILE = /^(?:mesh_(?:[0-9]{4,8}_)?[a-f0-9]{12}\.json|tex_(?:[0-9]{4,8}_)?[a-f0-9]{12}\.png)$/
export function assetUrl(session, file) {
  if (!/^sess_[A-Za-z0-9_]{1,80}$/.test(session) || !SAFE_FILE.test(file)) throw new Error('场景资产地址无效')
  return `/api/v1/sessions/${encodeURIComponent(session)}/visual-assets/${file}`
}

export function geometryFromAsset(data) {
  if (!Array.isArray(data.positions) || data.positions.length % 3 || data.positions.length > 3_000_000 || !data.positions.length || !data.positions.every(Number.isFinite)) throw new Error('模型顶点数据无效')
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3))
  if (Array.isArray(data.indices)) {
    const count = data.positions.length / 3
    if (data.indices.length % 3 || data.indices.some(v => !Number.isInteger(v) || v < 0 || v >= count)) throw new Error('模型三角形索引无效')
    geometry.setIndex(data.indices)
  }
  if (Array.isArray(data.normals) && data.normals.length === data.positions.length && data.normals.every(Number.isFinite)) geometry.setAttribute('normal', new THREE.Float32BufferAttribute(data.normals, 3))
  else geometry.computeVertexNormals()
  if (Array.isArray(data.uvs) && data.uvs.length === data.positions.length / 3 * 2 && data.uvs.every(Number.isFinite)) geometry.setAttribute('uv', new THREE.Float32BufferAttribute(data.uvs, 2))
  geometry.computeBoundingBox()
  geometry.userData.assetShared = true
  return geometry
}

async function imageFromBlob(blob) {
  if (typeof createImageBitmap === 'function') return createImageBitmap(blob, { imageOrientation: 'flipY' })
  const url = URL.createObjectURL(blob)
  try {
    const image = new Image()
    await new Promise((resolve,reject) => { image.onload=resolve;image.onerror=()=>reject(new Error('纹理图像无法解码'));image.src=url })
    return image
  } finally { URL.revokeObjectURL(url) }
}

export async function loadVisualAssets(manifest, session, signal) {
  const geometries = new Map(), textures = new Map(), failures = []
  const token = getToken()
  const jobs = []
  for (const [id,entry] of Object.entries(manifest.meshes || {})) jobs.push(async () => {
    const response = await fetch(assetUrl(session, entry.file), {headers:token?{'X-Console-Token':token}:{},signal})
    if (!response.ok) throw new Error(`模型 ${id} 加载失败 (${response.status})`)
    geometries.set(id,geometryFromAsset(await response.json()))
  })
  for (const [id,entry] of Object.entries(manifest.textures || {})) jobs.push(async () => {
    const response = await fetch(assetUrl(session,entry.file), {headers:token?{'X-Console-Token':token}:{},signal})
    if (!response.ok) throw new Error(`纹理 ${id} 加载失败 (${response.status})`)
    const image = await imageFromBlob(await response.blob())
    if (signal.aborted) { image.close?.(); return }
    const texture = new THREE.Texture(image)
    texture.flipY = !(typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap)
    texture.colorSpace = THREE.SRGBColorSpace
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping
    texture.anisotropy = 4
    texture.needsUpdate = true
    textures.set(id,texture)
  })
  let cursor = 0
  await Promise.all(Array.from({length:Math.min(5,jobs.length)}, async () => {
    while (cursor < jobs.length && !signal.aborted) {
      const index = cursor++
      try { await jobs[index]() } catch(error) { if(error.name!=='AbortError')failures.push(error.message) }
    }
  }))
  if (signal.aborted) { disposeVisualAssets({geometries,textures}); return null }
  return { geometries,textures,failures }
}

export function disposeVisualAssets(assets) {
  for (const geometry of assets?.geometries?.values() || []) geometry.dispose()
  for (const texture of assets?.textures?.values() || []) { texture.image?.close?.(); texture.dispose() }
}

/** Per-draw UV mapping: keep authored mesh UVs; project uniform materials in metres.
 * Cube textures use the compiled six-face atlas; nonuniform directional mesh
 * mapping remains a measured fallback instead of inventing a different texture.
 */
export function texturedGeometry(source, entry, material, info) {
  if (!info) return null
  const cube = info.type === 'cube'
  const uniformCube = cube && info.cube?.uniform
  const authored = source.getAttribute('uv') && entry.texture_support === 'mesh_uv'
  const project = entry.texture_support === 'planar_projection'
  const boxCube = cube && entry.kind === 'box' && info.height === info.width * 6
  if (!authored && !project && !boxCube && !uniformCube) return null
  const geometry = source.clone()
  geometry.userData.assetShared = false
  const position = geometry.getAttribute('position'), normal = geometry.getAttribute('normal')
  const existing = geometry.getAttribute('uv')
  const uv = new Float32Array(position.count * 2)
  const repeat = material.repeat || [1,1]
  for(let i=0;i<position.count;i++) {
    const x=position.getX(i),y=position.getY(i),z=position.getZ(i)
    const nx=normal?.getX(i)||0,ny=normal?.getY(i)||0,nz=normal?.getZ(i)||1
    let u=existing?.getX(i)||0,v=existing?.getY(i)||0
    if(project || (!authored && !boxCube && !existing)) {
      if(Math.abs(nz)>=Math.abs(nx)&&Math.abs(nz)>=Math.abs(ny)){u=x;v=y}
      else if(Math.abs(ny)>=Math.abs(nx)){u=x;v=z}
      else {u=y;v=z}
    }
    if(boxCube || uniformCube) {
      let block = 0
      if(boxCube && !uniformCube) {
        // +X/-X/+Y/-Y/+Z/-Z -> L/R/D/U/B/F in compiled atlas.
        const letters=info.cube?.block_letters||['R','L','U','D','F','B']
        const face=Math.abs(nx)>=Math.abs(ny)&&Math.abs(nx)>=Math.abs(nz)?(nx>0?'L':'R'):Math.abs(ny)>=Math.abs(nz)?(ny>0?'D':'U'):(nz>0?'B':'F')
        block=Math.max(0,letters.indexOf(face))
      }
      const faces=info.height===info.width*6?6:1
      uv[i*2]=u
      uv[i*2+1]=(faces-1-block+Math.min(.9999,Math.max(.0001,v)))/faces
    } else {uv[i*2]=u*repeat[0];uv[i*2+1]=v*repeat[1]}
  }
  geometry.setAttribute('uv',new THREE.BufferAttribute(uv,2))
  return geometry
}
