import assert from 'node:assert/strict'
import { displacementAt } from '../src/glassOptics.js'
import { assetUrl,geometryFromAsset,texturedGeometry } from '../src/sceneVisualAssets.js'
import * as THREE from 'three'
let count=0
const test=(name,fn)=>{fn();count++;console.log('PASS '+name)}
test('glass has a flat centre and finite dimension-limited bevel',()=>{
  for(const [w,h] of [[44,44],[220,54],[800,64]]){
    assert.deepEqual(displacementAt(w/2,h/2,w,h,h/2,12),[0,0])
    for(let x=0;x<w;x+=3)for(let y=0;y<h;y+=3){const d=displacementAt(x,y,w,h,h/2,12);assert(d.every(Number.isFinite));assert(Math.hypot(...d)<=6.01)}
  }
})
test('glass left and right edges refract symmetrically',()=>{
  const a=displacementAt(3,27,220,54,27,16),b=displacementAt(217,27,220,54,27,16)
  assert(Math.abs(a[0]+b[0])<1e-9);assert(a[0]>0);assert(b[0]<0);assert(Math.abs(a[1])<1e-9)
})
test('asset URLs accept published hashes and reject traversal/other origins',()=>{
  assert.equal(assetUrl('sess_abc','mesh_0123456789ab.json'),'/api/v1/sessions/sess_abc/visual-assets/mesh_0123456789ab.json')
  assert(assetUrl('sess_abc','tex_0123456789ab.png').endsWith('.png'))
  for(const bad of ['../manifest.json','https://a.example/p.png','manifest.json','tex_deadbeef.png'])assert.throws(()=>assetUrl('sess_abc',bad))
  assert.throws(()=>assetUrl('../sess_abc','mesh_0123456789ab.json'))
})
test('compiled mesh positions are not rescaled or shifted',()=>{
  const p=[-1,0,0,2,0,0,0,3,0]
  const g=geometryFromAsset({positions:p,indices:[0,1,2],uvs:[0,0,1,0,0,1]})
  assert.deepEqual([...g.attributes.position.array],p);assert.equal(g.index.count,3);assert.equal(g.attributes.uv.count,3);assert(g.attributes.normal)
  g.dispose()
})
test('mesh decoder refuses malformed/out of range geometry',()=>{
  assert.throws(()=>geometryFromAsset({positions:[0,0,NaN]}))
  assert.throws(()=>geometryFromAsset({positions:[0,0,0],indices:[0,1,2]}))
  assert.throws(()=>geometryFromAsset({positions:[0,0,0,1]}))
})
test('UV repeat is applied to a clone and authored asset remains reusable',()=>{
  const g=geometryFromAsset({positions:[0,0,0,1,0,0,0,1,0],uvs:[0,0,1,0,0,1]})
  const mapped=texturedGeometry(g,{texture_support:'mesh_uv',kind:'mesh'},{repeat:[2,3]},{type:'2d'})
  assert.deepEqual([...mapped.attributes.uv.array],[0,0,2,0,0,3]);assert.deepEqual([...g.attributes.uv.array],[0,0,1,0,0,1]);assert.notEqual(mapped,g)
  mapped.dispose();g.dispose()
})
test('unsupported directional cube mapping does not invent planar UVs',()=>{
  const g=geometryFromAsset({positions:[0,0,0,1,0,0,0,1,0]})
  assert.equal(texturedGeometry(g,{texture_support:'cube_texture_requires_direction',kind:'mesh'},{repeat:[1,1]},{type:'cube',cube:{uniform:false}}),null)
  g.dispose()
})
console.log(`${count}/${count} passed`)
