<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import LiquidGlass from './LiquidGlass.vue'
import { loadVisualAssets, disposeVisualAssets, texturedGeometry } from '../sceneVisualAssets.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import {
  axis_fix_radians,
  bounds_of,
  quat_wxyz_to_three,
  describe_geometry,
  geom_target_position,
  trail_point,
} from '../sceneHierarchy.js'

const props = defineProps({
  scene: Object,
  running: Boolean,
  selected: String,
  stale: Boolean,
  live: Boolean,
  sessionId: String,
  overlays: Object,
  toolSafety: Object,
  opaque: Boolean,
  frosted: Boolean,
})

const emit = defineEmits(['select'])

const host = ref(null)
const frameInfo = ref(null)
const unavailable = ref('')
const eefText = ref('')
const counts = ref({ robot: 0, objects: 0, world: 0 })
const provenance = ref(null)
const cutaway = ref(false)
const safetyLabel = ref('未接入')
const assetStatus = ref('等待场景')
let visualAssets = null
let visualManifest = null
let assetRequestKey = ''
let assetController = null

let renderer = null
let scene = null
let camera = null
let controls = null
let grid = null
let axes = null
let bodyHolders = new Map()
let eefMesh = null
let eefAxes = null
let baseRig = null
let trail = null
let trailPoints = []
let trailKey = ''
let geometryKey = ''
let pickables = []
let raycaster = null
let pointer = null
let overlayGroup = null
let viewAdapted = false
let userMovedView = false
let overlaySignature = ''
let worldHolders = new Set()
let raf = 0
let observer = null

const TRAIL_LIMIT = 400

function initScene() {
  if (!host.value) return
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0xe0e7dd)

  const width = host.value.clientWidth || 800
  const height = host.value.clientHeight || 520
  camera = new THREE.PerspectiveCamera(50, width / height, 0.05, 120)
  camera.up.set(0, 0, 1) // MuJoCo 世界 +Z 向上
  camera.position.set(3.2, -3.6, 2.6)

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(width, height)
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.05
  host.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.addEventListener('start', () => { userMovedView = true })
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.minDistance = 0.3
  controls.maxDistance = 30

  grid = new THREE.GridHelper(8, 32, 0x9cafa1, 0xc2cec0)
  grid.rotation.x = Math.PI / 2 // 世界地面是 XY 平面
  grid.material.transparent = true
  grid.material.opacity = 0.14
  scene.add(grid)

  axes = new THREE.AxesHelper(0.4)
  scene.add(axes)

  overlayGroup = new THREE.Group()
  scene.add(overlayGroup)

  baseRig = new THREE.Group()
  baseRig.add(new THREE.AxesHelper(0.28))
  scene.add(baseRig)

  eefMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.035, 24, 16),
    new THREE.MeshStandardMaterial({ color: 0x00d4ff, emissive: 0x0b3d55, roughness: 0.25 }),
  )
  eefAxes = new THREE.AxesHelper(0.18)
  scene.add(eefMesh, eefAxes)

  // 中性光：任何有色光都会污染几何颜色，无法与真实相机画面核对
  scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3a44, 1.05))
  const key = new THREE.DirectionalLight(0xffffff, 1.9)
  key.position.set(3, 2.5, 5)
  const fill = new THREE.DirectionalLight(0xffffff, 0.55)
  fill.position.set(-3, -2, 2)
  scene.add(key, fill)

  trail = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([]),
    new THREE.LineBasicMaterial({ color: 0xa855f7, transparent: true, opacity: 0.9 }),
  )
  scene.add(trail)

  raycaster = new THREE.Raycaster()
  pointer = new THREE.Vector2()
  renderer.domElement.addEventListener('pointerdown', onPointerDown)

  observer = new ResizeObserver(resize)
  observer.observe(host.value)
  animate()
}

function resize() {
  if (!host.value || !renderer || !camera) return
  const width = host.value.clientWidth || 800
  const height = host.value.clientHeight || 520
  camera.aspect = width / height
  camera.updateProjectionMatrix()
  renderer.setSize(width, height)
}

function animate() {
  raf = requestAnimationFrame(animate)
  controls?.update()
  renderer?.render(scene, camera)
}

function disposeDeep(root) {
  if (!root) return
  root.traverse?.((node) => {
    if (!node.geometry?.userData?.assetShared) node.geometry?.dispose?.()
    if (Array.isArray(node.material)) node.material.forEach((m) => m.dispose?.())
    else node.material?.dispose?.()
  })
  if (root.parent) root.parent.remove(root)
  root.clear?.()
}

function primitiveFor(entry) {
  const [a, b, c] = entry.size
  const approx = entry.representation === 'mesh_bbox' || entry.representation === 'mesh_placeholder'
  if (approx) {
    return new THREE.BoxGeometry(Math.max(a * 2, 0.004), Math.max(b * 2, 0.004), Math.max(c * 2, 0.004))
  }
  if (entry.kind === 'sphere') return new THREE.SphereGeometry(Math.max(a, 0.004), 20, 14)
  if (entry.kind === 'cylinder') return new THREE.CylinderGeometry(Math.max(a, 0.004), Math.max(a, 0.004), Math.max(b * 2, 0.004), 20)
  if (entry.kind === 'capsule') return new THREE.CapsuleGeometry(Math.max(a, 0.004), Math.max(b * 2, 0.004), 8, 16)
  if (entry.kind === 'ellipsoid') {
    const geo = new THREE.SphereGeometry(1, 20, 14)
    geo.scale(Math.max(a, 0.004), Math.max(b, 0.004), Math.max(c, 0.004))
    return geo
  }
  return new THREE.BoxGeometry(Math.max(a * 2, 0.004), Math.max(b * 2, 0.004), Math.max(c * 2, 0.004))
}

function materialFor(entry, highlighted) {
  const [r, g, b] = entry.colour || [0.5, 0.5, 0.5]
  const base = new THREE.Color(r, g, b)
  return new THREE.MeshStandardMaterial({
    color: highlighted ? 0xffffff : base,
    emissive: highlighted ? new THREE.Color(0x557b79) : new THREE.Color(0x000000),
    emissiveIntensity: highlighted ? 0.4 : 0,
    roughness: entry.representation?.startsWith('mesh_') ? 0.75 : 0.5,
    metalness: 0.05,
    transparent: entry.alpha < 1,
    opacity: entry.alpha,
    side: THREE.DoubleSide,
  })
}

function makeGeomHolder(entry, highlighted, worldOpacity = null) {
  // worldOpacity 仅用于显式的剖切开关；默认使用几何自身的真实 alpha
  const geomHolder = new THREE.Group()
  const actualGeometry = visualAssets?.geometries.get(String(entry.mesh_id))
  geomHolder.position.fromArray(actualGeometry ? entry.local_position : geom_target_position(entry))
  const q = quat_wxyz_to_three(entry.local_quaternion)
  geomHolder.quaternion.set(q[0], q[1], q[2], q[3])

  let shape = actualGeometry || primitiveFor(entry)
  const material = materialFor(entry, highlighted)
  const original = visualManifest?.materials?.[entry.material_id]
  const textureInfo = visualManifest?.textures?.[original?.texture]
  const texture = visualAssets?.textures.get(original?.texture)
  if (original && texture) {
    const mapped = texturedGeometry(shape, entry, original, textureInfo)
    if (mapped) {
      if (!shape.userData.assetShared) shape.dispose()
      shape = mapped
      material.map = texture
      material.roughness = Math.max(0.32, 0.85 - (original.shininess || 0) * 0.45)
      material.needsUpdate = true
    }
  }
  const mesh = new THREE.Mesh(shape, material)
  if (worldOpacity !== null) {
    mesh.material.transparent = true
    mesh.material.opacity = worldOpacity
  }
  const fix = axis_fix_radians(entry.kind)
  if (fix && !actualGeometry) mesh.rotation.x = fix // 仅圆柱/胶囊需要 Y→Z
  mesh.userData = {
    geom: entry.geom,
    body: entry.body,
    kind: entry.kind,
    representation: entry.representation,
    // 每个 geom 独立材质：记下原始外观，取消选中时逐项还原
    base: {
      color: mesh.material.color.getHex(),
      emissive: mesh.material.emissive.getHex(),
      intensity: mesh.material.emissiveIntensity,
    },
  }
  geomHolder.add(mesh)
  pickables.push(mesh)
  return geomHolder
}

function buildBodyHolder(bodyId, entries, { highlightedName } = {}) {
  const holder = new THREE.Group()
  holder.userData.bodyId = String(bodyId)
  for (const entry of entries) holder.add(makeGeomHolder(entry, false, null))
  const key = highlightedName !== undefined ? highlightedName : props.selected
  applyHighlight(holder, key)
  return holder
}

function applyHighlight(holder, selectedName) {
  const on = Boolean(holder.userData.objectName && holder.userData.objectName === selectedName)
  holder.traverse((node) => {
    if (!node.isMesh || !node.material || !node.userData.base) return
    const base = node.userData.base
    if (on) {
      node.material.color.setHex(0xffffff)
      node.material.emissive.setHex(0x00b4dd)
      node.material.emissiveIntensity = 0.75
    } else {
      // 必须逐项还原：只清除高亮会让切换选中时白色累积
      node.material.color.setHex(base.color)
      node.material.emissive.setHex(base.emissive)
      node.material.emissiveIntensity = base.intensity
    }
  })
}

function startAssetLoad(geometry) {
  const manifest = geometry?.assets
  if (!manifest || !props.sessionId) { assetStatus.value = '简化几何 · 旧会话'; return }
  const key = `${props.sessionId}:${manifest.asset_key || geometry.key}`
  if (key === assetRequestKey) return
  assetController?.abort()
  const controller = new AbortController()
  assetController = controller
  assetRequestKey = key
  visualManifest = manifest
  assetStatus.value = '正在载入真实模型'
  loadVisualAssets(manifest, props.sessionId, controller.signal).then(assets => {
    if (!assets || controller.signal.aborted || key !== assetRequestKey) { disposeVisualAssets(assets); return }
    const previous = visualAssets
    visualAssets = assets
    assetStatus.value = assets.failures.length ? `部分模型降级 · ${assets.failures.length} 项` : `真实网格 ${assets.geometries.size} · 纹理 ${assets.textures.size}`
    geometryKey = ''
    if (props.scene?.geometry) { applyGeometry(props.scene.geometry); applyPose(); if (!userMovedView) focusWorkspace() }
    disposeVisualAssets(previous)
  }).catch(error => {
    if (!controller.signal.aborted && key === assetRequestKey) assetStatus.value = `模型载入失败：${error.message}`
  })
}

function applyGeometry(geometry) {
  if (!geometry || !scene) return
  startAssetLoad(geometry)
  if (geometry.key === geometryKey) return
  geometryKey = geometry.key

  // 递归释放旧树（含所有后代 mesh 与材质），避免泄漏与残留
  for (const holder of bodyHolders.values()) disposeDeep(holder)
  bodyHolders.clear()
  worldHolders = new Set()
  pickables = []

  const info = describe_geometry(geometry)
  counts.value = { robot: info.robot, objects: info.objects, world: info.world }
  provenance.value = info.provenance

  for (const entry of geometry.robot?.geoms || []) {
    const id = String(entry.body)
    if (!bodyHolders.has(id)) {
      const holder = new THREE.Group()
      holder.userData.bodyId = id
      bodyHolders.set(id, holder)
      scene.add(holder)
    }
    bodyHolders.get(id).add(makeGeomHolder(entry, false))
  }

  for (const item of geometry.objects || []) {
    const id = String(item.id)
    const holder = new THREE.Group()
    holder.userData.bodyId = id
    holder.userData.objectName = item.name
    for (const entry of item.geoms || []) holder.add(makeGeomHolder(entry, false))
    bodyHolders.set(`obj:${id}`, holder)
    scene.add(holder)
  }

  for (const entry of geometry.world?.geoms || []) {
    const id = String(entry.body)
    if (!bodyHolders.has(id)) {
      const holder = new THREE.Group()
      holder.userData.bodyId = id
      bodyHolders.set(id, holder)
      scene.add(holder)
    }
    // 厨房几何：默认用真实 alpha；剖切开关只降低不透明几何的可见度
    bodyHolders.get(id).add(makeGeomHolder(entry, false, cutaway.value ? 0.32 : null))
    worldHolders.add(id)
  }
  applySelection()
}

function applyPose() {
  const transforms = props.scene?.transforms || {}
  for (const holder of bodyHolders.values()) {
    const bodyId = holder.userData.bodyId.startsWith('obj:')
      ? holder.userData.bodyId.slice(4)
      : holder.userData.bodyId
    const pose = transforms[bodyId]
    if (!pose) continue
    holder.position.set(pose.p[0], pose.p[1], pose.p[2])
    const q = quat_wxyz_to_three(pose.q)
    holder.quaternion.set(q[0], q[1], q[2], q[3])
  }
}

function applyOverlays(overlays) {
  if (!overlayGroup) return
  const data = overlays || props.overlays
  const signature = data ? JSON.stringify(data) : 'none'
  if (signature === overlaySignature) return
  overlaySignature = signature

  disposeDeep(overlayGroup)
  overlayGroup = new THREE.Group()
  scene.add(overlayGroup)

  if (!data) return

  // zone: {kind, center?, start, end, radius, clearance, label, source}
  // 注意：sphere 在后端只有 start/end（没有 center），必须回退到 start
  for (const zone of data.risk_zones || []) {
    const radius = Math.max(Number(zone.radius) || 0.05, 0.01)
    const material = zoneMaterial(false)
    if (zone.kind === 'sphere') {
      const point = zone.center || zone.start
      if (!point) continue
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 14), material)
      mesh.position.set(...point)
      overlayGroup.add(mesh)
    } else if ((zone.kind === 'capsule' || zone.kind === 'segment') && zone.start && zone.end) {
      overlayGroup.add(segmentMesh(zone.start, zone.end, radius, zoneMaterial(false), 0.18))
      // 胶囊体积 = 圆柱 + 两端球；缺端球会低估占用区域
      for (const end of [zone.start, zone.end]) {
        const cap = new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 12), zoneMaterial(false))
        cap.material.transparent = true
        cap.material.opacity = 0.18
        cap.position.set(...end)
        overlayGroup.add(cap)
      }
    }
  }

  // 工具必须用后端算好的世界段；局部段是相对 body 的，直接画会错位
  for (const segment of data.world_tool_segments || []) {
    if (!segment.start || !segment.end) continue
    const radius = Math.max(Number(segment.radius) || 0.005, 0.004)
    overlayGroup.add(segmentMesh(segment.start, segment.end, radius, toolMaterial(), 1))
    const tip = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 1.6, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0xffd700 }),
    )
    tip.position.set(...segment.end)
    overlayGroup.add(tip)
  }
}

function zoneMaterial(violation) {
  return new THREE.MeshBasicMaterial({
    color: violation ? 0xff5c7a : 0xffcc4d,
    transparent: true,
    opacity: 0.18,
    wireframe: true,
  })
}

function toolMaterial() {
  return new THREE.MeshBasicMaterial({ color: 0xffd700 })
}

function segmentMesh(start, end, radius, material, opacity) {
  const a = new THREE.Vector3(...start)
  const b = new THREE.Vector3(...end)
  const length = a.distanceTo(b)
  if (length < 1e-4) return new THREE.Group()
  const geo = new THREE.CylinderGeometry(radius, radius, length, 12)
  const mesh = new THREE.Mesh(geo, material)
  mesh.position.copy(a).add(b).multiplyScalar(0.5)
  // 圆柱沿 +Y，需对齐到 a->b 方向
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize())
  if (opacity !== undefined) {
    mesh.material.transparent = true
    mesh.material.opacity = opacity
  }
  return mesh
}

function apply(data) {
  if (!renderer || !eefMesh || !data) return
  frameInfo.value = data.frame || null
  if (data.error) {
    unavailable.value = data.error
    return
  }
  unavailable.value = data.available?.geometry === false ? '当前模型没有可导出的几何' : ''
  if (data.geometry) applyGeometry(data.geometry)
  applyPose()

  if (data.eef?.position) {
    const [x, y, z] = data.eef.position
    eefMesh.position.set(x, y, z)
    eefAxes.position.set(x, y, z)
    const q = quat_wxyz_to_three(data.eef.quaternion)
    eefAxes.quaternion.set(q[0], q[1], q[2], q[3])
    eefText.value = `${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)} m`
  }
  if (data.base?.position) {
    baseRig.position.set(...data.base.position)
    const q = quat_wxyz_to_three(data.base.quaternion)
    baseRig.quaternion.set(q[0], q[1], q[2], q[3])
  }
  const safety = data.tool_safety
  safetyLabel.value = data.overlays?.source === 'tool_safety_disabled' || safety?.enabled === false
    ? '未启用'
    : safety?.status || (data.overlays?.source === 'tool_safety_error' ? '观测异常' : '未接入')
  applyOverlays(data.overlays)
  // 首次拿到有效几何后聚焦机械臂工作区一次；之后轮询不再动用户视角
  if (!viewAdapted && data.geometry) {
    viewAdapted = true
    requestAnimationFrame(focusWorkspace)
  }
}

function applySelection() {
  for (const holder of bodyHolders.values()) applyHighlight(holder, props.selected)
}

function onPointerDown(event) {
  if (!renderer || !raycaster) return
  const rect = renderer.domElement.getBoundingClientRect()
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
  raycaster.setFromCamera(pointer, camera)
  const hits = raycaster.intersectObjects(pickables, false)
  if (!hits.length) return
  const hit = hits[0].object
  const holder = hit.parent
  // 拾取只做选择：不发编辑命令，不推进物理
  emit('select', holder?.parent?.userData?.objectName || describeBody(hit.userData.body))
}

function describeBody(bodyId) {
  return `body:${bodyId}`
}

function focusSelection() {
  const target = props.selected
  let centre = null
  for (const holder of bodyHolders.values()) {
    if (holder.userData.objectName !== target) continue
    const box = new THREE.Box3().setFromObject(holder)
    if (!box.isEmpty()) centre = box.getCenter(new THREE.Vector3())
  }
  if (!centre) return
  controls.target.copy(centre)
  const distance = Math.max(0.5, Math.min(1.6, controls.target.distanceTo(camera.position) * 0.5))
  camera.position.copy(centre).add(new THREE.Vector3(distance * 0.7, -distance, distance * 0.6))
  camera.updateProjectionMatrix()
}

/** 默认视角：基座与末端的中间，约 2.3 m，侧俯视。 */
function focusWorkspace() {
  if (!camera || !controls) return
  const base = (props.scene?.base?.position) || props.scene?.geometry && null
  const eef = props.scene?.eef?.position
  if (!base && !eef) {
    adaptView()
    return
  }
  const centre = new THREE.Vector3()
  if (base && eef) {
    centre.set((base[0] + eef[0]) / 2, (base[1] + eef[1]) / 2, (base[2] + eef[2]) / 2)
  } else {
    centre.set(...(eef || base))
  }
  // Pick an interior view using the actual visual geometry. Broad sample rays
  // penalise near walls at the sides, rather than centring only the robot.
  const obstacles = []
  for (const id of worldHolders) bodyHolders.get(String(id))?.traverse(node => { if (node.isMesh) obstacles.push(node) })
  scene.updateMatrixWorld(true)
  const probe = new THREE.Raycaster()
  let best = null
  for (const degrees of [135, 90, 180, 45, 225, 0, 270, 315]) {
    const angle = degrees * Math.PI / 180
    const offset = new THREE.Vector3(Math.cos(angle) * 2.3, Math.sin(angle) * 2.3, 1.2)
    const candidate = centre.clone().add(offset)
    const right = new THREE.Vector3(Math.sin(angle), -Math.cos(angle), 0)
    let score = 0
    for (const horizontal of [-1, 0, 1]) for (const vertical of [-0.3, 0.5]) {
      const target = centre.clone().addScaledVector(right, horizontal).add(new THREE.Vector3(0, 0, vertical))
      const direction = target.sub(candidate)
      probe.set(candidate, direction.clone().normalize())
      probe.far = Math.max(0.1, direction.length() - 0.28)
      if (probe.intersectObjects(obstacles, false).length) score += horizontal === 0 ? 3 : 1
    }
    if (!best || score < best.score) best = { score, position: candidate }
    if (score === 0) break
  }
  controls.target.copy(centre)
  camera.position.copy(best.position)
  controls.update()
  camera.updateProjectionMatrix()
}

function adaptView() {
  const byBody = new Map()
  const transforms = props.scene?.transforms || {}
  for (const holder of bodyHolders.values()) {
    const id = holder.userData.bodyId.startsWith('obj:') ? holder.userData.bodyId.slice(4) : holder.userData.bodyId
    const pos = holder.position
    if (!byBody.has(id)) byBody.set(id, [])
    byBody.get(id).push({ local_position: [pos.x, pos.y, pos.z], local_quaternion: [1, 0, 0, 0] })
  }
  const bounds = bounds_of(
    Object.fromEntries([...byBody].map(([id, entries]) => [id, entries])),
    transforms,
  )
  void bounds
  const box = new THREE.Box3()
  for (const holder of bodyHolders.values()) box.expandByObject(holder)
  box.expandByObject(eefMesh)
  if (box.isEmpty()) return
  const centre = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3()).length() || 1
  controls.target.copy(centre)
  camera.position.copy(centre).add(new THREE.Vector3(size * 0.55, -size * 0.8, size * 0.45))
  camera.updateProjectionMatrix()
}

watch(() => props.scene, (data) => {
  if (!data) return
  apply(data)
  const nextKey = trail_point(data, props.sessionId, trailKey)
  trailKey = nextKey[0]
  if (nextKey[2]) {
    trailPoints = [] // episode 变化或 step 回退：清空旧轨迹
  }
  if (nextKey[1]) {
    trailPoints.push(new THREE.Vector3(...(data.eef?.position || [0, 0, 0])))
    if (trailPoints.length > TRAIL_LIMIT) trailPoints.shift()
    trail?.geometry.setFromPoints(trailPoints)
  }
}, { immediate: true })

watch(() => props.sessionId, () => {
  trailPoints = []
  trailKey = ''
  geometryKey = ''
  assetController?.abort()
  assetRequestKey = ''
  visualManifest = null
  disposeVisualAssets(visualAssets)
  visualAssets = null
  overlaySignature = ''
  viewAdapted = false
  userMovedView = false
  trail?.geometry.setFromPoints([])
  for (const holder of bodyHolders.values()) disposeDeep(holder)
  bodyHolders.clear()
  worldHolders = new Set()
  pickables = []
  if (props.scene) apply(props.scene)
})

watch(() => props.selected, applySelection)

watch(() => props.overlays, (value) => applyOverlays(value))

onMounted(() => {
  try { initScene() } catch (error) { unavailable.value = `三维渲染无法启动：${error.message}。可切换仿真相机。`; return }
  if (props.scene) {
    apply(props.scene)
    const nextKey = trail_point(props.scene, props.sessionId, '')
    trailKey = nextKey[0]
    if (nextKey[1]) {
      trailPoints.push(new THREE.Vector3(...(props.scene.eef?.position || [0, 0, 0])))
      trail.geometry.setFromPoints(trailPoints)
    }
  }
})

onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  assetController?.abort()
  disposeVisualAssets(visualAssets)
  visualAssets = null
  renderer?.domElement?.removeEventListener('pointerdown', onPointerDown)
  observer?.disconnect()
  for (const holder of bodyHolders.values()) disposeDeep(holder)
  bodyHolders.clear()
  worldHolders = new Set()
  pickables = []
  // 释放场景内其余资源（eef/grid/axes/trail/overlay/灯光材质），避免页面切换与热重载泄漏
  for (const child of [...(scene?.children || [])]) {
    if (!child.isLight) disposeDeep(child)
  }
  scene?.clear?.()
  controls?.dispose()
  renderer?.dispose()
  if (renderer?.domElement?.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement)
})

function toggleCutaway() {
  cutaway.value = !cutaway.value
  // 只调世界几何的可见度，不重建几何也不改动其他材质
  for (const id of worldHolders) {
    const holder = bodyHolders.get(String(id))
    holder?.traverse((node) => {
      if (!node.isMesh || !node.material) return
      node.material.transparent = cutaway.value
      node.material.opacity = cutaway.value ? 0.32 : 1
      node.material.needsUpdate = true
    })
  }
}

defineExpose({ focusSelection, adaptView })

const liveLabel = computed(() => {
  if (!props.live) return '未连接'
  return props.stale ? '陈旧' : '实时'
})
const geometryLabel = computed(() => {
  const p = provenance.value
  if (!p) return '几何未加载'
  const truncated = p.truncated ? ' · 已截断' : ''
  const skipped = p.geoms_skipped ? `（碰撞代理跳过 ${p.geoms_skipped}）` : ''
  return `几何 ${p.geometry} · 视觉 ${p.geoms_exported}/${p.geoms_total}${skipped} · ${p.representation}${truncated}`
})
</script>

<template>
  <div class="scene3d">
    <div ref="host" class="canvas" />
    <div class="hud"><span class="tag" :class="{warn:!live || stale}">{{ liveLabel }}</span><span class="tag">{{ safetyLabel === 'violation' ? '保护区越界' : safetyLabel === 'ok' ? '当前未越界' : '安全观测：' + safetyLabel }}</span></div>
    <LiquidGlass light class="tools" :opaque="opaque" :frosted="frosted"><div class="scene-actions"><button type="button" :class="{on:cutaway}" @click="toggleCutaway">{{ cutaway ? '还原外观' : '透视厨房' }}</button><button type="button" @click="adaptView">全景视角</button><button type="button" :disabled="!selected" @click="focusSelection">聚焦选中</button></div></LiquidGlass>
    <p v-if="unavailable" class="note bad" role="status">{{ unavailable }}</p>
    <details v-else class="scene-provenance"><summary>{{ assetStatus }} <span aria-hidden="true">⌄</span></summary><div>{{ geometryLabel }}<br />{{ provenance?.approximate_mesh_geoms ? `仍有 ${provenance.approximate_mesh_geoms} 个网格使用包围盒近似。` : '' }}纹理按原资产 UV 或局部平面映射；定向立方体纹理的复杂网格可能降级。<br />末端 {{ eefText || '—' }} · 连杆 {{ counts.robot }} · 物体 {{ counts.objects }} · 厨房 {{ counts.world }}<br />点击只做选择，拖动只旋转视角；世界 +Z 向上。</div></details>
  </div>
</template>

<style scoped>
.scene3d {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 320px;
  border-radius: var(--radius);
  border: 1px solid var(--line);
  background: radial-gradient(700px 460px at 30% 5%, rgba(168, 85, 247, 0.18), transparent 70%), #070713;
  overflow: hidden;
}

.canvas {
  position: absolute;
  inset: 0;
}

.hud {
  position: absolute;
  top: 10px;
  left: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-width: calc(100% - 20px);
  pointer-events: none;
}

.tag {
  padding: 3px 10px;
  border-radius: 999px;
  background: rgba(10, 10, 31, 0.82);
  border: 1px solid var(--line);
  font-size: 11px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.tag.warn {
  color: #ffe9b0;
  border-color: rgba(255, 204, 77, 0.5);
}

.tools {
  position: absolute;
  top: 10px;
  right: 10px;
  display: flex;
  gap: 6px;
}

.tools button {
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(10, 10, 31, 0.82);
  color: var(--ink);
  font-size: 11px;
}

.note {
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 10px;
  margin: 0;
  font-size: 11px;
  line-height: 1.5;
  color: var(--muted);
}

.note.bad {
  color: #ffd7de;
}
</style>

<style scoped>
.scene-actions{display:flex;flex-direction:column;padding:4px;gap:2px}
.scene-actions button{background:transparent!important;border:0!important;color:inherit!important;min-height:44px!important;padding:8px 14px!important;font-size:12px!important;backdrop-filter:none!important}
.scene-provenance{position:absolute;bottom:90px;left:20px;max-width:calc(100% - 40px);color:#35584a;font-size:10px;z-index:3}
.scene-provenance summary{cursor:pointer;list-style:none;background:rgba(246,249,242,.9);padding:10px 12px;min-height:44px;display:flex;align-items:center;gap:8px;border:1px solid rgba(255,255,255,.7);border-radius:10px;max-width:100%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.scene-provenance>div{position:absolute;bottom:100%;margin-bottom:8px;width:320px;max-width:calc(100vw - 64px);background:#fcfcfa;color:#4c6256;padding:16px;border-radius:16px;box-shadow:0 6px 30px #203d2820;line-height:1.7;max-height:220px;overflow:auto}
@media(max-width:640px){.scene-provenance{left:12px;bottom:82px}.hud{bottom:118px!important}}
</style>
