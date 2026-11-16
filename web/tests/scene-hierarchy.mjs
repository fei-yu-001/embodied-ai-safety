/**
 * Synthetic-pose tests for the 3D transform hierarchy.
 *
 * These run without a browser or MuJoCo: they pin the maths that the reviewer
 * flagged (geom local offset + rotation composing with the body pose) so a
 * regression cannot hide behind "it looked fine in the viewport".
 */

import assert from 'node:assert/strict'
import {
  axis_fix_radians,
  compose_point,
  describe_geometry,
  geom_target_position,
  quat_wxyz_to_three,
  trail_point,
  world_position,
} from '../src/sceneHierarchy.js'

const cases = []
function test(name, fn) {
  try {
    fn()
    cases.push(['PASS', name])
  } catch (error) {
    cases.push(['FAIL', `${name} :: ${error.message}`])
  }
}

const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol
const nearVec = (a, b, tol = 1e-6) => a.length === b.length && a.every((v, i) => near(v, b[i], tol))

// --- 90 度旋转 ---
const Q_Z90 = [Math.SQRT1_2, 0, 0, Math.SQRT1_2] // wxyz
const Q_X90 = [Math.SQRT1_2, Math.SQRT1_2, 0, 0]

test('quat wxyz -> three xyzw reorders only', () => {
  assert.deepEqual(quat_wxyz_to_three([0.5, 0.1, 0.2, 0.3]), [0.1, 0.2, 0.3, 0.5])
})

test('90deg about Z maps +X to +Y', () => {
  assert.ok(nearVec(compose_point([0, 0, 0], Q_Z90, [1, 0, 0]), [0, 1, 0]), 'expected +X -> +Y')
})

test('bbox centre is rotated by the geom quaternion before being added', () => {
  // 局部 +Z 偏移在 geom 绕 X 转 90 度后应指向 -Y（不能直接相加）
  const result = compose_point([0.1, 0, 0], Q_X90, [0, 0, 0.5])
  assert.ok(nearVec(result, [0.1, -0.5, 0], 1e-6), `got ${JSON.stringify(result)}`)
})

test('zero bbox centre reduces to the geom local position', () => {
  assert.deepEqual(compose_point([0.2, 0.3, 0.4], Q_X90, [0, 0, 0]), [0.2, 0.3, 0.4])
})

// --- body + geom 复合 ---
const body = { p: [1, 2, 3], q: Q_Z90 }
const entry = {
  kind: 'box',
  size: [0.05, 0.05, 0.05],
  local_position: [0.1, 0, 0],
  local_quaternion: [1, 0, 0, 0],
  bbox_centre: [0, 0, 0],
}

test('world position composes body pose with geom local offset', () => {
  // body 绕 Z 转 90 度：局部 +X 变成世界 +Y
  const result = world_position(body, entry)
  assert.ok(nearVec(result, [1, 2.1, 3], 1e-6), `got ${JSON.stringify(result)}`)
})

test('geom quaternion rotates the shape, not the geom origin', () => {
  const rotated = { ...entry, local_quaternion: Q_X90, local_position: [0, 0, 0.2] }
  // 位置与旋转是同一变换的独立分量：geom 的原点只受 local_position 影响，
  // body 绕 Z 转 90 度而 local +Z 不受影响，所以世界原点仍是 [1, 2, 3.2]
  const origin = world_position(body, rotated)
  assert.ok(nearVec(origin, [1, 2, 3.2], 1e-6), `got ${JSON.stringify(origin)}`)
})

test('a point inside the geom goes through body and geom rotation', () => {
  const rotated = { ...entry, local_quaternion: Q_X90, local_position: [0, 0, 0.2], bbox_centre: [0, 0, 0.5] }
  // 局部 [0,0,0.5] 绕 X 90 -> [0,-0.5,0]；加局部位置 -> [0,-0.5,0.2]；
  // body 绕 Z 90 -> [0.5,0,0.2]；加 body 位置 -> [1.5,2,3.2]
  const point = world_position(body, rotated)
  assert.ok(nearVec(point, [1.5, 2, 3.2], 1e-6), `got ${JSON.stringify(point)}`)
})

test('geom target position includes rotated bbox centre', () => {
  const withCentre = { ...entry, bbox_centre: [0, 0, 0.5], local_quaternion: Q_X90 }
  assert.ok(nearVec(geom_target_position(withCentre), [0.1, -0.5, 0], 1e-6))
})

// --- 轴修正只作用于圆柱/胶囊 ---
test('axis fix applies to cylinders and capsules only', () => {
  assert.ok(axis_fix_radians('cylinder') !== 0)
  assert.ok(axis_fix_radians('capsule') !== 0)
  for (const kind of ['box', 'sphere', 'ellipsoid', 'mesh']) {
    assert.equal(axis_fix_radians(kind), 0, `${kind} 不应旋转`)
  }
})

// --- 轨迹按 episode/step 去重 ---
const sceneAt = (episode, step, eef = [0, 0, 1]) => ({
  episode,
  step,
  eef: { position: eef, quaternion: [1, 0, 0, 0] },
})

test('trail appends on a new step and skips a paused repeat', () => {
  const first = trail_point(sceneAt(1, 0), 'sess_a', '')
  assert.equal(first[1], true)
  const repeat = trail_point(sceneAt(1, 0), 'sess_a', first[0])
  assert.equal(repeat[1], false, '同一 episode/step 的轮询不能追加')
  const next = trail_point(sceneAt(1, 1), 'sess_a', first[0])
  assert.equal(next[1], true)
})

test('trail clears when episode changes or step goes back', () => {
  const key = 'sess_a:1:5'
  assert.equal(trail_point(sceneAt(2, 0), 'sess_a', key)[2], true, '换 episode 必须清空')
  assert.equal(trail_point(sceneAt(1, 0), 'sess_a', key)[2], true, 'step 回退（重置）必须清空')
  assert.equal(trail_point(sceneAt(1, 6), 'sess_a', key)[2], false, '正常前进不清空')
})

test('trail does not append on error or missing eef', () => {
  assert.equal(trail_point({ error: 'boom' }, 'sess_a', 'k')[1], false)
  assert.equal(trail_point({ episode: 1, step: 2 }, 'sess_a', 'k')[1], false)
  assert.equal(trail_point(null, 'sess_a', 'k')[1], false)
})

test('trail keys are per session', () => {
  const a = trail_point(sceneAt(1, 3), 'sess_a', '')
  const b = trail_point(sceneAt(1, 3), 'sess_b', '')
  assert.notEqual(a[0], b[0])
})

// --- 计数与来源 ---
test('describe_geometry counts object geoms and keeps provenance', () => {
  const geometry = {
    provenance: { geometry: 'mj_model_geom', geoms_exported: 3, geoms_total: 9, representation: 'primitive' },
    robot: { geoms: [{ body: 1 }, { body: 1 }] },
    objects: [{ id: '4', geoms: [{ body: 4 }] }],
    world: { geoms: [{ body: 3 }] },
  }
  const info = describe_geometry(geometry)
  assert.deepEqual({ robot: info.robot, objects: info.objects, world: info.world }, { robot: 2, objects: 1, world: 1 })
  assert.equal(info.provenance.representation, 'primitive')
})

test('describe_geometry tolerates missing payloads', () => {
  assert.deepEqual(describe_geometry(null), { robot: 0, objects: 0, world: 0, provenance: null })
})

let failed = 0
for (const [status, name] of cases) {
  if (status === 'FAIL') failed += 1
  console.log(`${status} ${name}`)
}
console.log(`\n${cases.length - failed}/${cases.length} passed`)
process.exit(failed ? 1 : 0)
