/**
 * Pure geometry-tree helpers for the 3D inspection view.
 *
 * Kept framework-free so the transform hierarchy can be unit-tested with
 * synthetic poses instead of a 30 s kitchen load.
 *
 * Hierarchy (three levels, one source of truth per level):
 *
 *   bodyHolder   <- dynamic body world pose (updated every poll)
 *     geomHolder <- geom local_position / local_quaternion (static)
 *       mesh     <- bbox_centre offset only, plus axis fix for cylinders
 *
 * Cylinders/capsules are the only shapes whose axis convention differs
 * (Three.js builds them along +Y, MuJoCo along +Z), so the rotation belongs on
 * the mesh and never on a shared holder.
 */

const PI = Math.PI

// MuJoCo geom kinds whose Three.js primitive is built along +Y instead of +Z.
export const Y_AXIS_KINDS = new Set(['cylinder', 'capsule'])

/** MuJoCo stores wxyz; Three.js wants xyzw. */
export function quat_wxyz_to_three(quat) {
  const q = Array.isArray(quat) && quat.length >= 4 ? quat.slice(0, 4) : [1, 0, 0, 0]
  return [q[1], q[2], q[3], q[0]]
}

/** Rotate `point` by a wxyz quaternion. */
export function apply_quat_to_point(quatWxyz, point) {
  const [w, x, y, z] = Array.isArray(quatWxyz) && quatWxyz.length >= 4 ? quatWxyz.slice(0, 4) : [1, 0, 0, 0]
  const [px, py, pz] = [point[0] || 0, point[1] || 0, point[2] || 0]
  // q * p * q^-1, expanded
  const ix = w * px + y * pz - z * py
  const iy = w * py + z * px - x * pz
  const iz = w * pz + x * py - y * px
  const iw = -x * px - y * py - z * pz
  return [
    ix * w + iw * -x + iy * -z - iz * -y,
    iy * w + iw * -y + iz * -x - ix * -z,
    iz * w + iw * -z + ix * -y - iy * -x,
  ]
}

/**
 * `offset` is expressed in the geom's local frame, then translated.
 *
 * The bbox centre must be rotated by the geom quaternion before adding the
 * geom position; adding them directly ignores the rotation and misplaces the
 * shape whenever the geom is not axis-aligned.
 */
export function compose_point(localPosition, quatWxyz, offset) {
  const rotated = apply_quat_to_point(quatWxyz, offset || [0, 0, 0])
  return [
    Number(localPosition[0] || 0) + rotated[0],
    Number(localPosition[1] || 0) + rotated[1],
    Number(localPosition[2] || 0) + rotated[2],
  ]
}

export function geom_target_position(entry) {
  return compose_point(
    entry.local_position || [0, 0, 0],
    entry.local_quaternion || [1, 0, 0, 0],
    entry.bbox_centre || [0, 0, 0],
  )
}

/** Static local transform of one geom, relative to its body. */
export function geometry_transform(entry) {
  return {
    position: geom_target_position(entry),
    quaternion: quat_wxyz_to_three(entry.local_quaternion),
    axis_fix: axis_fix_radians(entry.kind),
    kind: entry.kind,
    representation: entry.representation,
  }
}

/** Rotation that maps a +Y-built primitive onto MuJoCo's +Z axis. */
export function axis_fix_radians(kind) {
  return Y_AXIS_KINDS.has(kind) ? -PI / 2 : 0
}

/** Where a geom's centre actually sits in world space. */
export function world_position(bodyPose, entry) {
  const local = geometry_transform(entry).position
  if (!bodyPose) return local
  const offset = apply_quat_to_point(bodyPose.q || [1, 0, 0, 0], local)
  const base = bodyPose.p || [0, 0, 0]
  return [base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]]
}

/** Counts and provenance for the HUD. */
export function describe_geometry(geometry) {
  if (!geometry) return { robot: 0, objects: 0, world: 0, provenance: null }
  return {
    robot: (geometry.robot?.geoms || []).length,
    objects: (geometry.objects || []).reduce((sum, item) => sum + (item.geoms || []).length, 0),
    world: (geometry.world?.geoms || []).length,
    provenance: geometry.provenance || null,
  }
}

/** Axis-aligned bounds of placed geometry, for camera framing. */
export function bounds_of(entriesByBody, transforms) {
  const points = []
  for (const [bodyId, entries] of Object.entries(entriesByBody || {})) {
    const pose = (transforms || {})[String(bodyId)] || (transforms || {})[bodyId]
    for (const entry of entries) points.push(world_position(pose, entry))
  }
  if (!points.length) return null
  const low = [0, 1, 2].map((i) => Math.min(...points.map((p) => p[i])))
  const high = [0, 1, 2].map((i) => Math.max(...points.map((p) => p[i])))
  const centre = [0, 1, 2].map((i) => (low[i] + high[i]) / 2)
  const radius = Math.max(0.2, Math.max(...[0, 1, 2].map((i) => high[i] - low[i])) / 2)
  return { centre, radius }
}

/**
 * Returns [key, shouldAppend, shouldClear] for the EEF trail.
 *
 * The adapter reports episode/step at the top level. A paused poll repeats the
 * same step, so it must not add a second point; a reset steps back to 0 and
 * must clear the trail.
 */
export function trail_point(scene, sessionId, previousKey) {
  if (!scene || scene.error) return [previousKey, false, false]
  const { episode, step } = scene
  if (episode === undefined || step === undefined || !scene.eef) return [previousKey, false, false]
  const key = `${sessionId}:${episode}:${step}`
  if (key === previousKey) return [previousKey, false, false]
  let clear = false
  if (previousKey) {
    const parts = String(previousKey).split(':')
    const prevEpisode = Number(parts[parts.length - 2])
    const prevStep = Number(parts[parts.length - 1])
    if (Number.isFinite(prevEpisode) && Number.isFinite(prevStep)) {
      clear = episode !== prevEpisode || step < prevStep
    }
  }
  return [key, true, clear]
}
