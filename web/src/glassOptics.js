/** Analytic rounded-surface refraction map; generated locally, no scene readback.
 * Original implementation. The neutral centre is flat; only the bevel bends.
 */
export function displacementAt(x, y, width, height, radius, band = 16) {
  const r = Math.min(radius, width / 2, height / 2)
  const px = x - width / 2, py = y - height / 2
  const qx = Math.abs(px) - (width / 2 - r)
  const qy = Math.abs(py) - (height / 2 - r)
  const ox = Math.max(qx, 0), oy = Math.max(qy, 0)
  const length = Math.hypot(ox, oy)
  const distance = length + Math.min(Math.max(qx, qy), 0) - r
  const depth = -distance
  if (depth <= 0 || depth >= band) return [0, 0]
  let nx = 0, ny = 0
  if (length > 0) { nx = ox / length; ny = oy / length }
  else if (qx > qy) nx = 1
  else ny = 1
  nx *= Math.sign(px); ny *= Math.sign(py)
  const t = Math.max(0.015, Math.min(1, depth / band))
  // Quarter-circle bevel slope -> Snell refraction (air / neutral glass).
  const slope = (1 - t) / Math.sqrt(Math.max(0.001, 2 * t - t * t))
  const incidence = Math.atan(slope * 0.8)
  const refracted = Math.asin(Math.sin(incidence) / 1.45)
  const shift = Math.min(band * 0.5, band * 0.55 * Math.tan(incidence - refracted)) * (1 - t) ** 0.65
  return [-nx * shift, -ny * shift]
}

export function createRefractionMap(width, height, radius) {
  const w = Math.max(1, Math.round(width)), h = Math.max(1, Math.round(height))
  const canvas = document.createElement('canvas')
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  const image = ctx.createImageData(w, h)
  const band = Math.min(20, h * 0.29, w * 0.2)
  const scale = Math.max(1, band)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [dx, dy] = displacementAt(x + 0.5, y + 0.5, w, h, radius, band)
    const i = (y * w + x) * 4
    image.data[i] = Math.round(255 * (0.5 + dx / scale))
    image.data[i + 1] = Math.round(255 * (0.5 + dy / scale))
    image.data[i + 2] = 128; image.data[i + 3] = 255
  }
  ctx.putImageData(image, 0, 0)
  return { url: canvas.toDataURL(), scale, width: w, height: h }
}
