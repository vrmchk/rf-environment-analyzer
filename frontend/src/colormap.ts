// Viridis colormap as a 256-entry RGB lookup table (linear interpolation between anchors).
const ANCHORS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [180, 222, 44],
  [253, 231, 37],
]

export const VIRIDIS: Uint8ClampedArray = (() => {
  const lut = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (ANCHORS.length - 1)
    const k = Math.min(Math.floor(x), ANCHORS.length - 2)
    const f = x - k
    for (let c = 0; c < 3; c++) {
      lut[i * 3 + c] = ANCHORS[k][c] + f * (ANCHORS[k + 1][c] - ANCHORS[k][c])
    }
  }
  return lut
})()

export function cssColor(level: number): string {
  const i = Math.max(0, Math.min(255, Math.round(level))) * 3
  return `rgb(${VIRIDIS[i]}, ${VIRIDIS[i + 1]}, ${VIRIDIS[i + 2]})`
}
