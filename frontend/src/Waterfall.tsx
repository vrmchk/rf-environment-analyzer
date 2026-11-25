import { forwardRef, useImperativeHandle, useRef, useState } from 'react'
import { VIRIDIS, cssColor } from './colormap'
import type { Row } from './protocol'

export const HISTORY_ROWS = 400 // 20 s at 20 rows/s

export interface WaterfallHandle {
  push: (row: Row) => void
  clear: () => void
}

interface Scale {
  f0: number
  fs: number
  lo: number
  hi: number
  rowsPerS: number
}

// Waterfall display on a Canvas 2D: the newest row is drawn on top, older rows move down.
export const Waterfall = forwardRef<WaterfallHandle, { rowsPerS: number }>(function Waterfall(
  { rowsPerS },
  ref,
) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [scale, setScale] = useState<Scale | null>(null)
  const [now, setNow] = useState(0)

  useImperativeHandle(ref, () => ({
    push(row: Row) {
      const c = canvas.current
      if (!c) return
      if (c.width !== row.n) {
        c.width = row.n
        c.height = HISTORY_ROWS
      }
      const ctx = c.getContext('2d')
      if (!ctx) return
      ctx.drawImage(c, 0, 1)
      const img = ctx.createImageData(row.n, 1)
      for (let i = 0; i < row.n; i++) {
        const v = row.levels[i] * 3
        img.data[i * 4] = VIRIDIS[v]
        img.data[i * 4 + 1] = VIRIDIS[v + 1]
        img.data[i * 4 + 2] = VIRIDIS[v + 2]
        img.data[i * 4 + 3] = 255
      }
      ctx.putImageData(img, 0, 0)
      if (row.row % 10 === 0) {
        setScale({ f0: row.f0_hz, fs: row.fs_hz, lo: row.lo_db, hi: row.hi_db, rowsPerS })
      }
      setNow(row.t)
    },
    clear() {
      const c = canvas.current
      c?.getContext('2d')?.clearRect(0, 0, c.width, c.height)
      setScale(null)
      setNow(0)
    },
  }))

  const ticks = scale
    ? [-0.5, -0.25, 0, 0.25, 0.5].map((x) => ({
        left: `${(x + 0.5) * 100}%`,
        label: ((scale.f0 + x * scale.fs) / 1e6).toFixed(3),
      }))
    : []
  const span = HISTORY_ROWS / (scale?.rowsPerS || rowsPerS || 20)

  return (
    <div className="waterfall">
      <div className="waterfall-plot">
        <div className="time-axis">
          <span>{now.toFixed(1)} s</span>
          <span>{Math.max(0, now - span).toFixed(1)} s</span>
        </div>
        <canvas ref={canvas} width={1024} height={HISTORY_ROWS} />
        <div className="color-scale">
          <span>{scale ? `${scale.hi.toFixed(0)} dB` : ''}</span>
          <div
            className="color-bar"
            style={{
              background: `linear-gradient(to top, ${[0, 64, 128, 192, 255]
                .map(cssColor)
                .join(', ')})`,
            }}
          />
          <span>{scale ? `${scale.lo.toFixed(0)} dB` : ''}</span>
        </div>
      </div>
      <div className="freq-axis">
        {ticks.map((t) => (
          <span key={t.left} style={{ left: t.left }}>
            {t.label}
          </span>
        ))}
      </div>
      <div className="axis-title">Frequency, MHz</div>
    </div>
  )
})
