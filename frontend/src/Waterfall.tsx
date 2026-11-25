import { forwardRef, useImperativeHandle, useRef, useState } from 'react'
import { VIRIDIS, cssColor } from './colormap'
import type { Row, TrackItem, TracksMessage } from './protocol'

export const HISTORY_ROWS = 400 // 20 s at 20 rows/s

export interface WaterfallHandle {
  push: (row: Row) => void
  pushTracks: (msg: TracksMessage) => void
  clear: () => void
}

interface Scale {
  f0: number
  fs: number
  lo: number
  hi: number
  rowsPerS: number
}

interface Box extends TrackItem {
  t1: number // scene time of the last row it was seen in
}

const STATE_COLOR: Record<TrackItem['state'], string> = {
  observing: '#ffd166',
  identified: '#06d6a0',
  unknown: '#ef476f',
}

function boxLabel(b: Box): string {
  if (b.state === 'observing') return `observing ${b.n_frames}`
  const conf = b.confidence === null ? '' : ` ${(b.confidence * 100).toFixed(0)}%`
  return b.state === 'unknown' ? `unknown${conf}` : `${b.label}${conf}`
}

// Waterfall display on a Canvas 2D: the newest row is drawn on top, older rows move down.
// Frames of detected signals are drawn on an overlay canvas at screen resolution.
export const Waterfall = forwardRef<WaterfallHandle, { rowsPerS: number }>(function Waterfall(
  { rowsPerS },
  ref,
) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const overlay = useRef<HTMLCanvasElement>(null)
  const boxes = useRef(new Map<number, Box>())
  const band = useRef({ f0: 0, fs: 1 })
  const [scale, setScale] = useState<Scale | null>(null)
  const [now, setNow] = useState(0)

  const span = () => HISTORY_ROWS / (rowsPerS || 20)

  function drawOverlay(t: number) {
    const o = overlay.current
    if (!o) return
    const w = o.clientWidth
    const h = o.clientHeight
    const dpr = window.devicePixelRatio || 1
    if (o.width !== Math.round(w * dpr) || o.height !== Math.round(h * dpr)) {
      o.width = Math.round(w * dpr)
      o.height = Math.round(h * dpr)
    }
    const ctx = o.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const { f0, fs } = band.current
    const s = span()
    const x = (f: number) => ((f - (f0 - fs / 2)) / fs) * w
    const y = (tt: number) => ((t - tt) / s) * h
    ctx.font = '12px system-ui, sans-serif'
    ctx.lineWidth = 1.5
    for (const [id, b] of boxes.current) {
      if (t - b.t1 > s) {
        boxes.current.delete(id)
        continue
      }
      const color = STATE_COLOR[b.state]
      const x0 = x(b.f_lo_hz)
      const top = y(b.t1)
      ctx.strokeStyle = color
      ctx.setLineDash(b.state === 'observing' ? [4, 3] : [])
      ctx.strokeRect(x0, top, x(b.f_hi_hz) - x0, Math.min(y(b.t0), h) - top)
      ctx.fillStyle = color
      ctx.fillText(boxLabel(b), x0, Math.max(top - 3, 12))
    }
  }

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
      band.current = { f0: row.f0_hz, fs: row.fs_hz }
      if (row.row % 10 === 0) {
        setScale({ f0: row.f0_hz, fs: row.fs_hz, lo: row.lo_db, hi: row.hi_db, rowsPerS })
      }
      setNow(row.t)
    },
    pushTracks(msg: TracksMessage) {
      for (const tr of msg.tracks) boxes.current.set(tr.id, { ...tr, t1: msg.t })
      drawOverlay(msg.t)
    },
    clear() {
      const c = canvas.current
      c?.getContext('2d')?.clearRect(0, 0, c.width, c.height)
      boxes.current.clear()
      drawOverlay(0)
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

  return (
    <div className="waterfall">
      <div className="waterfall-plot">
        <div className="time-axis">
          <span>{now.toFixed(1)} s</span>
          <span>{Math.max(0, now - span()).toFixed(1)} s</span>
        </div>
        <div className="waterfall-stack">
          <canvas ref={canvas} width={1024} height={HISTORY_ROWS} />
          <canvas ref={overlay} className="overlay" />
        </div>
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
