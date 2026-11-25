import { decode } from '@msgpack/msgpack'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { Command, JsonMessage, Row, Scene, SessionStarted } from './protocol'
import { Waterfall, type WaterfallHandle } from './Waterfall'

type Status = 'disconnected' | 'idle' | 'running' | 'paused'

export default function App() {
  const ws = useRef<WebSocket | null>(null)
  const waterfall = useRef<WaterfallHandle>(null)
  const [scenes, setScenes] = useState<Scene[]>([])
  const [scene, setScene] = useState('')
  const [status, setStatus] = useState<Status>('disconnected')
  const [session, setSession] = useState<SessionStarted | null>(null)
  const [message, setMessage] = useState('')

  useEffect(() => {
    fetch('/api/scenes')
      .then((r) => r.json())
      .then((list: Scene[]) => {
        setScenes(list)
        if (list.length) setScene(list[0].name)
      })
      .catch(() => setMessage('Cannot load scenes'))
  }, [])

  // One WebSocket for the page; reconnects after a drop (the session keeps running).
  useEffect(() => {
    let closed = false
    let retry: number | undefined
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const sock = new WebSocket(`${proto}://${location.host}/ws`)
      sock.binaryType = 'arraybuffer'
      sock.onopen = () => setStatus('idle')
      sock.onclose = () => {
        setStatus('disconnected')
        if (!closed) retry = window.setTimeout(connect, 1000)
      }
      sock.onmessage = (ev) => {
        if (ev.data instanceof ArrayBuffer) {
          waterfall.current?.push(decode(new Uint8Array(ev.data)) as Row)
          return
        }
        const msg = JSON.parse(ev.data) as JsonMessage
        if (msg.type === 'tracks') {
          waterfall.current?.pushTracks(msg)
        } else if (msg.type === 'session_started') {
          setSession((prev) => {
            if (prev?.session_id !== msg.session_id) waterfall.current?.clear()
            return msg
          })
          setStatus(msg.paused ? 'paused' : 'running')
          setMessage('')
        } else if (msg.type === 'session_ended') {
          setStatus('idle')
          setMessage(`Session ${msg.reason}${msg.error ? `: ${msg.error}` : ''}`)
        } else if (msg.type === 'error') {
          setMessage(msg.message)
        }
      }
      ws.current = sock
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(retry)
      ws.current?.close()
    }
  }, [])

  const send = useCallback((cmd: Command) => ws.current?.send(JSON.stringify(cmd)), [])
  const active = status === 'running' || status === 'paused'

  return (
    <div className="app">
      <header className="panel">
        <strong>RF environment analyzer</strong>
        <label>
          Scene{' '}
          <select value={scene} onChange={(e) => setScene(e.target.value)} disabled={active}>
            {scenes.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name} ({s.duration_s.toFixed(0)} s)
              </option>
            ))}
          </select>
        </label>
        <button
          disabled={active || !scene || status === 'disconnected'}
          onClick={() => send({ type: 'start', scene, params: {} })}
        >
          Start
        </button>
        {status === 'paused' ? (
          <button onClick={() => send({ type: 'resume' })}>Resume</button>
        ) : (
          <button disabled={status !== 'running'} onClick={() => send({ type: 'pause' })}>
            Pause
          </button>
        )}
        <button disabled={!active} onClick={() => send({ type: 'stop' })}>
          Stop
        </button>
        <span className="status">
          {status}
          {session && active ? ` · ${session.scene}` : ''}
          {message ? ` · ${message}` : ''}
        </span>
      </header>
      <Waterfall ref={waterfall} rowsPerS={session?.rows_per_s ?? 20} />
    </div>
  )
}
