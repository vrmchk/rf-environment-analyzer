import { decode } from '@msgpack/msgpack'
import { useCallback, useEffect, useRef, useState } from 'react'
import { EventLog } from './EventLog'
import type {
  Command,
  EventMessage,
  JsonMessage,
  Row,
  Scene,
  SessionParams,
  SessionStarted,
  SessionSummary,
} from './protocol'
import { Waterfall, type WaterfallHandle } from './Waterfall'

type Status = 'disconnected' | 'idle' | 'running' | 'paused'

const DEFAULT_PARAMS: SessionParams = { gamma_db: 3, tau: 0.95, k_max: 20 }

export default function App() {
  const ws = useRef<WebSocket | null>(null)
  const waterfall = useRef<WaterfallHandle>(null)
  const selectedRef = useRef<string | null>(null)
  const [scenes, setScenes] = useState<Scene[]>([])
  const [scene, setScene] = useState('')
  const [params, setParams] = useState<SessionParams>(DEFAULT_PARAMS)
  const [status, setStatus] = useState<Status>('disconnected')
  const [session, setSession] = useState<SessionStarted | null>(null)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [events, setEvents] = useState<EventMessage[]>([])
  const [message, setMessage] = useState('')

  const refreshSessions = useCallback(() => {
    fetch('/api/sessions')
      .then((r) => r.json())
      .then((list: SessionSummary[]) => {
        setSessions(list)
        if (selectedRef.current === null && list.length) select(list[0].id)
      })
      .catch(() => setMessage('Cannot load sessions'))
  }, [])

  // The log shows one session: stored events from the database, then live ones.
  function select(id: string) {
    selectedRef.current = id
    setSelected(id)
    setEvents([])
    fetch(`/api/sessions/${id}/events`)
      .then((r) => r.json())
      .then((list: EventMessage[]) => {
        if (selectedRef.current !== id) return
        setEvents((live) => {
          const seen = new Set(live.map((e) => e.track_id))
          return [...live, ...list.filter((e) => !seen.has(e.track_id)).reverse()]
        })
      })
      .catch(() => setMessage('Cannot load events'))
  }

  useEffect(() => {
    fetch('/api/scenes')
      .then((r) => r.json())
      .then((list: Scene[]) => {
        setScenes(list)
        if (list.length) setScene(list[0].name)
      })
      .catch(() => setMessage('Cannot load scenes'))
    refreshSessions()
  }, [refreshSessions])

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
        } else if (msg.type === 'event') {
          if (msg.session_id === selectedRef.current) setEvents((prev) => [msg, ...prev])
        } else if (msg.type === 'session_started') {
          setSession((prev) => {
            if (prev?.session_id !== msg.session_id) {
              waterfall.current?.clear()
              select(msg.session_id)
              refreshSessions()
            }
            return msg
          })
          setStatus(msg.paused ? 'paused' : 'running')
          setMessage('')
        } else if (msg.type === 'session_ended') {
          setStatus('idle')
          setMessage(`Session ${msg.reason}${msg.error ? `: ${msg.error}` : ''}`)
          refreshSessions()
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
  }, [refreshSessions])

  const send = useCallback((cmd: Command) => ws.current?.send(JSON.stringify(cmd)), [])
  const active = status === 'running' || status === 'paused'
  const numberInput = (key: keyof SessionParams, label: string, step: number) => (
    <label>
      {label}{' '}
      <input
        type="number"
        step={step}
        value={params[key]}
        disabled={active}
        onChange={(e) => setParams({ ...params, [key]: Number(e.target.value) })}
      />
    </label>
  )

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
        {numberInput('gamma_db', 'Detection, dB', 0.5)}
        {numberInput('tau', 'Confidence τ', 0.01)}
        {numberInput('k_max', 'Max frames', 1)}
        <button
          disabled={active || !scene || status === 'disconnected'}
          onClick={() => send({ type: 'start', scene, params })}
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
      <div className="panel">
        <label>
          Event log{' '}
          <select value={selected ?? ''} onChange={(e) => select(e.target.value)}>
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {new Date(s.started_at).toLocaleString()} · {s.scene} · {s.n_events} events
                {s.end_reason ? '' : ' · running'}
              </option>
            ))}
          </select>
        </label>
        {selected && (
          <a className="button" href={`/api/sessions/${selected}/events.csv`} download>
            Export CSV
          </a>
        )}
      </div>
      <EventLog events={events} />
    </div>
  )
}
