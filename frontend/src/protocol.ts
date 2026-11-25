// Messages of the backend WebSocket /ws (docs: architecture.md, sections 3 and 11).

export interface SessionParams {
  gamma_db: number
  tau: number
  k_max: number
}

export interface SessionStarted {
  type: 'session_started'
  session_id: string
  scene: string
  fs_hz: number
  f0_hz: number
  n: number
  rows_per_s: number
  params: SessionParams
  paused: boolean
}

export interface SessionEnded {
  type: 'session_ended'
  session_id: string
  reason: 'finished' | 'stopped' | 'error'
  error: string | null
}

export interface ErrorMessage {
  type: 'error'
  message: string
}

export type JsonMessage = SessionStarted | SessionEnded | ErrorMessage

// Waterfall row, MessagePack; `levels` is the 0-255 quantized spectrum.
export interface Row {
  type: 'row'
  row: number
  t: number
  f0_hz: number
  fs_hz: number
  n: number
  lo_db: number
  hi_db: number
  wall: number
  levels: Uint8Array
}

export interface Scene {
  name: string
  sample_rate: number
  center_freq: number
  duration_s: number
  n_signals: number
}

export type Command =
  | { type: 'start'; scene: string; params: Partial<SessionParams> }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'stop' }
