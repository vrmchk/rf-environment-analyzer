import type { EventMessage } from './protocol'

const fmt = (v: number | null, digits: number) => (v === null ? '—' : v.toFixed(digits))

// Event log of the session (FR-10, FR-11): newest first.
export function EventLog({ events }: { events: EventMessage[] }) {
  return (
    <div className="event-log">
      <table>
        <thead>
          <tr>
            <th>Detected, s</th>
            <th>Decided, s</th>
            <th>Frequency, MHz</th>
            <th>Bandwidth, kHz</th>
            <th>Class</th>
            <th>Confidence</th>
            <th>Frames</th>
            <th>SNR, dB</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e) => (
            <tr key={`${e.session_id}-${e.track_id}`} className={e.status}>
              <td>{e.detected_at_s.toFixed(2)}</td>
              <td>{e.decided_at_s.toFixed(2)}</td>
              <td>{(e.center_freq_hz / 1e6).toFixed(4)}</td>
              <td>
                {(e.bandwidth_hz / 1e3).toFixed(1)}
                {e.bandwidth_out_of_range ? ' *' : ''}
              </td>
              <td>{e.status === 'identified' ? e.label : 'unknown'}</td>
              <td>{e.confidence === null ? '—' : `${(e.confidence * 100).toFixed(0)}%`}</td>
              <td>{e.n_frames}</td>
              <td>{fmt(e.snr_db, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {events.length === 0 && <div className="empty">No events yet</div>}
    </div>
  )
}
