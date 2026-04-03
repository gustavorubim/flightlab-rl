import type { ConnectionStatus, SessionSnapshot } from '@/types';
import { formatMeters, formatSpeed } from '@/lib/mission';

interface SessionStatusCardProps {
  snapshot: SessionSnapshot;
  connectionStatus: ConnectionStatus;
  lastUpdatedAt: number;
  error: string | null;
}

function statusLabel(connectionStatus: ConnectionStatus): string {
  switch (connectionStatus) {
    case 'live':
      return 'LIVE';
    case 'stale':
      return 'STALE';
    case 'offline':
      return 'OFFLINE';
    case 'connecting':
    default:
      return 'CONNECTING';
  }
}

export function SessionStatusCard({
  snapshot,
  connectionStatus,
  lastUpdatedAt,
  error,
}: SessionStatusCardProps) {
  const runtime = snapshot.runtime_health;

  return (
    <section
      className="panel status-card"
      aria-label="Session status"
      data-status={connectionStatus}
    >
      <div className="panel-header">
        <div>
          <p className="eyebrow">Session</p>
          <h2>Mission Telemetry</h2>
        </div>
        <span className={`status-pill status-pill-${connectionStatus}`}>
          {statusLabel(connectionStatus)}
        </span>
      </div>

      <dl className="status-grid">
        <div>
          <dt>Phase</dt>
          <dd>{snapshot.session.phase}</dd>
        </div>
        <div>
          <dt>Controller</dt>
          <dd>{snapshot.session.controller_mode}</dd>
        </div>
        <div>
          <dt>Sim Time</dt>
          <dd>{snapshot.session.sim_time_s.toFixed(1)} s</dd>
        </div>
        <div>
          <dt>Paused</dt>
          <dd>{snapshot.session.paused ? 'Yes' : 'No'}</dd>
        </div>
        <div>
          <dt>Airspeed</dt>
          <dd>{formatSpeed(snapshot.aircraft.airspeed_mps)}</dd>
        </div>
        <div>
          <dt>Altitude</dt>
          <dd>{formatMeters(snapshot.aircraft.altitude_m)}</dd>
        </div>
      </dl>

      <div className="status-channel">
        <div className="status-line">
          <span>Command</span>
          <strong>{runtime.command_status}</strong>
        </div>
        <div className="status-line">
          <span>Last sync</span>
          <strong>
            {connectionStatus === 'live'
              ? 'streaming'
              : `${Math.max(0, Math.round((Date.now() - lastUpdatedAt) / 1000))}s ago`}
          </strong>
        </div>
        <div className="status-line">
          <span>Last error</span>
          <strong>{error ?? runtime.last_error ?? 'None'}</strong>
        </div>
      </div>

      <div className="status-events" aria-label="Recent events">
        <p className="eyebrow">Recent events</p>
        {runtime.recent_events.length > 0 ? (
          <ul>
            {runtime.recent_events.slice(0, 4).map((event) => (
              <li key={event}>{event}</li>
            ))}
          </ul>
        ) : (
          <p className="muted">No events captured yet.</p>
        )}
      </div>
    </section>
  );
}
