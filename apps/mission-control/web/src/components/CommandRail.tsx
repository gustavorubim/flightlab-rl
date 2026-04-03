import type { ConnectionStatus, ControllerMode, ControllerOption, SessionSnapshot } from '@/types';
import { SessionStatusCard } from '@/components/SessionStatusCard';

interface CommandRailProps {
  snapshot: SessionSnapshot;
  controllers: ControllerOption[];
  connectionStatus: ConnectionStatus;
  lastUpdatedAt: number;
  error: string | null;
  selectedControllerMode: ControllerMode;
  onSelectedControllerModeChange: (mode: ControllerMode) => void;
  onStartSession: () => void;
  onResetSession: () => void;
  onTakeoff: () => void;
  onPause: () => void;
  onResume: () => void;
  hasUnsavedRoute: boolean;
}

export function CommandRail({
  snapshot,
  controllers,
  connectionStatus,
  lastUpdatedAt,
  error,
  selectedControllerMode,
  onSelectedControllerModeChange,
  onStartSession,
  onResetSession,
  onTakeoff,
  onPause,
  onResume,
  hasUnsavedRoute,
}: CommandRailProps) {
  return (
    <div className="rail">
      <div className="brand-block panel">
        <p className="eyebrow">FlightLab</p>
        <h1>Mission Control</h1>
        <p className="brand-copy">
          Runway-ready launch, live replanning, and tactical 3D telemetry in one local
          operations room.
        </p>
      </div>

      <SessionStatusCard
        snapshot={snapshot}
        connectionStatus={connectionStatus}
        lastUpdatedAt={lastUpdatedAt}
        error={error}
      />

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Command stack</p>
            <h2>Operator controls</h2>
          </div>
          {hasUnsavedRoute ? <span className="status-pill status-pill-warning">DIRTY ROUTE</span> : null}
        </div>

        <label className="field">
          <span>Controller mode</span>
          <select
            value={selectedControllerMode}
            onChange={(event) =>
              onSelectedControllerModeChange(event.target.value as ControllerMode)
            }
            aria-label="Controller mode"
          >
            {controllers.map((controller) => (
              <option key={controller.mode} value={controller.mode}>
                {controller.label}
              </option>
            ))}
          </select>
        </label>

        <div className="control-stack">
          <button type="button" className="button button-primary" onClick={onStartSession}>
            Start Session
          </button>
          <button type="button" className="button" onClick={onTakeoff}>
            Trigger Takeoff
          </button>
          <button type="button" className="button" onClick={onPause}>
            Pause
          </button>
          <button type="button" className="button" onClick={onResume}>
            Resume
          </button>
          <button type="button" className="button button-ghost" onClick={onResetSession}>
            Reset
          </button>
        </div>

        <div className="command-footnote">
          <p>
            Current phase: <strong>{snapshot.session.phase}</strong>
          </p>
          <p>
            Aircraft state: <strong>{snapshot.aircraft.on_ground ? 'on deck' : 'airborne'}</strong>
          </p>
        </div>
      </section>
    </div>
  );
}
