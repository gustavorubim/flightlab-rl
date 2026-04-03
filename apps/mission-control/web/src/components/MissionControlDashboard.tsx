import type {
  ConnectionStatus,
  ControllerMode,
  ControllerOption,
  MissionDraft,
  SessionSnapshot,
} from '@/types';
import { CommandRail } from '@/components/CommandRail';
import { MissionPlanner } from '@/components/MissionPlanner';
import { MissionScene } from '@/components/MissionScene';

interface MissionControlDashboardProps {
  snapshot: SessionSnapshot;
  controllers: ControllerOption[];
  connectionStatus: ConnectionStatus;
  lastUpdatedAt: number;
  error: string | null;
  selectedControllerMode: ControllerMode;
  onSelectedControllerModeChange: (mode: ControllerMode) => void;
  mission: MissionDraft;
  selectedWaypointId: string | null;
  onSelectWaypoint: (waypointId: string | null) => void;
  onAddWaypoint: (coords: { x_m: number; y_m: number }) => string;
  onMoveWaypoint: (waypointId: string, coords: { x_m: number; y_m: number }) => void;
  onUpdateWaypoint: (
    waypointId: string,
    changes: Partial<{
      name: string;
      x_m: number;
      y_m: number;
      altitude_m: number;
      target_airspeed_mps: number;
      acceptance_radius_m: number;
    }>,
  ) => void;
  onReorderWaypoint: (waypointId: string, direction: -1 | 1) => void;
  onDeleteWaypoint: (waypointId: string) => void;
  onCommitRoute: () => void;
  onStartSession: () => void;
  onResetSession: () => void;
  onTakeoff: () => void;
  onPause: () => void;
  onResume: () => void;
  hasUnsavedRoute: boolean;
  cameraMode: 'orbit' | 'chase';
  onCameraModeChange: (mode: 'orbit' | 'chase') => void;
}

export function MissionControlDashboard(props: MissionControlDashboardProps) {
  const {
    snapshot,
    controllers,
    connectionStatus,
    lastUpdatedAt,
    error,
    selectedControllerMode,
    onSelectedControllerModeChange,
    mission,
    selectedWaypointId,
    onSelectWaypoint,
    onAddWaypoint,
    onMoveWaypoint,
    onUpdateWaypoint,
    onReorderWaypoint,
    onDeleteWaypoint,
    onCommitRoute,
    onStartSession,
    onResetSession,
    onTakeoff,
    onPause,
    onResume,
    hasUnsavedRoute,
    cameraMode,
    onCameraModeChange,
  } = props;

  return (
    <div className="mission-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Local mission room</p>
          <h1>Mission Control Interface</h1>
        </div>
        <div className="topbar-tags">
          <span className="status-pill status-pill-live">{snapshot.session.phase.toUpperCase()}</span>
          <span className="status-pill status-pill-warning">
            {connectionStatus.toUpperCase()}
          </span>
        </div>
      </header>

      <div className="workspace">
        <aside className="left-column">
          <CommandRail
            snapshot={snapshot}
            controllers={controllers}
            connectionStatus={connectionStatus}
            lastUpdatedAt={lastUpdatedAt}
            error={error}
            selectedControllerMode={selectedControllerMode}
            onSelectedControllerModeChange={onSelectedControllerModeChange}
            onStartSession={onStartSession}
            onResetSession={onResetSession}
            onTakeoff={onTakeoff}
            onPause={onPause}
            onResume={onResume}
            hasUnsavedRoute={hasUnsavedRoute}
          />
        </aside>

        <main className="center-column">
          <MissionPlanner
            mission={mission}
            selectedWaypointId={selectedWaypointId}
            onSelectWaypoint={onSelectWaypoint}
            onAddWaypoint={onAddWaypoint}
            onMoveWaypoint={onMoveWaypoint}
            onUpdateWaypoint={onUpdateWaypoint}
            onReorderWaypoint={onReorderWaypoint}
            onDeleteWaypoint={onDeleteWaypoint}
            onCommitRoute={onCommitRoute}
            hasUnsavedRoute={hasUnsavedRoute}
          />
        </main>

        <section className="right-column">
          <MissionScene
            snapshot={snapshot}
            cameraMode={cameraMode}
            onCameraModeChange={onCameraModeChange}
          />

          <section className="panel event-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Intel feed</p>
                <h2>Recent events</h2>
              </div>
            </div>
            <ul className="event-list">
              {snapshot.runtime_health.recent_events.slice(0, 6).map((event) => (
                <li key={event}>{event}</li>
              ))}
            </ul>
          </section>
        </section>
      </div>
    </div>
  );
}
