import { useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';

import {
  calculateRouteDistance,
  formatMeters,
  formatSpeed,
} from '@/lib/mission';
import type { MissionDraft, MissionWaypoint } from '@/types';

const DEFAULT_FIELD = { width: 1120, height: 700 };
const WORLD_EXTENT_X = 5200;
const WORLD_EXTENT_Y = 3800;

interface MissionPlannerProps {
  mission: MissionDraft;
  selectedWaypointId: string | null;
  onSelectWaypoint: (waypointId: string | null) => void;
  onAddWaypoint: (coords: Pick<MissionWaypoint, 'x_m' | 'y_m'>) => string;
  onMoveWaypoint: (waypointId: string, coords: Pick<MissionWaypoint, 'x_m' | 'y_m'>) => void;
  onUpdateWaypoint: (
    waypointId: string,
    changes: Partial<Omit<MissionWaypoint, 'id'>>,
  ) => void;
  onReorderWaypoint: (waypointId: string, direction: -1 | 1) => void;
  onDeleteWaypoint: (waypointId: string) => void;
  onCommitRoute: () => void;
  hasUnsavedRoute: boolean;
}

function toFieldCoordinates(
  event: ReactMouseEvent<HTMLDivElement>,
  element: HTMLDivElement | null,
): Pick<MissionWaypoint, 'x_m' | 'y_m'> {
  const rect = element?.getBoundingClientRect();
  const width = rect && rect.width > 0 ? rect.width : DEFAULT_FIELD.width;
  const height = rect && rect.height > 0 ? rect.height : DEFAULT_FIELD.height;
  const left = rect?.left ?? 0;
  const top = rect?.top ?? 0;
  const normalizedX = (event.clientX - left) / width;
  const normalizedY = (event.clientY - top) / height;

  return {
    x_m: (normalizedX - 0.5) * WORLD_EXTENT_X,
    y_m: (0.5 - normalizedY) * WORLD_EXTENT_Y,
  };
}

function fieldToPercent(value: number, extent: number): number {
  return ((value / extent) + 0.5) * 100;
}

function displayMeters(value: number): string {
  return `${Math.round(value).toLocaleString('en-US')} m`;
}

export function MissionPlanner({
  mission,
  selectedWaypointId,
  onSelectWaypoint,
  onAddWaypoint,
  onMoveWaypoint,
  onUpdateWaypoint,
  onReorderWaypoint,
  onDeleteWaypoint,
  onCommitRoute,
  hasUnsavedRoute,
}: MissionPlannerProps) {
  const fieldRef = useRef<HTMLDivElement | null>(null);
  const [draggingWaypointId, setDraggingWaypointId] = useState<string | null>(null);

  const selectedWaypoint = useMemo(
    () => mission.waypoints.find((waypoint) => waypoint.id === selectedWaypointId) ?? null,
    [mission.waypoints, selectedWaypointId],
  );

  const totalDistance = useMemo(() => calculateRouteDistance(mission), [mission]);

  function handleFieldClick(event: ReactMouseEvent<HTMLDivElement>) {
    const coords = toFieldCoordinates(event, fieldRef.current);
    const newWaypointId = onAddWaypoint(coords);
    onSelectWaypoint(newWaypointId);
  }

  function handleFieldMove(event: ReactMouseEvent<HTMLDivElement>) {
    if (!draggingWaypointId) {
      return;
    }

    const coords = toFieldCoordinates(event, fieldRef.current);
    onMoveWaypoint(draggingWaypointId, coords);
  }

  function beginDragging(waypointId: string) {
    setDraggingWaypointId(waypointId);
    onSelectWaypoint(waypointId);
  }

  function endDragging() {
    setDraggingWaypointId(null);
  }

  return (
    <section className="panel planner-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Mission board</p>
          <h2>Tactical planner</h2>
        </div>
        <div className="planner-metrics">
          <span>{mission.waypoints.length} waypoints</span>
          <span>{displayMeters(totalDistance)}</span>
        </div>
      </div>

      <div className="planner-layout">
        <div className="planner-canvas-wrap">
          <div
            ref={fieldRef}
            className="planner-canvas"
            data-testid="planner-field"
            onClick={handleFieldClick}
            onMouseMove={handleFieldMove}
            onMouseUp={endDragging}
            onMouseLeave={endDragging}
            role="application"
            aria-label="Mission planner map"
          >
            <div className="planner-grid" aria-hidden="true" />
            <div className="planner-runway" aria-hidden="true">
              <span>RUNWAY 27</span>
            </div>
            <div className="planner-axis planner-axis-x">EAST</div>
            <div className="planner-axis planner-axis-y">NORTH</div>

            {mission.waypoints.map((waypoint, index) => {
              const left = fieldToPercent(waypoint.x_m, WORLD_EXTENT_X);
              const top = fieldToPercent(-waypoint.y_m, WORLD_EXTENT_Y);
              const isSelected = waypoint.id === selectedWaypointId;

              return (
                <button
                  key={waypoint.id}
                  type="button"
                  data-testid={isSelected ? 'selected-waypoint-marker' : `waypoint-marker-${waypoint.id}`}
                  className={`waypoint-marker ${isSelected ? 'is-selected' : ''}`}
                  style={{ left: `${left}%`, top: `${top}%` }}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelectWaypoint(waypoint.id);
                  }}
                  onMouseDown={(event) => {
                    event.stopPropagation();
                    beginDragging(waypoint.id);
                  }}
                  aria-label={`Waypoint ${index + 1}`}
                >
                  <span className="waypoint-marker-index">{index + 1}</span>
                </button>
              );
            })}
          </div>

          <div className="planner-help">
            <p>Click the map to add a waypoint. Drag a marker to reposition it.</p>
            <p>Waypoints are committed to the backend as a full route replacement.</p>
          </div>
        </div>

        <div className="planner-sidebar">
          <div className="panel planner-details">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Selected target</p>
                <h3>{selectedWaypoint?.name ?? 'No waypoint selected'}</h3>
              </div>
              {hasUnsavedRoute ? (
                <span className="status-pill status-pill-warning">UNSAVED</span>
              ) : (
                <span className="status-pill status-pill-live">SYNCED</span>
              )}
            </div>

            {selectedWaypoint ? (
              <div className="planner-fields">
                <label className="field">
                  <span>Name</span>
                  <input
                    value={selectedWaypoint.name}
                    onChange={(event) =>
                      onUpdateWaypoint(selectedWaypoint.id, { name: event.target.value })
                    }
                    aria-label="Waypoint name"
                  />
                </label>
                <label className="field">
                  <span>Altitude (m)</span>
                  <input
                    type="number"
                    value={selectedWaypoint.altitude_m}
                    onChange={(event) =>
                      onUpdateWaypoint(selectedWaypoint.id, {
                        altitude_m: Number(event.target.value),
                      })
                    }
                    aria-label="Altitude (m)"
                  />
                </label>
                <label className="field">
                  <span>Target speed (m/s)</span>
                  <input
                    type="number"
                    value={selectedWaypoint.target_airspeed_mps}
                    onChange={(event) =>
                      onUpdateWaypoint(selectedWaypoint.id, {
                        target_airspeed_mps: Number(event.target.value),
                      })
                    }
                    aria-label="Target speed (m/s)"
                  />
                </label>
                <label className="field">
                  <span>Acceptance radius (m)</span>
                  <input
                    type="number"
                    value={selectedWaypoint.acceptance_radius_m}
                    onChange={(event) =>
                      onUpdateWaypoint(selectedWaypoint.id, {
                        acceptance_radius_m: Number(event.target.value),
                      })
                    }
                    aria-label="Acceptance radius (m)"
                  />
                </label>
                <div className="planner-coords">
                  <span data-testid="selected-waypoint-x">
                    X {displayMeters(selectedWaypoint.x_m)}
                  </span>
                  <span data-testid="selected-waypoint-y">
                    Y {displayMeters(selectedWaypoint.y_m)}
                  </span>
                </div>
              </div>
            ) : (
              <p className="muted">
                Select a waypoint marker or row to expose altitude, speed, and position controls.
              </p>
            )}

            <button type="button" className="button button-primary" onClick={onCommitRoute}>
              Commit Route
            </button>
          </div>

          <div className="panel planner-list">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Route order</p>
                <h3>Waypoint stack</h3>
              </div>
              <span className="status-pill status-pill-amber">LIVE EDIT</span>
            </div>

            <ul className="route-list">
              {mission.waypoints.map((waypoint, index) => (
                <li
                  key={waypoint.id}
                  className={waypoint.id === selectedWaypointId ? 'is-selected' : ''}
                >
                  <button
                    type="button"
                    className="route-list-item"
                    onClick={() => onSelectWaypoint(waypoint.id)}
                  >
                    <span className="route-index">{index + 1}</span>
                    <div className="route-meta">
                      <strong>{waypoint.name}</strong>
                      <span>
                        {displayMeters(waypoint.altitude_m)} | {formatSpeed(waypoint.target_airspeed_mps)}
                      </span>
                    </div>
                  </button>
                  <div className="route-actions">
                    <button
                      type="button"
                      className="icon-button"
                      onClick={() => onReorderWaypoint(waypoint.id, -1)}
                      disabled={index === 0}
                      aria-label={`Move ${waypoint.name} up`}
                    >
                      Move Up
                    </button>
                    <button
                      type="button"
                      className="icon-button"
                      onClick={() => onReorderWaypoint(waypoint.id, 1)}
                      disabled={index === mission.waypoints.length - 1}
                      aria-label={`Move ${waypoint.name} down`}
                    >
                      Move Down
                    </button>
                    <button
                      type="button"
                      className="icon-button danger"
                      onClick={() => onDeleteWaypoint(waypoint.id)}
                      aria-label={`Delete ${waypoint.name}`}
                    >
                      Delete
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
