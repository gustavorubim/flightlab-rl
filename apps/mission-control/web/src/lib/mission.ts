import type {
  BackendControllerOption,
  BackendSessionSnapshot,
  ControllerMode,
  ControllerOption,
  MissionDraft,
  MissionPayload,
  MissionWaypoint,
  MissionWaypointPayload,
  SessionSnapshot,
} from '@/types';

let waypointSequence = 0;

const DEFAULT_WAYPOINTS: MissionWaypoint[] = [
  {
    id: 'umbra-1',
    name: 'Umbra-1',
    x_m: 800,
    y_m: 0,
    altitude_m: 180,
    target_airspeed_mps: 62,
    acceptance_radius_m: 120,
  },
  {
    id: 'umbra-2',
    name: 'Umbra-2',
    x_m: 2200,
    y_m: 1300,
    altitude_m: 410,
    target_airspeed_mps: 76,
    acceptance_radius_m: 150,
  },
  {
    id: 'umbra-3',
    name: 'Umbra-3',
    x_m: 4200,
    y_m: -900,
    altitude_m: 620,
    target_airspeed_mps: 88,
    acceptance_radius_m: 170,
  },
];

export function nextWaypointId(): string {
  waypointSequence += 1;
  return `wp-${waypointSequence}`;
}

export function createWaypointDraft(
  index: number,
  coords: Pick<MissionWaypoint, 'x_m' | 'y_m'>,
): MissionWaypoint {
  return {
    id: nextWaypointId(),
    name: `Waypoint ${index + 1}`,
    x_m: coords.x_m,
    y_m: coords.y_m,
    altitude_m: 250 + index * 80,
    target_airspeed_mps: 68 + index * 6,
    acceptance_radius_m: 140,
  };
}

export function createDefaultMissionDraft(): MissionDraft {
  return {
    name: 'Operation Black Kite',
    waypoints: DEFAULT_WAYPOINTS.map((waypoint) => ({ ...waypoint })),
  };
}

export function createDemoSnapshot(): SessionSnapshot {
  return {
    session: {
      session_id: 'demo-session',
      phase: 'standby',
      controller_mode: 'pid',
      sim_time_s: 0,
      paused: false,
    },
    aircraft: {
      position_x_m: 0,
      position_y_m: 0,
      altitude_m: 120,
      roll_rad: 0,
      pitch_rad: 0,
      heading_rad: 0,
      airspeed_mps: 0,
      groundspeed_mps: 0,
      vertical_speed_mps: 0,
      on_ground: true,
      time_s: 0,
    },
    mission: {
      name: 'Operation Black Kite',
      waypoints: DEFAULT_WAYPOINTS.map((waypoint) => ({ ...waypoint })),
      active_waypoint_index: 0,
      desired_track_rad: 0,
      distance_to_waypoint_m: 800,
    },
    runtime_health: {
      command_status: 'idle',
      last_error: null,
      recent_events: ['Awaiting operator commands'],
    },
    trail: [],
  };
}

function normalizeNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function normalizeString(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.trim().length > 0 ? value : fallback;
}

function normalizeBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

export function normalizeControllerOptions(
  raw: unknown,
): ControllerOption[] {
  const candidates = Array.isArray(raw)
    ? raw
    : Array.isArray((raw as { controllers?: unknown[] } | null)?.controllers)
      ? (raw as { controllers: unknown[] }).controllers
      : Array.isArray((raw as { options?: unknown[] } | null)?.options)
        ? (raw as { options: unknown[] }).options
        : [];

  const normalized = candidates
    .map((item): ControllerOption | null => {
      if (!item || typeof item !== 'object') {
        return null;
      }

      const candidate = item as BackendControllerOption & { mode?: string };
      const mode = candidate.mode === 'rl_phase_switched' ? 'rl_phase_switched' : 'pid';
      return {
        mode,
        label:
          candidate.label ??
          (mode === 'pid' ? 'PID Autopilot' : 'Phase-Switched RL'),
        description:
          candidate.description ??
          (mode === 'pid'
            ? 'Deterministic classical baseline'
            : 'Switches between takeoff and enroute RL checkpoints'),
        enabled: candidate.available ?? candidate.enabled ?? true,
      };
    })
    .filter((option): option is ControllerOption => option !== null);

  if (normalized.length > 0) {
    return normalized;
  }

  return [
    {
      mode: 'pid',
      label: 'PID Autopilot',
      description: 'Deterministic classical baseline',
      enabled: true,
    },
    {
      mode: 'rl_phase_switched',
      label: 'Phase-Switched RL',
      description: 'Switches between takeoff and enroute RL checkpoints',
      enabled: true,
    },
  ];
}

export function normalizeMissionDraft(
  raw: BackendSessionSnapshot['mission'] | MissionDraft | null | undefined,
): MissionDraft {
  const sourceWaypoints = Array.isArray(raw?.waypoints)
    ? (raw.waypoints as Array<Partial<MissionWaypointPayload> & { id?: string }>)
    : null;
  const rawName =
    raw && typeof raw === 'object' && 'mission_name' in raw
      ? raw.mission_name
      : raw?.name;

  if (!sourceWaypoints || sourceWaypoints.length === 0) {
    return createDefaultMissionDraft();
  }

  return {
    name: normalizeString(rawName, 'Operation Black Kite'),
    waypoints: sourceWaypoints.map((waypoint, index) => ({
      id: normalizeString(waypoint.id, nextWaypointId()),
      name: normalizeString(waypoint.name, `Waypoint ${index + 1}`),
      x_m: normalizeNumber(waypoint.x_m, index * 800),
      y_m: normalizeNumber(waypoint.y_m, index * 250),
      altitude_m: normalizeNumber(waypoint.altitude_m, 200 + index * 60),
      target_airspeed_mps: normalizeNumber(
        waypoint.target_airspeed_mps,
        65 + index * 5,
      ),
      acceptance_radius_m: normalizeNumber(waypoint.acceptance_radius_m, 140),
    })),
  };
}

export function normalizeSessionSnapshot(
  raw: BackendSessionSnapshot | null | undefined,
): SessionSnapshot {
  const demo = createDemoSnapshot();

  if (!raw || typeof raw !== 'object') {
    return demo;
  }

  const session = raw.session ?? {};
  const aircraft = raw.aircraft ?? {};
  const mission = raw.mission ?? {};
  const runtimeHealth = raw.runtime_health ?? {};
  const trail = Array.isArray(raw.trail) ? raw.trail : [];

  return {
    session: {
      session_id: normalizeString(session.session_id, demo.session.session_id),
      phase: normalizeString(session.phase, demo.session.phase),
      controller_mode: normalizeControllerMode(
        normalizeString(session.controller_mode, demo.session.controller_mode),
      ),
      sim_time_s: normalizeNumber(session.sim_time_s, demo.session.sim_time_s),
      paused: normalizeBoolean(session.paused, demo.session.paused),
    },
    aircraft: {
      position_x_m: normalizeNumber(aircraft.position_x_m, demo.aircraft.position_x_m),
      position_y_m: normalizeNumber(aircraft.position_y_m, demo.aircraft.position_y_m),
      altitude_m: normalizeNumber(aircraft.altitude_m, demo.aircraft.altitude_m),
      roll_rad: normalizeNumber(aircraft.roll_rad, demo.aircraft.roll_rad),
      pitch_rad: normalizeNumber(aircraft.pitch_rad, demo.aircraft.pitch_rad),
      heading_rad: normalizeNumber(aircraft.heading_rad, demo.aircraft.heading_rad),
      airspeed_mps: normalizeNumber(aircraft.airspeed_mps, demo.aircraft.airspeed_mps),
      groundspeed_mps: normalizeNumber(
        aircraft.groundspeed_mps,
        demo.aircraft.groundspeed_mps,
      ),
      vertical_speed_mps: normalizeNumber(
        aircraft.vertical_speed_mps,
        demo.aircraft.vertical_speed_mps,
      ),
      on_ground: normalizeBoolean(aircraft.on_ground, demo.aircraft.on_ground),
      time_s: normalizeNumber(aircraft.time_s, normalizeNumber(session.sim_time_s, 0)),
    },
    mission: {
      name: normalizeString(mission.mission_name ?? mission.name, demo.mission.name),
      waypoints: Array.isArray(mission.waypoints)
        ? mission.waypoints.map((waypoint, index) => ({
            id: normalizeString(
              waypoint.id,
              demo.mission.waypoints[index]?.id ?? `snapshot-${index + 1}`,
            ),
            name: normalizeString(waypoint.name, `Waypoint ${index + 1}`),
            x_m: normalizeNumber(waypoint.x_m, index * 800),
            y_m: normalizeNumber(waypoint.y_m, index * 250),
            altitude_m: normalizeNumber(waypoint.altitude_m, 200 + index * 60),
            target_airspeed_mps: normalizeNumber(
              waypoint.target_airspeed_mps,
              65 + index * 5,
            ),
            acceptance_radius_m: normalizeNumber(waypoint.acceptance_radius_m, 140),
          }))
        : demo.mission.waypoints,
      active_waypoint_index:
        typeof mission.active_waypoint_index === 'number'
          ? mission.active_waypoint_index
          : demo.mission.active_waypoint_index,
      desired_track_rad: normalizeNumber(
        mission.desired_track_rad,
        demo.mission.desired_track_rad ?? 0,
      ),
      distance_to_waypoint_m: normalizeNumber(
        mission.distance_to_waypoint_m,
        demo.mission.distance_to_waypoint_m ?? 0,
      ),
    },
    runtime_health: {
      command_status: normalizeString(
        runtimeHealth.last_command_status ?? runtimeHealth.command_status,
        demo.runtime_health.command_status,
      ),
      last_error:
        typeof runtimeHealth.last_error === 'string'
          ? runtimeHealth.last_error
          : demo.runtime_health.last_error,
      recent_events: Array.isArray(runtimeHealth.recent_events)
        ? runtimeHealth.recent_events.filter(
            (event): event is string => typeof event === 'string',
          )
        : demo.runtime_health.recent_events,
    },
    trail: trail.map((point) => ({
      x_m: normalizeNumber(point.position_x_m ?? point.x_m, 0),
      y_m: normalizeNumber(point.position_y_m ?? point.y_m, 0),
      altitude_m: normalizeNumber(point.altitude_m, 0),
      time_s: normalizeNumber(point.sim_time_s ?? point.time_s, 0),
    })),
  };
}

export function cloneMissionDraft(mission: MissionDraft): MissionDraft {
  return {
    name: mission.name,
    waypoints: mission.waypoints.map((waypoint) => ({ ...waypoint })),
  };
}

export function missionToPayload(mission: MissionDraft): MissionPayload {
  return {
    name: mission.name,
    waypoints: mission.waypoints.map((waypoint): MissionWaypointPayload => ({
      name: waypoint.name,
      x_m: waypoint.x_m,
      y_m: waypoint.y_m,
      altitude_m: waypoint.altitude_m,
      target_airspeed_mps: waypoint.target_airspeed_mps,
      acceptance_radius_m: waypoint.acceptance_radius_m,
    })),
  };
}

export function updateMissionWaypoint(
  mission: MissionDraft,
  waypointId: string,
  changes: Partial<Omit<MissionWaypoint, 'id'>>,
): MissionDraft {
  return {
    ...mission,
    waypoints: mission.waypoints.map((waypoint) =>
      waypoint.id === waypointId ? { ...waypoint, ...changes } : waypoint,
    ),
  };
}

export function moveMissionWaypoint(
  mission: MissionDraft,
  waypointId: string,
  coords: Pick<MissionWaypoint, 'x_m' | 'y_m'>,
): MissionDraft {
  return updateMissionWaypoint(mission, waypointId, coords);
}

export function reorderMissionWaypoint(
  mission: MissionDraft,
  waypointId: string,
  direction: -1 | 1,
): MissionDraft {
  const currentIndex = mission.waypoints.findIndex((waypoint) => waypoint.id === waypointId);
  if (currentIndex < 0) {
    return mission;
  }

  const targetIndex = currentIndex + direction;
  if (targetIndex < 0 || targetIndex >= mission.waypoints.length) {
    return mission;
  }

  const reordered = [...mission.waypoints];
  const [entry] = reordered.splice(currentIndex, 1);
  if (!entry) {
    return mission;
  }

  reordered.splice(targetIndex, 0, entry);

  return { ...mission, waypoints: reordered };
}

export function deleteMissionWaypoint(
  mission: MissionDraft,
  waypointId: string,
): MissionDraft {
  return {
    ...mission,
    waypoints: mission.waypoints.filter((waypoint) => waypoint.id !== waypointId),
  };
}

export function calculateRouteDistance(mission: MissionDraft): number {
  return mission.waypoints.reduce((distance, waypoint, index) => {
    if (index === 0) {
      return distance;
    }

    const previous = mission.waypoints[index - 1];
    if (!previous) {
      return distance;
    }

    const dx = waypoint.x_m - previous.x_m;
    const dy = waypoint.y_m - previous.y_m;
    return distance + Math.hypot(dx, dy);
  }, 0);
}

export function formatMeters(value: number): string {
  return `${Math.round(value).toLocaleString('en-US')} m`;
}

export function formatSpeed(value: number): string {
  return `${Math.round(value)} m/s`;
}

export function normalizeControllerMode(value: string | null | undefined): ControllerMode {
  return value === 'rl_phase_switched' ? 'rl_phase_switched' : 'pid';
}
