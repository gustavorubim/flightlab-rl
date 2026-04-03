export type ControllerMode = 'pid' | 'rl_phase_switched';

export type ConnectionStatus = 'connecting' | 'live' | 'stale' | 'offline';

export interface ControllerOption {
  mode: ControllerMode;
  label: string;
  description: string;
  enabled: boolean;
}

export interface MissionWaypoint {
  id: string;
  name: string;
  x_m: number;
  y_m: number;
  altitude_m: number;
  target_airspeed_mps: number;
  acceptance_radius_m: number;
}

export interface MissionDraft {
  name: string;
  waypoints: MissionWaypoint[];
}

export interface MissionWaypointPayload {
  name: string;
  x_m: number;
  y_m: number;
  altitude_m: number;
  target_airspeed_mps: number;
  acceptance_radius_m: number;
}

export interface MissionPayload {
  name: string;
  waypoints: MissionWaypointPayload[];
}

export interface SessionInfo {
  session_id: string;
  phase: string;
  controller_mode: string;
  sim_time_s: number;
  paused: boolean;
}

export interface AircraftState {
  position_x_m: number;
  position_y_m: number;
  altitude_m: number;
  roll_rad: number;
  pitch_rad: number;
  heading_rad: number;
  airspeed_mps: number;
  groundspeed_mps: number;
  vertical_speed_mps: number;
  on_ground: boolean;
  time_s: number;
}

export interface MissionState {
  name: string;
  waypoints: MissionWaypoint[];
  active_waypoint_index: number | null;
  desired_track_rad: number | null;
  distance_to_waypoint_m: number | null;
}

export interface RuntimeHealth {
  command_status: string;
  last_error: string | null;
  recent_events: string[];
}

export interface TrailPoint {
  x_m: number;
  y_m: number;
  altitude_m: number;
  time_s: number;
}

export interface SessionSnapshot {
  session: SessionInfo;
  aircraft: AircraftState;
  mission: MissionState;
  runtime_health: RuntimeHealth;
  trail: TrailPoint[];
}

export interface BackendControllerOption {
  mode?: string;
  label?: string;
  description?: string;
  available?: boolean;
  enabled?: boolean;
}

export interface BackendSessionInfo {
  session_id?: string;
  phase?: string;
  controller_mode?: string;
  sim_time_s?: number;
  paused?: boolean;
}

export interface BackendAircraftState {
  position_x_m?: number;
  position_y_m?: number;
  altitude_m?: number;
  roll_rad?: number;
  pitch_rad?: number;
  heading_rad?: number;
  airspeed_mps?: number;
  groundspeed_mps?: number;
  vertical_speed_mps?: number;
  on_ground?: boolean;
  time_s?: number;
}

export interface BackendMissionState {
  mission_name?: string;
  name?: string;
  waypoints?: Array<Partial<MissionWaypointPayload> & { id?: string }>;
  active_waypoint_index?: number | null;
  desired_track_rad?: number | null;
  distance_to_waypoint_m?: number | null;
}

export interface BackendRuntimeHealth {
  last_command_status?: string;
  command_status?: string;
  last_error?: string | null;
  recent_events?: string[];
}

export interface BackendTrailPoint {
  position_x_m?: number;
  position_y_m?: number;
  x_m?: number;
  y_m?: number;
  altitude_m?: number;
  sim_time_s?: number;
  time_s?: number;
}

export interface BackendSessionSnapshot {
  session?: BackendSessionInfo;
  aircraft?: BackendAircraftState;
  mission?: BackendMissionState;
  runtime_health?: BackendRuntimeHealth;
  trail?: BackendTrailPoint[];
  runway?: {
    name?: string;
    length_m?: number;
    width_m?: number;
    heading_rad?: number;
    threshold_x_m?: number;
    threshold_y_m?: number;
    elevation_m?: number;
  };
}
