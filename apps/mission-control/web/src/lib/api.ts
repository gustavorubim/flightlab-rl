import type {
  BackendControllerOption,
  BackendSessionSnapshot,
  ControllerOption,
  ControllerMode,
  MissionPayload,
  SessionSnapshot,
} from '@/types';
import {
  normalizeControllerMode,
  normalizeControllerOptions,
  normalizeSessionSnapshot,
} from '@/lib/mission';

function apiBaseUrl(): string {
  const configured = import.meta.env.VITE_MISSION_CONTROL_API_URL;
  if (typeof configured === 'string' && configured.length > 0) {
    return configured.replace(/\/$/, '');
  }
  return '';
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl()}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const message = await response.text().catch(() => response.statusText);
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

function wsProtocol(): 'ws:' | 'wss:' {
  return window.location.protocol === 'https:' ? 'wss:' : 'ws:';
}

export function telemetrySocketUrl(): string {
  const url = new URL('/ws/telemetry', apiBaseUrl() || window.location.origin);
  url.protocol = wsProtocol();
  return url.toString();
}

export async function fetchSessionSnapshot(): Promise<SessionSnapshot> {
  const raw = await requestJson<BackendSessionSnapshot>('/api/session');
  return normalizeSessionSnapshot(raw);
}

export async function fetchControllerOptions(): Promise<ControllerOption[]> {
  const raw = await requestJson<unknown>('/api/controllers');
  return normalizeControllerOptions(raw);
}

export async function startSession(request: {
  controllerMode: ControllerMode;
  mission?: MissionPayload;
  runway?: unknown;
}): Promise<void> {
  const body: Record<string, unknown> = {
    controller_mode: request.controllerMode,
  };

  if (request.mission) {
    body.mission = request.mission;
  }

  if (request.runway) {
    body.runway = request.runway;
  }

  await requestJson<void>('/api/session/start', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function resetSession(): Promise<void> {
  await requestJson<void>('/api/session/reset', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function triggerTakeoff(): Promise<void> {
  await requestJson<void>('/api/commands/takeoff', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function pauseSession(): Promise<void> {
  await requestJson<void>('/api/commands/pause', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function resumeSession(): Promise<void> {
  await requestJson<void>('/api/commands/resume', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function commitMission(mission: MissionPayload): Promise<void> {
  await requestJson<void>('/api/mission', {
    method: 'PUT',
    body: JSON.stringify(mission),
  });
}

export { normalizeControllerMode };
