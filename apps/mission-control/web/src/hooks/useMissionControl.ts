import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  commitMission,
  fetchControllerOptions,
  fetchSessionSnapshot,
  pauseSession,
  resetSession,
  resumeSession,
  startSession,
  telemetrySocketUrl,
  triggerTakeoff,
} from '@/lib/api';
import {
  createDemoSnapshot,
  normalizeControllerOptions,
  normalizeSessionSnapshot,
} from '@/lib/mission';
import type {
  ConnectionStatus,
  ControllerMode,
  ControllerOption,
  MissionPayload,
  SessionSnapshot,
} from '@/types';

export interface MissionControlCommands {
  startSession: (args: {
    controllerMode: ControllerMode;
    mission?: MissionPayload;
    runway?: unknown;
  }) => Promise<void>;
  resetSession: () => Promise<void>;
  takeoff: () => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  commitMission: (mission: MissionPayload) => Promise<void>;
  refreshSession: () => Promise<void>;
}

export interface MissionControlState {
  snapshot: SessionSnapshot;
  controllers: ControllerOption[];
  connectionStatus: ConnectionStatus;
  lastUpdatedAt: number;
  error: string | null;
}

export function useMissionControl(): MissionControlState & MissionControlCommands {
  const [snapshot, setSnapshot] = useState<SessionSnapshot>(createDemoSnapshot());
  const [controllers, setControllers] = useState<ControllerOption[]>(
    normalizeControllerOptions([]),
  );
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>('connecting');
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number>(Date.now());
  const [error, setError] = useState<string | null>(null);

  const refreshSession = useCallback(async () => {
    try {
      const next = await fetchSessionSnapshot();
      setSnapshot(next);
      setConnectionStatus('live');
      setLastUpdatedAt(Date.now());
      setError(null);
    } catch (fetchError) {
      setConnectionStatus((current) => (current === 'live' ? 'stale' : 'offline'));
      setError(fetchError instanceof Error ? fetchError.message : 'Failed to fetch session');
    }
  }, []);

  useEffect(() => {
    let active = true;

    void (async () => {
      try {
        const nextControllers = await fetchControllerOptions();
        if (active) {
          setControllers(nextControllers);
        }
      } catch {
        if (active) {
          setControllers(normalizeControllerOptions([]));
        }
      }
    })();

    void refreshSession();

    return () => {
      active = false;
    };
  }, [refreshSession]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let disposed = false;

    try {
      socket = new WebSocket(telemetrySocketUrl());
    } catch {
      setConnectionStatus((current) => (current === 'live' ? 'stale' : 'offline'));
      return () => undefined;
    }

    socket.addEventListener('open', () => {
      if (!disposed) {
        setConnectionStatus('live');
      }
    });

    socket.addEventListener('message', (event) => {
      if (disposed) {
        return;
      }

      try {
        const raw = JSON.parse(event.data as string) as Parameters<
          typeof normalizeSessionSnapshot
        >[0];
        setSnapshot(normalizeSessionSnapshot(raw));
        setConnectionStatus('live');
        setLastUpdatedAt(Date.now());
        setError(null);
      } catch {
        setConnectionStatus('stale');
      }
    });

    socket.addEventListener('close', () => {
      if (!disposed) {
        setConnectionStatus((current) => (current === 'live' ? 'stale' : current));
      }
    });

    socket.addEventListener('error', () => {
      if (!disposed) {
        setConnectionStatus((current) => (current === 'live' ? 'stale' : current));
      }
    });

    return () => {
      disposed = true;
      socket?.close();
    };
  }, []);

  const startMissionSession = useCallback(
    async (args: {
      controllerMode: ControllerMode;
      mission?: MissionPayload;
      runway?: unknown;
    }) => {
      await startSession(args);
      await refreshSession();
    },
    [refreshSession],
  );

  const resetMissionSession = useCallback(async () => {
    await resetSession();
    await refreshSession();
  }, [refreshSession]);

  const takeoffMission = useCallback(async () => {
    await triggerTakeoff();
    await refreshSession();
  }, [refreshSession]);

  const pauseMission = useCallback(async () => {
    await pauseSession();
    await refreshSession();
  }, [refreshSession]);

  const resumeMission = useCallback(async () => {
    await resumeSession();
    await refreshSession();
  }, [refreshSession]);

  const commitMissionDraft = useCallback(
    async (mission: MissionPayload) => {
      await commitMission(mission);
      await refreshSession();
    },
    [refreshSession],
  );

  return useMemo(
    () => ({
      snapshot,
      controllers,
      connectionStatus,
      lastUpdatedAt,
      error,
      startSession: startMissionSession,
      resetSession: resetMissionSession,
      takeoff: takeoffMission,
      pause: pauseMission,
      resume: resumeMission,
      commitMission: commitMissionDraft,
      refreshSession,
    }),
    [
      commitMissionDraft,
      connectionStatus,
      controllers,
      error,
      lastUpdatedAt,
      pauseMission,
      refreshSession,
      resetMissionSession,
      resumeMission,
      snapshot,
      startMissionSession,
      takeoffMission,
    ],
  );
}
