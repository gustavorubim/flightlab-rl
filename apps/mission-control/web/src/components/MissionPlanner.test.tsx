import { useState } from 'react';

import { fireEvent, render, screen } from '@testing-library/react';

import {
  createDefaultMissionDraft,
  createWaypointDraft,
  deleteMissionWaypoint,
  moveMissionWaypoint,
  reorderMissionWaypoint,
  updateMissionWaypoint,
} from '@/lib/mission';
import type { MissionDraft, MissionWaypoint } from '@/types';
import { MissionPlanner } from '@/components/MissionPlanner';

function PlannerHarness({ onCommitRoute }: { onCommitRoute: () => void }) {
  const [mission, setMission] = useState<MissionDraft>(createDefaultMissionDraft());
  const [selectedWaypointId, setSelectedWaypointId] = useState<string | null>(
    mission.waypoints[0]?.id ?? null,
  );

  function handleAddWaypoint(coords: Pick<MissionWaypoint, 'x_m' | 'y_m'>): string {
    const waypoint = createWaypointDraft(mission.waypoints.length, coords);
    setMission((current: MissionDraft) => ({
      ...current,
      waypoints: [...current.waypoints, waypoint],
    }));
    setSelectedWaypointId(waypoint.id);
    return waypoint.id;
  }

  return (
    <MissionPlanner
      mission={mission}
      selectedWaypointId={selectedWaypointId}
      onSelectWaypoint={setSelectedWaypointId}
      onAddWaypoint={handleAddWaypoint}
      onMoveWaypoint={(waypointId, coords) =>
        setMission((current: MissionDraft) => moveMissionWaypoint(current, waypointId, coords))
      }
      onUpdateWaypoint={(waypointId, changes) =>
        setMission((current: MissionDraft) => updateMissionWaypoint(current, waypointId, changes))
      }
      onReorderWaypoint={(waypointId, direction) =>
        setMission((current: MissionDraft) => reorderMissionWaypoint(current, waypointId, direction))
      }
      onDeleteWaypoint={(waypointId) =>
        setMission((current: MissionDraft) => deleteMissionWaypoint(current, waypointId))
      }
      onCommitRoute={onCommitRoute}
      hasUnsavedRoute
    />
  );
}

describe('MissionPlanner', () => {
  it('supports click-to-add, drag-to-move, editing, and commit', () => {
    const onCommitRoute = vi.fn();
    render(<PlannerHarness onCommitRoute={onCommitRoute} />);

    const field = screen.getByTestId('planner-field');
    fireEvent.click(field, { clientX: 820, clientY: 260 });

    expect(screen.getByRole('button', { name: 'Waypoint 4' })).toBeInTheDocument();

    const altitudeInput = screen.getByLabelText('Altitude (m)') as HTMLInputElement;
    fireEvent.change(altitudeInput, { target: { value: '1250' } });
    expect(altitudeInput.value).toBe('1250');

    const speedInput = screen.getByLabelText('Target speed (m/s)') as HTMLInputElement;
    fireEvent.change(speedInput, { target: { value: '92' } });
    expect(speedInput.value).toBe('92');

    const xBefore = screen.getByTestId('selected-waypoint-x').textContent;
    const selectedMarker = screen.getByTestId('selected-waypoint-marker');
    fireEvent.mouseDown(selectedMarker, { clientX: 820, clientY: 260 });
    fireEvent.mouseMove(field, { clientX: 980, clientY: 180 });
    fireEvent.mouseUp(field);

    expect(screen.getByTestId('selected-waypoint-x').textContent).not.toBe(xBefore);

    fireEvent.click(screen.getByRole('button', { name: /delete waypoint 4/i }));
    expect(screen.queryByRole('button', { name: 'Waypoint 4' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /commit route/i }));
    expect(onCommitRoute).toHaveBeenCalledTimes(1);
  });
});
