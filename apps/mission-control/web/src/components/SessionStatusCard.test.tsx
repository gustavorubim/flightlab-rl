import { render, screen } from '@testing-library/react';

import { createDemoSnapshot } from '@/lib/mission';
import { SessionStatusCard } from '@/components/SessionStatusCard';

describe('SessionStatusCard', () => {
  it('renders a stale banner when telemetry is delayed', () => {
    render(
      <SessionStatusCard
        snapshot={createDemoSnapshot()}
        connectionStatus="stale"
        lastUpdatedAt={Date.now() - 15_000}
        error="Telemetry stream delayed"
      />,
    );

    expect(screen.getByText('STALE')).toBeInTheDocument();
    expect(screen.getByText(/telemetry stream delayed/i)).toBeInTheDocument();
  });
});
