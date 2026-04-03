# FlightLab Mission Control Web

This folder contains the standalone frontend for the local mission-control UI.

## What it does

- split command rail, tactical mission planner, and 3D scene
- live telemetry over `GET /api/session` plus `WS /ws/telemetry`
- click-to-add waypoint planning with drag edit, reorder, delete, and commit
- camera modes for orbit and chase views in the 3D panel

## Install

```bash
cd apps/mission-control/web
npm install
```

## Run locally

```bash
python3 ../run.py
```

If you only want the frontend process and already have the API running:

```bash
npm run dev
```

The Vite dev server proxies:

- `/api` to `http://localhost:8000`
- `/ws` to `ws://localhost:8000`

## Test

```bash
npm run test
```

## Build

```bash
npm run build
```

## Notes

- The app is resilient to a missing backend and will render a demo mission state.
- The backend contract it targets is documented in the parent mission-control design.
- The visual language is intentionally dark, tactical, and operator-centric.
