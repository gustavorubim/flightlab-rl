# flightlab-rl

`flightlab-rl` is a macOS-first fixed-wing simulation and reinforcement learning lab for benchmarking classical control and RL across takeoff, landing, and flight-plan-following tasks.

The repository is intentionally headless-first. The flight-dynamics backend owns the aircraft state, while environments, rewards, metrics, controllers, and replay tooling are layered on top of that state.

## What is implemented

The initial scaffold includes:

* a deterministic fixed-wing dynamics interface with a lightweight headless backend
* an optional JSBSim adapter for real FDM integration
* Gymnasium environments for takeoff, landing, and flight-plan following
* explicit task phases and decomposed rewards
* waypoint/runway guidance utilities
* a PID baseline controller
* benchmark metrics, replay export, and CLI entrypoints
* a `pytest` + `ruff` + coverage toolchain built around `uv`

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev,sim,rl]
ruff check .
ruff format .
pytest --cov=src/flightlab --cov-report=term-missing --cov-fail-under=95
```

If you only want the lightweight deterministic backend, `uv pip install -e .[dev]` is sufficient.

## Example commands

Run a short benchmark:

```bash
python scripts/benchmark.py --task flight_plan --episodes 5 --seed 7
```

Export a replay:

```bash
python scripts/export_replay.py --task takeoff --seed 42 --steps 120 --output replays/takeoff.json
```

Train with Stable-Baselines3 after installing the `rl` extra:

```bash
python scripts/train.py --algorithm ppo --task flight_plan --timesteps 2000 --seed 42
```

## Package layout

The code follows the architecture defined in [SPEC.md](/Users/gvrubim/Desktop/omscs/flightlab-rl/SPEC.md):

* `src/flightlab/core`: shared types, units, geometry, seeding, clocks
* `src/flightlab/dynamics`: dynamics interfaces and backends
* `src/flightlab/envs`: Gymnasium environments
* `src/flightlab/tasks`: reward and phase logic
* `src/flightlab/guidance`: runway and waypoint guidance
* `src/flightlab/controllers`: classical controller baselines
* `src/flightlab/sensors`: observation generation
* `src/flightlab/world`: missions, runways, and geometry
* `src/flightlab/render`: replay and debug-oriented visualization helpers
* `src/flightlab/rl`: RL integration utilities
* `src/flightlab/metrics`: benchmark and safety metrics

## Notes

The deterministic backend is intended for fast tests, headless debugging, and development before wiring a full JSBSim aircraft definition. The JSBSim adapter is available as the authoritative backend path when the `sim` extra is installed.
