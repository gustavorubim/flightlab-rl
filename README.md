# flightlab-rl

`flightlab-rl` is a macOS-first fixed-wing simulation and reinforcement learning lab for benchmarking classical control and RL across takeoff, landing, and flight-plan-following tasks.

The repository is intentionally headless-first. The flight-dynamics backend owns the aircraft state, while environments, rewards, metrics, controllers, and replay tooling are layered on top of that state.

## Project intent

This repo is a benchmark and experimentation platform, not a game engine.

It is organized around:

* fixed-wing flight control
* route and runway guidance
* explicit task phases
* reproducible evaluation
* fast headless rollouts
* side-by-side comparison of classical controllers and RL algorithms

The current default runtime path uses a deterministic lightweight dynamics backend for fast local iteration. An optional JSBSim adapter is present in the codebase, but the built-in CLI workflows currently run on the headless backend.

## Current status

The repository currently provides:

* Gymnasium environments for `flight_plan`, `takeoff`, and `landing`
* explicit task phases for takeoff and landing
* decomposed reward breakdowns in `info`
* deterministic resets and rollout support
* a PID baseline evaluation path
* benchmark aggregation and replay export
* Stable-Baselines3 training hooks for PPO and SAC
* a terminal-oriented render path

Current limitations to be aware of:

* there is not yet a full interactive 3D viewer
* replay visualization is currently JSON export plus terminal traces
* the training CLI exercises SB3 integration but does not yet save checkpoints or models to disk
* the JSBSim adapter exists as an integration surface, but backend selection is not yet exposed through the built-in scripts

## Repository layout

The code follows the architecture defined in [SPEC.md](/Users/gvrubim/Desktop/omscs/flightlab-rl/SPEC.md):

* `src/flightlab/core`: shared types, units, geometry, seeding, clocks
* `src/flightlab/dynamics`: dynamics interfaces and backends
* `src/flightlab/envs`: Gymnasium environments
* `src/flightlab/tasks`: reward logic, terminations, and explicit phases
* `src/flightlab/guidance`: waypoint, route, approach, and runway logic
* `src/flightlab/controllers`: classical controller baselines
* `src/flightlab/sensors`: observation generation
* `src/flightlab/world`: missions, runways, and geometry
* `src/flightlab/render`: replay and debug-oriented visualization helpers
* `src/flightlab/rl`: Stable-Baselines3 training integration
* `src/flightlab/metrics`: benchmark and safety summaries
* `configs/`: example aircraft, task, mission, weather, and training configuration files
* `scripts/`: training, evaluation, playback, replay export, and benchmarking entrypoints
* `tests/`: unit, integration, and regression coverage

## Environment API

All built-in tasks expose a Gymnasium-style API:

```python
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

The default action vector is:

* `elevator`
* `aileron`
* `rudder`
* `throttle`

The `info` dictionary includes:

* `reward_breakdown`
* `task_phase`
* `safety_flags`
* `stall_risk`
* `cross_track_error_m`
* `altitude_error_m`
* `touchdown_metrics` when applicable

## Built-in tasks

### `flight_plan`

The aircraft follows an ordered list of 3D waypoints while minimizing:

* cross-track error
* altitude error
* speed error
* heading error

The task also includes waypoint bonuses and mission completion bonuses.

### `takeoff`

The aircraft starts on the runway and progresses through:

* `TAXI_ALIGN`
* `TAKEOFF_ROLL`
* `ROTATE`
* `INITIAL_CLIMB`

The reward emphasizes runway tracking, heading alignment, speed buildup, rotation timing, and safe initial climb.

### `landing`

The aircraft starts on approach and progresses through:

* `APPROACH`
* `FINAL`
* `FLARE`
* `TOUCHDOWN`
* `ROLLOUT`

The reward emphasizes runway alignment, glideslope tracking, flare quality, touchdown quality, and rollout safety.

## Installation

### Recommended setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

Install optional extras as needed:

```bash
uv pip install -e '.[dev,rl]'
uv pip install -e '.[dev,sim]'
uv pip install -e '.[dev,rl,sim]'
```

If you use `zsh`, keep the extras spec quoted like `'.[dev,rl,sim]'` or escape the brackets. Otherwise the shell will try to expand them before `uv` sees the argument.

### Interpreter note

Use the repository virtualenv, not a global Python installation.

If your shell `python` points to Conda or a system interpreter, prefer explicit commands like:

```bash
./.venv/bin/python scripts/benchmark.py --task flight_plan --episodes 5 --seed 7
```

This avoids importing unrelated global packages and keeps behavior reproducible.

## Development commands

```bash
./.venv/bin/ruff check .
./.venv/bin/ruff format --check .
./.venv/bin/pytest
```

## Running the tasks

### 1. Run a benchmark

This runs multiple episodes with a simple fixed command policy and reports aggregate metrics:

```bash
./.venv/bin/python scripts/benchmark.py --task flight_plan --episodes 5 --seed 7
./.venv/bin/python scripts/benchmark.py --task takeoff --episodes 5 --seed 7
./.venv/bin/python scripts/benchmark.py --task landing --episodes 5 --seed 7
```

The output is a `BenchmarkSummary` containing:

* `success_rate`
* `crash_rate`
* `stall_rate`
* `runway_excursion_rate`
* `average_cross_track_error_m`
* `altitude_rmse_m`
* `average_action_smoothness`
* `average_return`
* `average_completion_time_s`

### 2. Evaluate the PID baseline

This runs the built-in PID autopilot against one task and prints an episode summary:

```bash
./.venv/bin/python scripts/eval.py --task flight_plan --seed 42 --steps 250
./.venv/bin/python scripts/eval.py --task takeoff --seed 42 --steps 250
./.venv/bin/python scripts/eval.py --task landing --seed 42 --steps 250
```

This is the current built-in “agent evaluation” path that uses an actual controller instead of a fixed action vector.

### 3. Play a rollout in the terminal

This prints one compact render line per step:

```bash
./.venv/bin/python scripts/play.py --task flight_plan --seed 42 --steps 100
```

Example render output looks like:

```text
time=3.2s task=flight_plan phase=ENROUTE pos=(145.2,-11.8,138.4) speed=25.1 reward=0.214
```

This is useful for:

* smoke-testing environment dynamics
* checking task phase transitions
* quickly inspecting whether a policy is obviously unstable

### 4. Export a replay

You can export a JSON replay for later analysis:

```bash
./.venv/bin/python scripts/export_replay.py --task takeoff --seed 42 --steps 120 --output replays/takeoff.json
./.venv/bin/python scripts/export_replay.py --task landing --seed 42 --steps 180 --output replays/landing.json
```

Each replay contains:

* a `reset` record with initial state and info
* one `step` record per transition
* the aircraft state
* the action applied
* the reward
* the per-step `info` payload

This is the current best path for offline rendering, plotting, and debugging.

## RL training

### Supported algorithms

The built-in training hook supports:

* `ppo`
* `sac`

Install the RL extra first:

```bash
uv pip install --python .venv/bin/python -e '.[dev,rl]'
```

Then run:

```bash
./.venv/bin/python scripts/train.py --algorithm ppo --task flight_plan --timesteps 2000 --seed 42
./.venv/bin/python scripts/train.py --algorithm sac --task flight_plan --timesteps 2000 --seed 42
./.venv/bin/python scripts/train.py --algorithm ppo --task takeoff --timesteps 2000 --seed 42
./.venv/bin/python scripts/train.py --algorithm ppo --task landing --timesteps 2000 --seed 42
```

### What the training script does today

The current training CLI is an integration hook around Stable-Baselines3. It:

* creates a built-in environment
* constructs a PPO or SAC model
* runs `learn(total_timesteps=...)`
* prints a `TrainingResult`

It does not yet:

* save checkpoints
* save the final model
* provide an out-of-the-box “play a trained model” command

So today the training script is most useful for:

* verifying that the RL stack is wired correctly
* quick smoke tests on a task
* iterating on environment interfaces and reward design

If you want persistent checkpoints next, the right place to extend is [baselines.py](/Users/gvrubim/Desktop/omscs/flightlab-rl/src/flightlab/rl/baselines.py) and [train.py](/Users/gvrubim/Desktop/omscs/flightlab-rl/scripts/train.py).

## How to visualize and render an agent

### Current render path

At the moment, rendering is debug-oriented rather than graphical:

* `env.render()` returns a compact text summary
* `scripts/play.py` prints those summaries live
* `scripts/export_replay.py` writes a deterministic JSON replay
* `src/flightlab/render/replay.py` is the main replay capture/export utility

There is not yet a built-in 3D viewer or desktop playback app.

### Visualizing a custom policy

If you already have a policy object in memory, you can roll it out directly and print the render trace:

```python
from flightlab.envs import make_env

env = make_env("flight_plan", seed=42)
obs, info = env.reset(seed=42)

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(env.render())
    if terminated or truncated:
        break

env.export_replay("replays/flight_plan_agent.json")
```

That pattern works for any action source:

* an SB3 policy
* a rule-based controller
* a PID autopilot
* a hand-written scripted policy

### Visualizing with the built-in controller

The repo already includes one complete controller-driven path:

```bash
./.venv/bin/python scripts/eval.py --task flight_plan --seed 42 --steps 250
```

That runs the PID autopilot and prints the final summary. If you want live per-step traces from the controller rather than the fixed-action `play.py` path, the easiest extension is to adapt [eval.py](/Users/gvrubim/Desktop/omscs/flightlab-rl/scripts/eval.py) so it prints `env.render()` inside the step loop.

### Working with replay files

Replay files are straightforward JSON, which makes them easy to:

* load in a notebook
* plot with `matplotlib`
* compare between policies
* inspect reward breakdowns and safety flags frame by frame

A typical next step is to build a notebook or small viewer that reads `replays/*.json` and plots:

* altitude over time
* airspeed over time
* cross-track error over time
* control commands over time
* reward components over time

## Programmatic use

You can also use the environments directly from Python:

```python
import numpy as np

from flightlab.envs import make_env

env = make_env("takeoff", seed=123)
obs, info = env.reset(seed=123)

for _ in range(50):
    action = np.asarray([0.0, 0.0, 0.0, 0.7], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(info)
```

## Testing and quality gates

The repository is set up for:

* `pytest`
* `ruff`
* enforced coverage thresholds
* deterministic regression tests

Run the full local quality gate with:

```bash
./.venv/bin/ruff check .
./.venv/bin/ruff format --check .
./.venv/bin/pytest
```

## Notes

The deterministic backend is intended for fast tests, headless debugging, and early task development. The JSBSim adapter is present so the repo can grow toward an authoritative flight-dynamics backend without changing the surrounding environment/task architecture.
