# V2X Intersection Safety Simulator

A 2-D Pygame simulation of an intersection with Vehicle-to-Everything (V2X)
communication, machine-learning-based GO / STOP decisions, and configurable
safety policies.

---

## Quick Start

```bash
# 1. Create a virtual environment (Python 3.10+)
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (First time only) Generate training data & train the model
python ml/learn/GenerateData.py
python ml/learn/Train.py

# 4. Run the simulator
python main.py
```

A window opens with a launch screen — press **START SIMULATION** or hit
**Enter** to begin.

---

## Controls

| Key / Mouse        | Action                         |
|---------------------|-------------------------------|
| **Space**           | Pause / Resume                |
| **+** / **−**       | Zoom in / out                 |
| **Arrow keys**      | Pan camera                    |
| **Scroll wheel**    | Zoom toward cursor            |
| **Right-drag**      | Pan camera (mouse)            |
| **R**               | Reset scenario                |
| **F12**             | Save screenshot               |

---

## Environment Variables

All optional — sensible defaults are built in.

| Variable                     | Default | Description                              |
|------------------------------|---------|------------------------------------------|
| `SIM_VEHICLE_COUNT`          | `6`     | Number of cars to spawn                  |
| `SIM_TICK_HZ`               | `20.0`  | Simulation ticks per second              |
| `SIM_RANDOM_SEED`           | —       | Seed for reproducibility                 |
| `SIM_PRIORITY_AXIS`         | `EW`    | `EW` or `NS` — which road gets priority |
| `SIM_WORLD_SIGNAL`          | `false` | Enable the virtual signal scheduler      |
| `SIM_WORLD_COLLISION_GUARD` | `false` | Enable pair-wise collision guard         |
| `SIM_WORLD_OVERLAP_RESOLVER`| `true`  | Last-resort push-apart for overlaps      |

---

## Project Structure

```
.
├── main.py                  Entry point
├── config.py                Application-wide defaults
├── logging_setup.py         Console + rotating-file logger
├── test_ml.py               Quick ML smoke test
├── requirements.txt         Minimal pip dependencies
│
├── sim/                     Simulation core
│   ├── world.py             Car entities, physics, stop-sign enforcement
│   ├── traffic_policy.py    SafetyPolicy dataclass & scoring helpers
│   ├── sim_bridge.py        Background thread orchestrator
│   ├── physics.py           Low-level conversion & distance helpers
│   └── test_collision_safety.py  Pytest collision-guard tests
│
├── bus/                     In-memory V2X message bus
│   ├── v2x_bus.py           Pub/sub transport with drop & latency
│   ├── message.py           V2XMessage dataclass
│   ├── metrics.py           BusMetrics counter snapshot
│   └── utils.py             ID generation, latency, fault injection
│
├── ml/                      Machine learning pipeline
│   ├── comunication/
│   │   ├── Inference.py     Run trained model on a traffic-state dict
│   │   └── api.py           Optional FastAPI REST server
│   ├── entities/
│   │   ├── Car.py           Lightweight ML car model
│   │   ├── Directions.py    LEFT / RIGHT / FORWARD enum
│   │   ├── Sign.py          STOP / YIELD / PRIORITY / NO_SIGN enum
│   │   └── Intersections.py Feature-vector extraction (59 floats)
│   ├── learn/
│   │   ├── GenerateData.py  Synthetic dataset generator
│   │   ├── Train.py         Random Forest trainer
│   │   └── Test.py          Validation-set evaluator
│   └── generated/           Model artefacts (CSV, .pkl, reports)
│
├── ui/                      Pygame renderer
│   ├── pygame_view.py       Main loop, launch screen, events
│   ├── draw_road.py         Map background, signs, decorations
│   ├── draw_vehicles.py     Car sprites, route lines, awareness rings
│   ├── hud.py               HUD panels, control bar, legend
│   ├── helpers.py           Coordinate transforms, interpolation
│   ├── constants.py         All colours, dimensions, font sizes
│   └── types.py             Camera, VehicleSnapshot, ButtonRect
│
├── docs/                    Sphinx documentation sources
└── screenshots/             Saved screenshots (F12)
```

---

## Architecture

```
 main.py
   │
   ├─ SimBridge (background thread)
   │     │
   │     ├─ World          — spawn cars, physics tick, stop-sign enforcement
   │     ├─ Inference.py   — ML GO/STOP per car (Random Forest)
   │     └─ V2XBus         — pub/sub with simulated packet loss
   │            │
   │            ├─ v2v.state     (each car broadcasts position)
   │            └─ i2v.command   (infrastructure publishes ML decisions)
   │
   └─ pygame_view (UI thread)
         │
         ├─ draw_map()            — road, signs, decorations
         ├─ draw_all_vehicles()   — car sprites, routes, awareness
         └─ HUD panels            — clock, vehicle list, controls
```

The **SimBridge** runs at a configurable tick rate (default 20 Hz).
Each tick:

1. Every car broadcasts its V2V state.
2. Infrastructure ML infers GO / STOP for each car.
3. Decisions are published on the I2V bus channel (subject to drop rate).
4. `World.update_physics()` reads bus-delivered decisions, applies stop-sign
   enforcement and optional safety layers, then moves cars.
5. The UI thread polls the bridge at 10 Hz and interpolates between snapshots
   at 60 fps.

---

## ML Pipeline

| Script              | Purpose                         |
|----------------------|---------------------------------|
| `GenerateData.py`   | 8 000 training + 1 500 val scenarios |
| `Train.py`          | Random Forest (25 trees, depth 12)   |
| `Test.py`           | Accuracy report + confidence sample  |

Feature vector (59 floats): ego car (7) + sign one-hot (4) + up to 6
neighbours × 8 features each (48) = **59 total**.

---

## Building Documentation

```bash
cd docs
make html
# open build/html/index.html
```

---

## License

Internal hackathon project — BEST Bucharest.
