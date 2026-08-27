# SWEHB

SWEHB stands for _Shallow Water Equations Herschel-Bulkley_, which models the free-surface flows of viscoplastic fluids described by the HB rheology over arbitrary bathymetries. This is a work in progress.

---

## Installation

Clone the repository
```bash
git clone https://github.com/username/swehb.git
cd swehb
```

Create a virtual environment (recommended)
```bash
python3 -m venv .venv
```

Activate it

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```powershell
.venv\Scripts\activate
```

Install the required packages
```bash
pip install -r swehb_reqs.txt
```
---

## Project files

### `solver.py`
Contains the `SWEHBSolver` class, i.e. the numerical implementation of the fractional-step method used to solve the shallow water equations with the Herschel-Bulkley rheology. This is the file you `import`, not the one you usually edit. Its public interface is:

- `SWEHBSolver(params, surge=False, reservoir=False, roll=False)`: builds the solver. `params` is a dict with the fluid/geometry/numerical parameters (see below); missing/invalid values raise a `ValueError`, and risky-but-valid setups (e.g. `CFL > 1`) raise a `warnings.warn`.
  - `surge`: enables the dam-break/surge scenario instead of a steady-state inflow.
  - `reservoir`: with `surge=True`, uses a finite reservoir volume instead of an infinite step dam-break.
  - `roll`: enables an oscillating inflow boundary condition (roll waves).
- `set_bathymetry(bath_type, center=0, contour_file=None, a=1)`: builds one of the profiles defined in `bathymetry.py` and stores it on the solver (see that file for the available `bath_type` options, including loading a custom contour or editing your own profile).
- `set_roll_wave(amp=None, freq=None, u0=None)`: customizes the roll-wave perturbation (amplitude, frequency, base velocity) when `roll=True`.
- `check_case(zeta_file=None, u_file=None, show=True)`: builds the initial condition from the current bathymetry and plots bathymetry, initial depth and initial velocity, so you can visually validate the setup before running.
- `run(...)`: runs the simulation and returns a `results` dict (free surface, depth, velocity, dimensionless numbers, etc.). Key arguments: `max_iter`, `plot_interval`, `probes` (x-locations to record as time series), `center`/`norm` (used to normalize plots), `grad` (enables the full Λ model), `live_plot` (enables real-time plotting during the run).

`params` dictionary keys:

| Key | Meaning |
|---|---|
| `h0` | Reference (uniform) flow depth |
| `ty` | Yield stress |
| `kn` | Consistency index |
| `m` | Flow behavior index |
| `rho` | Fluid density |
| `theta` | Slope angle (radians) |
| `Nx` | Number of grid cells |
| `CFL` | CFL number for the time step |
| `tend` | Simulation end time |
| `M` | Characteristic obstacle/normalization length scale |
| `xL`, `xR` | Domain limits |

### `bathymetry.py`
Holds the bathymetry profile definitions, kept separate from the solver so you can inspect/extend them without touching the numerical code.

- `build_bathymetry(xb, M, bath_type, center=0, a=1, contour_file=None)`: called internally by `solver.set_bathymetry`. Supported `bath_type` values: `'dead_zones'`, `'rectangle'`, `'squared_trapezoid'`, `'semi_circular'`, `'bump'`, `'ramp'`, `'flat'`/`'none'`, `'sinusoidal'`, `'contour'` (loads a two-column `x,z` CSV via `contour_file`), and `'custom'`.
- `custom_bathymetry(xb, M, center=0, a=1)`: edit this function directly to design your own bathymetry, then call `solver.set_bathymetry('custom')` to use it.

### `case_call.py`
The main user-facing entry point. Define the fluid rheology and geometry parameters here, choose/build a bathymetry, visually check the case, then run the simulation:

```python
solver = SWEHBSolver(params={...})
solver.set_bathymetry("rectangle", center=-M/2)
solver.check_case()          # visually inspect bathymetry/initial conditions
results = solver.run(max_iter=80001, plot_interval=5, live_plot=True)
```

Results are stored in a dictionary and optionally saved to a pickle (`.pkl`) file when prompted at the end of the run.

### `pkl_to_csv.py`
Utility script to convert a saved `.pkl` results file into `.csv` files (one per case) under a timestamped `RESULTS/` folder, for further analysis outside Python.

## How to use it

1. Edit `case_call.py`: set the fluid rheology (`ty`, `kn`, `m`, `rho`), geometry (`h0`, `theta`, `M`), and numerical parameters (`Nx`, `CFL`, `xL`, `xR`, `tend`).
2. Choose a bathymetry with `solver.set_bathymetry(...)`, or add your own profile in `bathymetry.py`.
3. Call `solver.check_case()` to confirm the bathymetry and initial conditions look right before running.
4. Call `solver.run(...)` to run the simulation (set `live_plot=True` to watch the solution evolve in real time).
5. Save the results to a `.pkl` file when prompted, then use `pkl_to_csv.py` if you need `.csv` output.

---

This was developed during my master's degree at the ReoSul lab, from the Graduate Program in Mechanical Engineering at the Federal University of Rio Grande do Sul (UFRGS), in Porto Alegre Brazil.
