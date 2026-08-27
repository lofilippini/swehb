import numpy as np
import pickle
from solver import SWEHBSolver

# Define parameters
# Fluid rheology
ty = 10
kn = 0.7
m = 0.6
rho = 1011

h0 = 0.01535
theta = np.radians(7)
M = 0.025

Nx = 1000
CFL = 0.9
xL = -M*30
xR = M*30
tend = 1000
filename = 'gomit_square'

max_iter = 80001
# max_iter = 100

# Store all results in a dictionary
all_results = {}
ctr = 0
probe_locs = None


'''
Here you call the solver. You can select the flow case by setting:
surge = True or False, to enable or disable the surge scenario (dam-break with constant discharge)
reservoir = True or False, to enable or disable the dam-break scenario (finite volume of fluid)
rollwaves = True or False, to enable or disable roll wave boundary condition (requires frequency and amplitude adjustments in the source code)

The code will always run a steady-state case, otherwise.
'''
solver = SWEHBSolver(params = {
    'h0': h0,
    'ty': ty,
    'kn': kn,
    'm': m,
    'rho': rho,
    'theta': theta,
    'Nx': Nx,
    'CFL': CFL,
    'tend': tend,
    'M': M,
    'xL': xL,
    'xR': xR,
})


# To change roll wave boundary conditions, uncomment the following section:

solver.set_roll_wave(
    amp=0.05,   # 2% velocity perturbation
    freq=1.5,   # oscillations per second
    u0=None,    # defaults to computed steady inlet velocity
)


solver.set_bathymetry("bump")

# Graphically check the bathymetry and initial conditions before running
solver.check_case()

# Run the simulation and store the results
# Comment these whole section if you just want to check the initial conditions or bathymetry
all_results[f'{filename}'] = solver.run()


save_confirm = input("Save file? (y/n): ").strip().lower()
if save_confirm in ("y", "yes"):
    with open(f'./{filename}.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print("\nAll simulations complete!")
    print(f"Results saved to '{filename}.pkl'")
    print(f"Available geometries: {list(all_results.keys())}")
    print(f"{'='*40}")
else:
    print("Save cancelled.")
