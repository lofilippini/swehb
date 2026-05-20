import numpy as np
import pickle
from solver import SWEHBSolver

# Define parameters
# Gomit et al. numerical
h0 = 0.01535
ty = 10
kn = 0.7
m = 0.6
rho = 1011
theta = np.radians(7)
Nx = 1000
CFL = 0.9
M = 0.025
alpha = M/h0
xL = -M*30
xR = M*30
tend = 1000
filename = 'gomit_square'

params = {
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
}

def bathymetry_profile(bt, M=params['M'], center = 0, contour_file=None, a = 1):
    global bath_type, h0, alpha
    bath_type = bt
    solver.bathb = np.zeros(solver.Nx)
    
    if  bath_type == 'dead_zones':    
        # Trapezoidal bathymetry parameters
        solver.x_left = -1.5 * M
        solver.x_flat_left = -M / 2
        solver.x_flat_right = M / 2
        solver.x_right = M
        
        mask_left = (solver.xb >= solver.x_left) & (solver.xb < solver.x_flat_left)
        t_left = (solver.xb[mask_left] - solver.x_left) / (solver.x_flat_left - solver.x_left)
        
        mask_flat = (solver.xb >= solver.x_flat_left) & (solver.xb <= solver.x_flat_right)
        
        mask_right = (solver.xb > solver.x_flat_right) & (solver.xb <= solver.x_right)
        t_right = (solver.xb[mask_right] - solver.x_flat_right) / (solver.x_right - solver.x_flat_right)
        
        solver.bathb[mask_left] = M * (np.exp(2*t_left) - 1) / (np.exp(2) - 1)
        solver.bathb[mask_flat] = M
        solver.bathb[mask_right] = M * (np.exp(3*(1-t_right)) - 1) / (np.exp(3) - 1)
        
    elif bath_type == 'rectangle':
        solver.x_flat_left = center
        solver.x_flat_right = center + M
        height = M
             
        mask_flat = (solver.xb >= solver.x_flat_left) & (solver.xb <= solver.x_flat_right)
        solver.bathb[mask_flat] = height

        solver.x_left, solver.x_right = None, None     
        
    elif bath_type == 'squared_trapezoid':
        solver.x_left = -M*1.5
        solver.x_flat_left = -M / 2
        solver.x_flat_right = M / 2
        solver.x_right = M*1.5
        height = M
        
        mask_left = (solver.xb >= solver.x_left) & (solver.xb < solver.x_flat_left)
        solver.bathb[mask_left] = height * (solver.xb[mask_left] - solver.x_left) / (solver.x_flat_left - solver.x_left)
        
        mask_flat = (solver.xb >= solver.x_flat_left) & (solver.xb <= solver.x_flat_right)
        solver.bathb[mask_flat] = height
        
        mask_right = (solver.xb > solver.x_flat_right) & (solver.xb <= solver.x_right)
        solver.bathb[mask_right] = height * (solver.x_right - solver.xb[mask_right]) / (solver.x_right - solver.x_flat_right)
        
    elif bath_type == 'semi_circular':
        radius = M 
        for i in range(solver.Nx):
            if abs(solver.xb[i] - center) <= radius:
                solver.bathb[i] = np.sqrt(radius**2 - (solver.xb[i] - center)**2)
            else:
                solver.bathb[i] = 0.0
        solver.x_left, solver.x_right, solver.x_flat_left, solver.x_flat_right = None, None, None, None
        
    elif bath_type == 'bump':
        height = M 
        width = M
        solver.bathb = height * np.exp(-((solver.xb - center)**(2*a)) / (2 * (width / 3)**(2*a)))
        solver.x_left, solver.x_right, solver.x_flat_left, solver.x_flat_right = None, None, None, None
        
    elif bath_type == 'ramp':
        height = M 
        center = 0.0
        width = M
        for i in range(solver.Nx):
            if abs(solver.xb[i] - center) <= width:
                solver.bathb[i] = (height / width) * (width - abs(solver.xb[i] - center))
            else:
                solver.bathb[i] = 0.0
        solver.x_left, solver.x_right, solver.x_flat_left, solver.x_flat_right = None, None, None, None
        
    elif bath_type == 'flat':
        solver.bathb = np.zeros(solver.Nx)
        solver.x_left, solver.x_right, solver.x_flat_left, solver.x_flat_right = None, None, None, None
        
    elif bath_type == 'sinusoidal':
        solver.x_flat_left = -M*7
        solver.x_flat_right = M*7
        amplitude = M / 8
        wavelength = M / 0.6
        
        # Only apply sinusoidal profile in the middle region
        mask_middle = (solver.xb >= solver.x_flat_left) & (solver.xb <= solver.x_flat_right)
        solver.bathb[mask_middle] = amplitude * np.cos(2 * np.pi * solver.xb[mask_middle] / wavelength)
        
        solver.x_left, solver.x_right = None, None

max_iter = 80001
# max_iter = 100

# Store all results in a dictionary
all_results = {}
ctr = 0
solver = SWEHBSolver(params)
bathymetry_profile("dead_zones", M=params['M'], center=-M/2, a=1)
probe_locs = None

# Run the simplified solver
# Note: removed plot_interval argument since the lite solver might still take it but won't plot
all_results[f'{filename}'] = solver.run(
                                bathymetry=solver.bathb, 
                                max_iter=max_iter, 
                                plot_interval=500, # Only for progress printing
                                probes=probe_locs, 
                                roll=False,
                                dam_break=False, 
                                reservoir=False, 
                                flux='Godunov', 
                                x_flat_left=solver.x_flat_left, 
                                x_flat_right=solver.x_flat_right,
                                x_left=solver.x_left, 
                                x_right=solver.x_right, 
                                center=ctr,
                                norm=M,
                                grad=True
                                )

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
