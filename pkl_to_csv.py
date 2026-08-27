import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

M = 0.041
# Fr = 0.75 ---------------------------------
h0 = M/0.8 
theta = np.radians(3.54)
filename = 'boghiFr075Alpha08'
# # Fr = 1.25 ---------------------------------
h0 = M/0.4 
theta = np.radians(2.09)
filename = 'boghiFr125Alpha04'
# # # Fr = 1.75 ---------------------------------
h0 = M/0.6  # Fr = 1.75
theta = np.radians(5.36)
filename = 'boghiFr175Alpha06'

ty = 0
kn = 1
m = 1
rho = 1000
CFL = 0.5
xL = -M*40
xR = M*40
Nx = 2500#int((xR - xL)/dx)
tend = 1000

center = 0
norm = M
#filename = f'acquabona_deadzone_contour_{alpha}'

def pkl_to_csv(pkl_file):
    if not os.path.exists(pkl_file):
        print(f"Error: File {pkl_file} not found.")
        return

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"RESULTS/{pkl_file}_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    print(f"Loading {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        all_results = pickle.load(f)

    print(f"Found {len(all_results)} cases: {list(all_results.keys())}")

    for case_name, results in all_results.items():
        print(f"Processing case: {case_name}")
        
        # Get xb for index
        if 'xb' not in results:
            print(f"  Skipping {case_name}: 'xb' array not found.")
            continue
            
        xb = (results['xb'] - center)/norm
        #x = np.linspace(xL, xR, Nx + 1)
        length = len(xb)
        
        tb0 = results['tb0']
        u0 = results['u0']

        # List of expected array keys to process
        array_keys = ['zetab', 'bathb', 'hb', 'ub', 'Fr', 'Re', 'Bi', 'Pl', 'plugheight', 'tau']
        
        for key in array_keys:
            if key in results:
                if key == 'ub' or key == 'u':
                    val = results[key]/u0
                elif key == 'tau':
                    val = results[key]/tb0
                else:
                    val = results[key]/norm
                if isinstance(val, (np.ndarray, list)):
                    if len(val) == length:
                        # Create DataFrame for this variable with xb as index column
                        df = pd.DataFrame({'xb': xb, key: val})
                        output_filename = f"{case_name}_{key}.csv"
                        output_path = os.path.join(output_dir, output_filename)
                        df.to_csv(output_path, index=False)
                        print(f"    Saved {key} to {output_filename}")
                    # elif len(val) == length + 1 and key in ['u']:
                    #     # Special case for 'u' which has length Nx + 1
                    #     df = pd.DataFrame({'x': x, key: val})
                    #     output_filename = f"{case_name}_{key}.csv"
                    #     output_path = os.path.join(output_dir, output_filename)
                    #     df.to_csv(output_path, index=False)
                    #     print(f"    Saved {key} to {output_filename}")
                    else:
                        print(f"    Warning: Length mismatch for {key} in {case_name}. Expected {length}, got {len(val)}")


if __name__ == "__main__":
    # You can change the filename here or pass it as an argument
    pkl_filename = f'{filename}.pkl' 
    pkl_to_csv(pkl_filename)
