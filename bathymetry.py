"""
Bathymetry profile definitions for the SWEHB solver.

Edit `custom_bathymetry()` below to design your own profile (bath_type='custom'),
or use bath_type='contour' with a two-column (x, z) CSV file to load a bathymetry
measured/exported elsewhere. Use SWEHBSolver.check_case() to visually confirm the
result before running a simulation.
"""
import numpy as np


def custom_bathymetry(xb, M, center=0, a=1):
    """
    Edit this function to build your own bathymetry profile.
    Must return an array with the same shape as xb.
    """
    return np.zeros_like(xb)


def build_bathymetry(xb, M, bath_type, center=0, a=1, contour_file=None):
    """
    Builds a bathymetry profile and the associated flat/slope region markers
    (x_left, x_right, x_flat_left, x_flat_right) used by the solver's dead-zone
    boundary handling.

    xb: cell-centered grid coordinates (solver.xb)
    M: characteristic obstacle scale (solver.M)
    bath_type: one of 'dead_zones', 'rectangle', 'squared_trapezoid',
        'semi_circular', 'bump', 'ramp', 'flat'/'none', 'sinusoidal',
        'contour' (loads bathymetry from contour_file, a two-column x,z CSV),
        'custom' (uses custom_bathymetry() defined in this file)
    center: horizontal offset of the obstacle (where applicable)
    a: shape exponent used by the 'bump' profile
    """
    Nx = xb.size
    bathb = np.zeros(Nx)
    x_left = x_right = x_flat_left = x_flat_right = None

    if bath_type == 'dead_zones':
        # Trapezoidal bathymetry parameters
        x_left = -1.5 * M
        x_flat_left = -M / 2
        x_flat_right = M / 2
        x_right = M

        mask_left = (xb >= x_left) & (xb < x_flat_left)
        t_left = (xb[mask_left] - x_left) / (x_flat_left - x_left)

        mask_flat = (xb >= x_flat_left) & (xb <= x_flat_right)

        mask_right = (xb > x_flat_right) & (xb <= x_right)
        t_right = (xb[mask_right] - x_flat_right) / (x_right - x_flat_right)

        bathb[mask_left] = M * (np.exp(2*t_left) - 1) / (np.exp(2) - 1)
        bathb[mask_flat] = M
        bathb[mask_right] = M * (np.exp(3*(1-t_right)) - 1) / (np.exp(3) - 1)

    elif bath_type == 'rectangle':
        center = -M/2
        x_flat_left = center
        x_flat_right = center + M
        height = M

        mask_flat = (xb >= x_flat_left) & (xb <= x_flat_right)
        bathb[mask_flat] = height

    elif bath_type == 'squared_trapezoid':
        x_left = -M*1.5
        x_flat_left = -M / 2
        x_flat_right = M / 2
        x_right = M*1.5
        height = M

        mask_left = (xb >= x_left) & (xb < x_flat_left)
        bathb[mask_left] = height * (xb[mask_left] - x_left) / (x_flat_left - x_left)

        mask_flat = (xb >= x_flat_left) & (xb <= x_flat_right)
        bathb[mask_flat] = height

        mask_right = (xb > x_flat_right) & (xb <= x_right)
        bathb[mask_right] = height * (x_right - xb[mask_right]) / (x_right - x_flat_right)

    elif bath_type == 'semi_circular':
        radius = M
        for i in range(Nx):
            if abs(xb[i] - center) <= radius:
                bathb[i] = np.sqrt(radius**2 - (xb[i] - center)**2)
            else:
                bathb[i] = 0.0

    elif bath_type == 'bump':
        height = M
        width = M
        bathb = height * np.exp(-((xb - center)**(2*a)) / (2 * (width / 3)**(2*a)))

    elif bath_type == 'ramp':
        height = M
        center = 0.0
        width = M
        for i in range(Nx):
            if abs(xb[i] - center) <= width:
                bathb[i] = (height / width) * (width - abs(xb[i] - center))
            else:
                bathb[i] = 0.0

    elif bath_type == 'flat' or bath_type == 'none':
        bathb = np.zeros(Nx)

    elif bath_type == 'sinusoidal':
        x_flat_left = -M*7
        x_flat_right = M*7
        amplitude = M / 8
        wavelength = M / 0.6

        # Only apply sinusoidal profile in the middle region
        mask_middle = (xb >= x_flat_left) & (xb <= x_flat_right)
        bathb[mask_middle] = amplitude * np.cos(2 * np.pi * xb[mask_middle] / wavelength)

    elif bath_type == 'contour':
        if contour_file is None:
            raise ValueError("contour_file must be provided when bath_type='contour'")
        data = np.loadtxt(contour_file, delimiter=',', skiprows=1)
        bathb = np.interp(xb, data[:, 0], data[:, 1])

    elif bath_type == 'custom':
        bathb = custom_bathymetry(xb, M, center=center, a=a)

    else:
        raise ValueError(f"Unknown bath_type '{bath_type}'")

    return bathb, x_left, x_right, x_flat_left, x_flat_right
