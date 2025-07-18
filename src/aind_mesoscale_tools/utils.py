"""
Utility functions for aind-mesoscale-tools.
"""

import numpy as np


def gaussian_3d_rotated(X, A, x0, y0, z0, sigma_x, sigma_y, sigma_z, alpha, beta, gamma):
    """
    3D Gaussian function with rotation for model fitting.
    
    Parameters
    ----------
    X : tuple of arrays
        Coordinate arrays (x, y, z)
    A : float
        Amplitude of the Gaussian
    x0, y0, z0 : float
        Center coordinates of the Gaussian
    sigma_x, sigma_y, sigma_z : float
        Standard deviations along each axis
    alpha, beta, gamma : float
        Rotation angles in radians
        
    Returns
    -------
    ndarray
        3D Gaussian values at the given coordinates
    """
    x, y, z = X
    # Center the coordinates
    xc = x - x0
    yc = y - y0
    zc = z - z0

    # Rotation matrix (radians)
    R = np.array([
        [np.cos(alpha) * np.cos(beta), 
         np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma),
         np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)],
        [np.sin(alpha) * np.cos(beta), 
         np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma),
         np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)],
        [-np.sin(beta), 
         np.cos(beta) * np.sin(gamma), 
         np.cos(beta) * np.cos(gamma)]
    ])

    # Rotate coordinates
    rotated_coords = R @ np.array([xc, yc, zc])
    xr, yr, zr = rotated_coords

    # Gaussian in rotated coordinates
    return A * np.exp(
        -((xr**2) / (2 * sigma_x**2) + (yr**2) / (2 * sigma_y**2) + (zr**2) / (2 * sigma_z**2))
    )
