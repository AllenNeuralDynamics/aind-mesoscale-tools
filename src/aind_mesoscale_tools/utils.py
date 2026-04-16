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
    x, y, z = (np.asarray(axis) for axis in X)
    input_shape = x.shape

    if y.shape != input_shape or z.shape != input_shape:
        raise ValueError("x, y, and z coordinate arrays must have the same shape")

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

    # Rotate coordinates in flattened form, then restore original shape.
    # This supports both raveled coordinates and meshgrid-style arrays.
    coords = np.vstack([xc.ravel(), yc.ravel(), zc.ravel()])
    rotated_coords = R @ coords
    xr = rotated_coords[0].reshape(input_shape)
    yr = rotated_coords[1].reshape(input_shape)
    zr = rotated_coords[2].reshape(input_shape)

    # Gaussian in rotated coordinates
    return A * np.exp(
        -((xr**2) / (2 * sigma_x**2) + (yr**2) / (2 * sigma_y**2) + (zr**2) / (2 * sigma_z**2))
    )
