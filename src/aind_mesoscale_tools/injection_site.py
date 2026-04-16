"""
Localization and quantification of injection sites in mesoscale whole-brain data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize
from dask import array as da
# from .base_wrapper import wholebrain_data
from .utils import gaussian_3d_rotated

class InjectionSite:
    initialization_method: str = None
    centering_method: str = None
    fill_method: str = None

    # Initiator
    def __init__(self, brain, ch : str, reagent : str = None):
        self.data = brain
        self.channel = ch
        self.reagent = reagent

    def get_injection_center(self, level = 3, orientation = "coronal", seed = False, center_volume = False, center_span = 60, center_quantiles = [.95, .995]):
        """ Function to manually specify or estimate center of injection site """
        
        # Set zarr level and orientation
        self.center_level = level
        self.center_orientation = orientation

        # Get appropriate reference volume
        self.data.set_zarr_level(self.center_level, verbose = False)
        ch_vol = self.data.orient_zarr_volume(self.channel, plane = self.center_orientation)
        
        if seed:
            # Manual seed provided, convert to appropriate zarr level. Expects seed at full resolution.
            indx_max = tuple(self.data._convert_zarr_index(seed, output_level = self.center_level)) # Convert seed to index
            self.initialization_method = "manual"
            self.initialization_coordinate = indx_max

        else:
            # If no seed is provided, find brightest pixel in entire volume, then convert to index
            pos_max = np.argmax(ch_vol).compute() # Find brightest pixel in entire volume, then convert to index
            indx_max = np.unravel_index(pos_max, ch_vol.shape)
            self.initialization_method = "brightest"
            self.initialization_coordinate = indx_max
         
        if center_volume:
            # If centering is requested, compute center of mass within volume governed by span
            x_slice, y_slice, z_slice = [slice(max(0, indx - center_span), min(ch_vol.shape[i], indx + center_span)) for i, indx in enumerate(indx_max)]
            center_vol = ch_vol[x_slice,y_slice,z_slice]

            # Clip volume to signal for CoM calculation
            clip_vals = np.quantile(center_vol,center_quantiles)
            center_vol = center_vol - clip_vals[0] # Set everything below first quantile (95%) to 0
            center_vol = center_vol.clip(0,clip_vals[1] - clip_vals[0]) # Clip to second quantile (99.5%)

            # Compute center of mass, convert back to global indices
            com = np.round(ndimage.center_of_mass(np.array(center_vol)))
            coord = [x_slice.start, y_slice.start, z_slice.start] + com
            self.center_coordinate = tuple([int(c) for c in coord]) # Ensure coordinates are integers
            self.centering_method = "center_of_mass"
            self.centering_slices = {"x": x_slice, "y": y_slice, "z": z_slice} # save slice dict for later indexing
            self.centering_quantiles = center_quantiles
        else:
            self.center_coordinate = indx_max
            self.centering_method = "none"
            self.centering_slices = {}
            self.centering_quantiles = {}

        return self.center_coordinate
    
    def plot_injection_center(self, vmax = 600, verbose = True):
        # Function to plot the injection center.

        if not hasattr(self, 'center_coordinate'):
            print(f"No injection site found for channel {self.channel}. Running get_injection_center with default parameters first.")
            self.get_injection_center()

        # Load parameters from injection site estimation
        inj_coordinate = self.data._convert_zarr_index(self.center_coordinate, output_level = 0, input_level = self.center_level) # Convert coordinates to index

        # Do plotting
        plt.figure(figsize = (12,4.8))
        indx_calls = [[2, 1], [2, 0], [0, 1]]
        for i, plane in enumerate(["coronal","horizontal","sagittal"]):
            ax = plt.subplot(1,3,i + 1)
            ax.set_box_aspect(1)
            ax.set_facecolor('black')
            self.data.plot_slice(self.channel, plane = plane, level = self.center_level, section = inj_coordinate[i], vmax = vmax, verbose = verbose)
            plt.plot(inj_coordinate[indx_calls[i][0]], inj_coordinate[indx_calls[i][1]], '+', markersize = 5, markeredgewidth = 2, 
                     color = 'white', markerfacecolor = 'black') # Plot injection site as white cross
        plt.tight_layout()
    
    def get_injection_volume(self, method='manual', local_span=60, level=None, radius=60, percentile=90, gaussian_init=None, gaussian_threshold_sigma=1.0):
        """Estimate injection site volume around the center coordinate.
        
        Creates a binary mask using one of three approaches: manual spherical mask,
        3D Gaussian fitting, or percentile thresholding. The method stores results 
        as instance attributes and returns the binary mask.
        
        Parameters
        ----------
        method : str, default 'manual'
            Volume estimation method. Options:
            - 'manual': Create spherical mask with specified radius
            - 'gaussian': Fit 3D Gaussian and create mask from parameters
            - 'percentile': Use percentile thresholding with post-processing
        local_span : int, default 60
            Only used for 'gaussian' and 'percentile' methods. Defines the region
            around center_coordinate used for computation (cubic region of size 
            2*local_span+1 in each direction).
        radius : int, default 60
            Only used for 'manual' method. Radius in voxels (at requested zarr level)
            for spherical mask.
        level : int, optional
            Zarr level for volume computation. If None, uses the same level as
            center_coordinate (from get_injection_center).
        percentile : float, default 90
            Only used when method='percentile'. Percentile value (0-100) for
            thresholding. Voxels above this percentile will be included in the mask.
        gaussian_init : dict, optional
            Only used when method='gaussian'. Custom initialization parameters for
            Gaussian fitting. Dictionary with keys:
            - 'amplitude': Initial amplitude guess (float)
            - 'center': Initial center coordinates (tuple of 3 floats)
            - 'sigmas': Initial standard deviations (tuple of 3 floats) 
            - 'rotations': Initial rotation angles in radians (tuple of 3 floats)
            If None, automatic initialization is used.
        gaussian_threshold_sigma : float, optional
            Only used when method='gaussian'. Threshold for creating binary mask
            expressed in terms of standard deviations from peak. The threshold
            value is calculated as amplitude * exp(-sigma^2 / 2). If None,
            uses half-maximum threshold (default behavior).
            
        Returns
        -------
        np.ndarray
            Binary mask of the injection volume as a boolean numpy array.
            
        Raises
        ------
        ValueError
            If center_coordinate is not available or method is invalid.
        RuntimeError
            If Gaussian fitting fails to converge.
            
        Examples
        --------
        >>> # Get injection center first
        >>> injection.get_injection_center()
        >>> 
        >>> # Manual spherical mask (default)
        >>> mask = injection.get_injection_volume(method='manual', radius=50)
        >>> 
        >>> # 3D Gaussian fitting
        >>> mask = injection.get_injection_volume(method='gaussian', local_span=60)
        >>> 
        >>> # 3D Gaussian fitting with custom initialization
        >>> init = {'amplitude': 1000, 'center': (30, 30, 30), 'sigmas': (5, 5, 5), 'rotations': (0, 0, 0)}
        >>> mask = injection.get_injection_volume(method='gaussian', local_span=60, gaussian_init=init)
        >>> 
        >>> # 3D Gaussian fitting with sigma-based threshold (2 standard deviations)
        >>> mask = injection.get_injection_volume(method='gaussian', local_span=60, gaussian_threshold_sigma=2.0)
        >>> 
        >>> # Percentile thresholding with post-processing
        >>> mask = injection.get_injection_volume(method='percentile', local_span=80, percentile=95)
        """
        # Validate inputs
        if not hasattr(self, 'center_coordinate'):
            raise ValueError("No injection center found. Run get_injection_center() first.")
            
        valid_methods = ['manual', 'gaussian', 'percentile']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
            
        if local_span <= 0:
            raise ValueError("Local span must be positive")
            
        if radius <= 0:
            raise ValueError("Radius must be positive")
            
        if method == 'percentile' and not (0 <= percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100")
            
        # Set working level
        if level is None:
            level = self.center_level
        working_level = level
        
        # Store method parameters
        self.volume_method = method
        if method == 'manual':
            self.volume_radius = radius
        else:
            self.volume_local_span = local_span
        self.volume_level = working_level
        if method == 'percentile':
            self.volume_percentile = percentile
        if method == 'gaussian':
            self.volume_gaussian_init = gaussian_init
            self.volume_gaussian_threshold_sigma = gaussian_threshold_sigma
        
        # Get volume data at working level
        self.data.set_zarr_level(working_level, verbose=False)
        ch_vol = self.data.orient_zarr_volume(self.channel, plane=self.center_orientation)
        
        # Convert center coordinate to working level
        if working_level != self.center_level:
            center_coord = tuple(self.data._convert_zarr_index(
                self.center_coordinate, output_level=working_level, input_level=self.center_level
            ))
        else:
            center_coord = self.center_coordinate
            
        # Create slices around center coordinate using local_span
        x_slice, y_slice, z_slice = [
            slice(max(0, coord - local_span), min(ch_vol.shape[i], coord + local_span))
            for i, coord in enumerate(center_coord)
        ]
        
        # Extract local volume for processing
        local_vol = ch_vol[x_slice, y_slice, z_slice].compute()
        
        # Apply selected method
        if method == 'manual':
            mask = self._volume_manual_span(local_vol, radius)
            # For manual method, center remains the same
            updated_center_coord = center_coord
            
        elif method == 'gaussian':
            mask = self._volume_gaussian_fit(local_vol, gaussian_init, gaussian_threshold_sigma)
            # For Gaussian method, use fitted center coordinates
            if hasattr(self, 'volume_gaussian_params'):
                # Convert local coordinates back to full volume coordinates
                local_center = self.volume_gaussian_params['center']
                updated_center_coord = (
                    local_center[0] + x_slice.start,
                    local_center[1] + y_slice.start,
                    local_center[2] + z_slice.start
                )
            else:
                updated_center_coord = center_coord
            
        elif method == 'percentile':
            mask = self._volume_percentile_threshold(local_vol, percentile)
            # For percentile method, compute center of mass of the final mask
            local_com = ndimage.center_of_mass(mask.astype(float))
            updated_center_coord = (
                local_com[0] + x_slice.start,
                local_com[1] + y_slice.start,
                local_com[2] + z_slice.start
            )
            
        # Create full-size mask
        full_mask = np.zeros(ch_vol.shape, dtype=bool)
        full_mask[x_slice, y_slice, z_slice] = mask
        
        # Store results
        self.volume_mask = full_mask
        self.volume_local_mask = mask
        self.volume_slices = {'x': x_slice, 'y': y_slice, 'z': z_slice}
        self.volume_center_coord = updated_center_coord
        
        return full_mask
    
    def _volume_percentile_threshold(self, volume, percentile):
        """Create binary mask using percentile thresholding with post-processing.
        
        Uses percentile-based threshold and applies morphological post-processing
        (hole filling and largest connected component selection).
        
        Parameters
        ----------
        volume : np.ndarray
            3D volume array
        percentile : float
            Percentile value (0-100) for thresholding
            
        Returns
        -------
        np.ndarray
            Post-processed binary mask from percentile thresholding
        """
        # Calculate percentile threshold
        threshold = np.percentile(volume, percentile)
        
        # Create binary mask from threshold
        mask = volume > threshold
        
        # Apply morphological post-processing (obligate for percentile method)
        processed_mask = self._postprocess_mask(mask)
        
        return processed_mask
    
    def _volume_manual_span(self, volume, radius):
        """Create binary mask using manual spherical approach.
        
        Creates a spherical mask centered in the volume with specified radius.
        
        Parameters
        ----------
        volume : np.ndarray
            3D volume array
        radius : int
            Radius of the sphere in voxels
            
        Returns
        -------
        np.ndarray
            Binary spherical mask
        """
        # Get center coordinates of the volume
        center_x, center_y, center_z = np.array(volume.shape) / 2
        
        # Create coordinate grids
        x = np.arange(volume.shape[0])
        y = np.arange(volume.shape[1])
        z = np.arange(volume.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate distances from center
        distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2)
        
        # Create spherical mask
        mask = distances <= radius
        return mask
    
    def _volume_gaussian_fit(self, volume, gaussian_init=None, gaussian_threshold_sigma=1.0):
        """Create binary mask using 3D Gaussian fitting.
        
        Fits a 3D Gaussian function to the volume data and creates a mask
        based on the fitted parameters.
        
        Parameters
        ----------
        volume : np.ndarray
            3D volume array
        gaussian_init : dict, optional
            Custom initialization parameters. Dictionary with keys:
            - 'amplitude': Initial amplitude guess (float)
            - 'center': Initial center coordinates (tuple of 3 floats)
            - 'sigmas': Initial standard deviations (tuple of 3 floats) 
            - 'rotations': Initial rotation angles in radians (tuple of 3 floats)
            If None, automatic initialization is used.
        gaussian_threshold_sigma : float, optional
            Threshold for creating binary mask expressed in terms of standard
            deviations from peak. If None, uses half-maximum threshold.
            
        Returns
        -------
        np.ndarray
            Binary mask from Gaussian fitting
        """
        # Create coordinate grids
        x = np.arange(volume.shape[0])
        y = np.arange(volume.shape[1])
        z = np.arange(volume.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initial parameter guess
        if gaussian_init is not None:
            # Use custom initialization
            amplitude = gaussian_init.get('amplitude', np.max(volume))
            center_x, center_y, center_z = gaussian_init.get('center', np.array(volume.shape) / 2)
            sigma_x, sigma_y, sigma_z = gaussian_init.get('sigmas', (min(volume.shape) / 4,) * 3)
            alpha, beta, gamma = gaussian_init.get('rotations', (0, 0, 0))
            
            initial_params = [
                amplitude,           # A
                center_x, center_y, center_z,  # x0, y0, z0
                sigma_x, sigma_y, sigma_z,  # sigma_x, sigma_y, sigma_z
                alpha, beta, gamma             # alpha, beta, gamma (rotation angles)
            ]
        else:
            # Automatic initialization
            center_x, center_y, center_z = np.array(volume.shape) / 2
            amplitude = np.max(volume)
            sigma_init = min(volume.shape) / 4
            
            initial_params = [
                amplitude,           # A
                center_x, center_y, center_z,  # x0, y0, z0
                sigma_init, sigma_init, sigma_init,  # sigma_x, sigma_y, sigma_z
                0, 0, 0             # alpha, beta, gamma (rotation angles)
            ]
        
        try:
            # Flatten data for curve fitting
            coords = (X.ravel(), Y.ravel(), Z.ravel())
            volume_flat = volume.ravel()
            
            # Fit Gaussian
            popt, _ = optimize.curve_fit(
                gaussian_3d_rotated, coords, volume_flat, 
                p0=initial_params, maxfev=5000
            )
            
            # Generate mask from fitted Gaussian
            fitted_gaussian = gaussian_3d_rotated(
                (X, Y, Z), *popt
            ).reshape(volume.shape)
            
            # Create mask using specified threshold, based on sigma distance from peak
            threshold = popt[0] * np.exp(-gaussian_threshold_sigma**2 / 2)
                
            mask = fitted_gaussian > threshold
            
            # Store fitting parameters
            self.volume_gaussian_params = {
                'amplitude': popt[0],
                'center': (popt[1], popt[2], popt[3]),
                'sigmas': (popt[4], popt[5], popt[6]),
                'rotations': (popt[7], popt[8], popt[9])
            }
            
        except (RuntimeError, ValueError) as e:
            raise RuntimeError(f"Gaussian fitting failed: {e}")
            
        return mask
    
    def _postprocess_mask(self, mask):
        """Apply morphological post-processing to binary mask.
        
        Fills holes and selects the largest connected component to clean up
        the binary mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask to post-process
            
        Returns
        -------
        np.ndarray
            Post-processed binary mask
        """
        # Fill holes
        filled_mask = ndimage.binary_fill_holes(mask)
        
        # Label connected components
        labeled_mask, num_labels = ndimage.label(filled_mask)
        
        if num_labels == 0:
            return filled_mask
        
        # Find largest connected component
        component_sizes = ndimage.sum(filled_mask, labeled_mask, range(1, num_labels + 1))
        largest_component_label = np.argmax(component_sizes) + 1
        
        # Create final mask with only largest component
        final_mask = labeled_mask == largest_component_label
        
        return final_mask