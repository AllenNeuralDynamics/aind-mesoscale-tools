"""
Localization and quantification of injection sites in mesoscale whole-brain data.
"""

import numpy as np
from scipy import ndimage
from dask import array as da
from .base_wrapper import wholebrain_data

class InjectionSite:
    initialization_method: str = None
    centering_method: str = None
    fill_method: str = None

    # Initiator
    def __init__(self, brain : wholebrain_data, ch : str, reagent : str = None):
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

        else:
            # If no seed is provided, find brightest pixel in entire volume, then convert to index
            pos_max = np.argmax(ch_vol).compute() # Find brightest pixel in entire volume, then convert to index
            indx_max = np.unravel_index(pos_max, ch_vol.shape)
            self.initialization_method = "brightest"
         
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