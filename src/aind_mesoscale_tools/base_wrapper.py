"""
Module for handling whole brain mesoscale imaging data.
"""
import os
from pathlib import Path
from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from dask import array as da

from .utils import gaussian_3d_rotated

class wholebrain_data:
    # Attributes
    base_resolution = [1.8, 1.8, 2] # microns
    zarr_multiple = {j:2**j for j in range(5)} # compression at each zarr level
    injection_sites = {} # injection site information, populated by get_injection_sites call
    
    # Initiator
    def __init__(self,sample, level = 3, verbose = True):
        self.sample = str(sample)
        self.get_data_paths(verbose)
        self.set_zarr_level(level, verbose)
        self.set_colormaps()
        self.injection_sites = {}
        
    # Methods
    def get_data_paths(self, verbose):
        # Method to get path to whole brain volume data
        sample_dir = self._find_sample_directory(verbose)
        self._get_image_volume_paths(sample_dir, verbose)
        self._get_atlas_transformation_paths(verbose)
        self._get_cell_segmentation_paths(verbose)
        self._get_atlas_quantification_paths(verbose)
    
    def _find_sample_directory(self, verbose):
        # Find the root directory for the sample and locate the sample directory
        root_dir = Path('../data')
        root_dir = [file for file in root_dir.iterdir() if self.sample in str(file)]
        
        # Check that the appropriate number of folders were found.
        if len(root_dir) > 1:
            raise ValueError("Found multiple directories matching requested sample ID.")
        elif len(root_dir) == 0:
            raise ValueError("Could not find a data directory matching input sample ID.")
        self.root_dir = root_dir[0]
        
        # Handle iteration of several formatting conventions ## MTS - Future versions might load metadata to direct loading
        sample_dir = self.root_dir.joinpath('processed', 'stitching', 'OMEZarr')
        if not sample_dir.exists():
            sample_dir = self.root_dir.joinpath('processed', 'OMEZarr')
            if not sample_dir.exists():
                sample_dir = self.root_dir.joinpath("image_tile_fusing","OMEZarr")
        if verbose:
            print(f"Loading data from {sample_dir}")
        return sample_dir
    
    def _get_image_volume_paths(self, sample_dir, verbose):
        # Grab channel, named by excitation
        ch_paths = {exCh.name.split('_')[1]: exCh for exCh in sample_dir.glob('Ex*.zarr')}
        self.channels = list(ch_paths.keys())
        self.ch_paths = ch_paths
        if verbose:
            print(f"Found image volumes in the following channels: {self.channels}")
    
    def _get_atlas_transformation_paths(self, verbose):
        # Grab template based transformations
        transform_dir = self.root_dir.joinpath("image_atlas_alignment")
        self.atlas_channels = [exCh.name.split('_')[1] for exCh in transform_dir.glob('Ex*') if exCh.joinpath('ls_to_template_SyN_1Warp.nii.gz').exists()]
        self.atlas_use_channel = self.atlas_channels[-1] if self.atlas_channels else None
        self.transform_paths = {} # Not yet implemented
        # transform_paths = {exCh.name.split('_')[1]: exCh.joinpath("detected_cells.xml") for exCh in transform_dir.glob('Ex*')}
        if verbose:
            print(f"Found atlas alignment in the following channels: {self.atlas_channels}. Grabbing transforms from: {self.atlas_use_channel} (Not yet implemented)")
    
    def _get_cell_segmentation_paths(self, verbose):
        # Grab cell proposals and classifications
        seg_dir = self.root_dir.joinpath("image_cell_segmentation")
        self.prop_paths = {exCh.name.split('_')[1]: exCh.joinpath("detected_cells.xml") for exCh in seg_dir.glob('Ex*')}
        if verbose:
            print(f"Found cell proposals in the following channels: {list(self.prop_paths.keys())}")
            
        # Grab cell classifications
        self.class_paths = {exCh.name.split('_')[1]: exCh.joinpath("classified_cells.xml") for exCh in seg_dir.glob('Ex*')}
        if verbose:
            print(f"Found cell classifications in the following channels: {list(self.class_paths.keys())}")
    
    def _get_atlas_quantification_paths(self, verbose):
        # Grab CCF quantifications
        quant_dir = self.root_dir.joinpath("image_cell_quantification")
        self.quant_paths = {exCh.name.split('_')[1]: exCh.joinpath("cell_count_by_region.csv") for exCh in quant_dir.glob('Ex*')}
        self.ccf_cells_paths = {exCh.name.split('_')[1]: exCh.joinpath("transformed_cells.xml") for exCh in quant_dir.glob('Ex*')}
        if verbose:
            print(f"Found atlas aligned quantifications in the following channels: {list(self.quant_paths.keys())}")
        
    def set_zarr_level(self,level,verbose = True):
        # Method to update level and grab hierarchical volume for corresponding resolution level
        self.level = level
        if verbose:
            print(f"Grabbing volumes for level: {level}")
        self.get_zarr_volume()

    def get_zarr_volume(self):
        # Method to mount volumetric imaging data
        self.vols = {channel: da.from_zarr(str(ch_path), self.level).squeeze() for channel,ch_path in self.ch_paths.items()}
    
    def orient_zarr_volume(self, ch, plane = "coronal", return_labels = False):
        # Method to orient requested channel volume to a particular plane. Return labels for internal methods, e.g. plot_slice

        # Check inputs
        ch = self._check_channel_provided(ch)[0]

        # Set axis and labels based on plane. Adjust later to read metadata.
        if (plane.lower() == "horizontal") | (plane.lower() == "transverse"):
            print_txt = "Plotting horizontal axis, "
            axis = 0
            x_label = "M/L"
            y_label = "A/P"
        elif plane.lower() == "sagittal":
            print_txt = "Plotting sagittal axis, "
            axis = 2
            x_label = "A/P"
            y_label = "D/V"
        else:
            plane = "coronal"
            print_txt = "Plotting coronal axis, "
            axis = 1
            x_label = "M/L"
            y_label = "D/V"
        ch_vol = da.moveaxis(self.vols[ch],axis,0)
        
        if return_labels:
            return ch_vol, x_label, y_label, print_txt
        else:
            return ch_vol
    
    def set_colormaps(self, base = "black",channel_colors = {}):
        # Method to establish each channel's color map for future plotting. Modifies default colors via channel_colors channel:color dictionary pairs
        color_sets = {"445":"turquoise","488":"lightgreen","561":"tomato","639":"white"} # default colors
        colormaps = {}
        
        # Modify color sets if channel colors are provided
        for ch, color in channel_colors.items():
            if ch not in self.channels:
                raise ValueError(f"Trying to set color for channel {ch}, but channel was not found in dataset.")
            else:
                color_sets[ch] = color
        
        # Generate color maps for channels present in data
        for ch in self.channels:
            if ch not in color_sets.keys():
                print(f"No default color exists for the {ch} channel, setting to white.")
                colormaps[ch] = sns.blend_palette([base,'white'], as_cmap = True)
            else:
                colormaps[ch] = sns.blend_palette([base,color_sets[ch]], as_cmap = True)
        self.colormaps = colormaps
        
    def get_injection_site(self, ch, level = 3, seed = False, center = False, plane = 'coronal', span = 60, verbose = True):
        # Method to localize viral injection sites. 

        # Check inputs
        ch = self._check_channel_provided(ch)[0]

        # Specify resolution level, and then retrieve properly oriented volume
        self.set_zarr_level(level, verbose)
        ch_vol = self.orient_zarr_volume(ch, plane = plane)

        if seed:
            # Manual seed provided, convert to appropriate zarr level
            indx_max = tuple(self._convert_zarr_index(seed, output_level = level)) # Convert seed to index
            
        else:
            # If no seed is provided, find brightest pixel in entire volume, then convert to index
            pos_max = np.argmax(ch_vol).compute() # Find brightest pixel in entire volume, then convert to index
            indx_max = np.unravel_index(pos_max, ch_vol.shape)
        
        if center:
            # Further process on volume centered at brightest point, size governed by span
            x_slice, y_slice, z_slice = [slice(max(0, indx - span), min(ch_vol.shape[i], indx + span)) for i, indx in enumerate(indx_max)]
            slice_dict = {"x": x_slice, "y": y_slice, "z": z_slice} # save slice dict for later indexing
            center_vol = ch_vol[x_slice,y_slice,z_slice]

            # Clip volume to signal for CoM calculation
            clip_vals = np.quantile(center_vol,[.95,.995])
            center_vol = center_vol - clip_vals[0] # Set everything below 95% to 0, clip to 95th percentile
            center_vol = center_vol.clip(0,clip_vals[1] - clip_vals[0])
            com = np.round(ndimage.center_of_mass(np.array(center_vol)))
            coord = [x_slice.start, y_slice.start, z_slice.start] + com
        else:
            coord = indx_max
            slice_dict = {}

        # Save injection site and estimation parameters to data object
        self.injection_sites[ch] = {"plane":plane, "level":level, "coordinates":coord, "seed":seed, "center":center, "slice_dict": slice_dict, "span":span}
        
        # Plot validation if requested
        if verbose:
            self.plot_injection_site(ch)
            pass
            
        # Add fitting function / volume estimation to another function
        return self.injection_sites[ch]
        
        
    def plot_slice(self,ch = [],plane = "coronal",section = [], extent = [], level = 3, vmin = 0, vmax = 600, alpha = 1, ticks = True, verbose = True):
        # Method to plot a particular slice
        
        # Check inputs
        ch = self._check_channel_provided(ch)[0]
        
        # Specify resolution level, and then retrieve properly oriented volume
        self.set_zarr_level(level, verbose)
        [ch_vol, x_label, y_label, print_txt] = self.orient_zarr_volume(ch, plane = plane, return_labels = True)
        
        # Get data indices to be plotted
        section = self._check_section_provided(section, ch_vol, level)
        section_index = self._convert_zarr_index(section, output_level = level)

        # Get extent indices for plotting
        extent = self._check_extent_provided(extent, ch_vol, level)
        extent_indices = self._convert_zarr_index(extent, output_level = level)

        if verbose:
            print(print_txt + 'secion: ' + str(section) + ' (level ' + str(level) + ' index: ' + str(section_index) + ')')
            
        # Plot data
        plt.imshow(ch_vol[section_index,extent_indices[3]:extent_indices[2],extent_indices[0]:extent_indices[1]], cmap = self.colormaps[ch], 
                   vmin = vmin, vmax = vmax, extent = extent, alpha = alpha, interpolation='none')
        if ticks:
            plt.title(ch)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        else:
            plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)

    def plot_injection_site(self, ch):
        # Function to plot the injection site for a given channel.

        # Check inputs
        ch = self._check_channel_provided(ch)[0]

        # Check if injection site exists
        if ch not in self.injection_sites.keys():
            print(f"No injection site found for channel {ch}. Running get_injection_site first.")
            self.get_injection_site(ch, verbose= False)

        # Load parameters from injection site estimation
        site_info = self.injection_sites[ch]
        level = site_info["level"]
        inj_coordinate = self._convert_zarr_index(site_info["coordinates"], output_level = 0, input_level = level) # Convert coordinates to index

        # Do plotting
        plt.figure(figsize = (12,4.8))
        indx_calls = [[2, 1], [2, 0], [0, 1]]
        for i, plane in enumerate(["coronal","horizontal","sagittal"]):
            ax = plt.subplot(1,3,i + 1)
            ax.set_box_aspect(1)
            ax.set_facecolor('black')
            self.plot_slice(ch = ch, plane = plane, level = level, section = inj_coordinate[i])
            plt.plot(inj_coordinate[indx_calls[i][0]], inj_coordinate[indx_calls[i][1]], '+', markersize = 5, markeredgewidth = 2, 
                     color = 'white', markerfacecolor = 'black') # Plot injection site as white cross
        plt.tight_layout()


    def plot_point(self, cst, ch: list = [], span = 20, vmin = 0, vmax = 600):
        # Method to plot a given point in 3 planes, specified by variable cst (coronal, sagittal, transverse).
        
        # Get default channel if none is provided
        ch = self._check_channel_provided(ch)[0]
        
        if span > 300:
            level = 1
        else:
            level = 0
        
        # Set up subplots
        n_channels = len(ch)
        plane_dict = {0:"Coronal",1:"Sagittal",2:"Transverse"}
        extent_dict = {0: [cst[1]-span, cst[1]+span, cst[2]+span,cst[2]-span], # M/L, D/V
                      1: [cst[0]-span, cst[0]+span, cst[2]+span,cst[2]-span], # A/P, D/V
                      2: [cst[1]-span, cst[1]+span, cst[0]+span,cst[0]-span], # M/L, A/P
                     }
        for ch_indx, channel in enumerate(ch):
            for plane_indx in range(3):
                plt.subplot(n_channels,3,1+plane_indx+ch_indx*3)
                self.plot_slice(ch=channel,plane = plane_dict[plane_indx], section = cst[plane_indx], extent = extent_dict[plane_indx],
                           level=0, vmin = vmin, vmax = vmax, verbose = False, ticks = False)
                if ch_indx == 0:
                    plt.title(plane_dict[plane_indx])
        plt.tight_layout()
        plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
            

    def plot_blend(self,ch: list = [],plane = "coronal",section = [], extent = [], level = 3, alpha_dict = [], v_dict = [], ticks = True):
        # Method to plot blended channels
        
        # If no channels are provided, plot all. Default to longest wavelength first
        if not ch:
            ch = self.channels
            if ch[0] < ch[-1]:
                ch = ch[::-1]
            
        # If no alpha dict values are provided, use default
        if not alpha_dict:
            default_alpha = 1/len(ch)
            alpha_dict = {channel:default_alpha for channel in ch}
        
        # If no vmin / vmax dict values are provided, use default
        if not v_dict:
            v_dict = {channel:[0,600] for channel in ch}
        
        # Plot blended channels
        for channel in ch:
            self.plot_slice(ch=channel,plane = plane, section = section, extent =extent, level=level, alpha = alpha_dict[channel],
                           vmin = v_dict[channel][0], vmax = v_dict[channel][1], verbose = False, ticks = ticks)
        plt.title('')
        
    def get_neuroglancer_link(self):
        # Method to print neuroglancer link of associated imaging data
        link_path = self.root_dir.joinpath("neuroglancer_config.json")
        # link_path =self.root_dir.joinpath("image_cell_segmentation/Ex_561_Em_593/visualization/neuroglancer_config.json")
        ng_json = pd.read_json(link_path, orient = 'index')
        print(ng_json[0]["ng_link"])
        
    def get_atlas_aligned_cells(self, ch: list):
        # Method to retrieve and format CCF transformed coordinates of segemented cells

        # Get default channel if none is provided
        ch = self._check_channel_provided(ch)

        ccf_dim = [528, 320, 456]
        location_dict = {}
        for channel in ch:
            loc_cells_df = pd.read_xml(self.ccf_cells_paths[channel],xpath = "//CellCounter_Marker_File//Marker_Data//Marker_Type//Marker")
            loc_cells = loc_cells_df.to_numpy() # Cells are output in [AP, DV, ML] ### NOTE, some older versions are oriented differently
            # Ensure cells fall within bounds of CCF annotation volume
            for i, dim in enumerate(ccf_dim):
                loc_cells[:,i] = loc_cells[:,i].clip(0,dim-1)
            location_dict[channel] = loc_cells
        return location_dict

    def get_max_intensity_projection(self,ch: list[str], section: int, plane: str = "coronal", span: int = 25, level: int = 3) -> dict:
        # Method to get maximum intensity projection of a channel in a given plane.

        # Check inputs
        ch = self._check_channel_provided(ch)

        # Use input slice conditions
        xSlice = slice(section - span, section + span)

        # Get volumes
        self.set_zarr_level(level = level)
        mip_dict = {}
        for channel in ch:
            ch_vol = self.orient_zarr_volume(channel,plane = plane)
            mip_dict[channel] = np.max(ch_vol[xSlice,:,:], axis = 0)

        return mip_dict
    
    def _check_section_provided(self, section: int, ch_vol: da.Array, level: int) -> int:
        """
        Helper method to check if a section index is provided.
        If no section is provided, uses the midpoint of the volume.
        """
        if not section:
            return int(self.zarr_multiple[level] * ch_vol.shape[0] / 2)
        return section
    
    def _convert_zarr_index(self, input: int, output_level: int, input_level: int = 0) -> int:
        """
        Helper method to convert between arbitrary zarr levels.
        """
        output = self.zarr_multiple[input_level] * np.asarray(input) / self.zarr_multiple[output_level]
        # Return scalar int if input was scalar, else ndarray of ints
        return int(output) if np.isscalar(input) else output.astype(int)

    def _check_extent_provided(self, extent: np.ndarray, ch_vol: da.Array, level: int) -> np.ndarray:
        """
        Helper method to check extent indices for plotting.
        """
        if (not extent) | (len(extent) != 4):
            if extent:
                print("Unexpected extent format, using full volume dimensions.")
            extent = np.array([0, ch_vol.shape[2], ch_vol.shape[1], 0]) * self.zarr_multiple[level]

        else: # check that extent is within bounds of volume
            extent_indices = self._convert_zarr_index(extent, level)
            if not (0 <= extent_indices[0] < extent_indices[1] <= ch_vol.shape[2] and \
                    0 <= extent_indices[3] < extent_indices[2] <= ch_vol.shape[1]):
                print("Extent indices out of bounds, using full volume dimensions.")
                extent = np.array([0, ch_vol.shape[2], ch_vol.shape[1], 0]) * self.zarr_multiple[level]
        return extent
    
    def _check_channel_provided(self, ch):
        """
        Helper method to clean channel inputs to a list of strings.
        If no channel is provided, returns the shortest wavelength channel.
        
        Args:
            ch: Channel input that can be:
                - int: single channel as integer
                - str: single channel as string  
                - list[int]: list of channels as integers
                - list[str]: list of channels as strings
                - empty list/None: will use default channel
                
        Returns:
            list[str]: List of channel names as strings
        """
        # Handle empty/None input - use default channel
        if not ch:
            print(f"No channel provided, using {min(self.channels)}")
            return [min(self.channels)]
        
        # Handle single values (int or str)
        if isinstance(ch, (int, str)):
            return [str(ch)]
        
        # Handle lists
        if isinstance(ch, list):
            # Convert all elements to strings
            return [str(channel) for channel in ch]
        
        # Fallback for unexpected types
        return [str(ch)]