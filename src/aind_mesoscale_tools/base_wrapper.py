"""
Module for handling whole brain mesoscale imaging data.
"""
import os

class brain:
    # Attributes
    baseResolution = [1.8, 1.8, 2] # microns
    zarrMultiple = {j:2**j for j in range(5)} # compression at each zarr level
    injectionSites = {} # injection site information, populated by getInjectionSites call
    
    # Initiator
    def __init__(self,sample, level = 3, verbose = True):
        self.sample = str(sample)
        self.getPath(verbose)
        self.setLevel(level, verbose)
        self.setColorMaps()
        self.injectionSites = {}
        
    # Methods
    def getPath(self, verbose):
        # Method to get path to whole brain volume data
        rootDir = Path('../data')
        rootDir = [file for file in rootDir.iterdir() if self.sample in str(file)]
        
        # Check that the appropriate number of folders were found.
        if len(rootDir) > 1:
            raise ValueError("Found multiple directories matching requested sample ID.")
        elif len(rootDir) == 0:
            raise ValueError("Could not find a data directory matching input sample ID.")
        self.rootDir = rootDir[0]
            
        # Handle iteration of several formatting conventions
        sampleDir = self.rootDir.joinpath('processed', 'stitching', 'OMEZarr')
        if not sampleDir.exists():
            sampleDir = self.rootDir.joinpath('processed', 'OMEZarr')
            if not sampleDir.exists():
                sampleDir = self.rootDir.joinpath("image_tile_fusing","OMEZarr")
        if verbose:
            print(f"Loading data from {sampleDir}")
        
        # Grab channel, named by excitation
        chPaths = {exCh.name.split('_')[1]: exCh for exCh in sampleDir.glob('Ex*.zarr')}
        self.channels = list(chPaths.keys())
        self.chPaths = chPaths
        if verbose:
            print(f"Found the following channels: {self.channels}")
        
        # Grab template based transformations
        transformDir = self.rootDir.joinpath("image_atlas_alignment")
        self.atlasChannels = [exCh.name.split('_')[1] for exCh in transformDir.glob('Ex*') if exCh.joinpath('ls_to_template_SyN_1Warp.nii.gz').exists()]
        self.atlasUseChannel = self.atlasChannels[-1]
        self.transformPaths = {} # Not yet implemented
        # transformPaths = {exCh.name.split('_')[1]: exCh.joinpath("detected_cells.xml") for exCh in transformDir.glob('Ex*')}
        if verbose:
            print(f"Found atlas alignment in the following channels: {self.atlasChannels}. Grabbing transforms from: {self.atlasUseChannel} (Not yet implemented)")
        
        # Grab cell proposals
        segDir = self.rootDir.joinpath("image_cell_segmentation")
        self.propPaths = {exCh.name.split('_')[1]: exCh.joinpath("detected_cells.xml") for exCh in segDir.glob('Ex*')}
        if verbose:
            print(f"Found cell proposals in the following channels: {list(self.propPaths.keys())}")
            
        # Grab cell classifications
        self.classPaths = {exCh.name.split('_')[1]: exCh.joinpath("classified_cells.xml") for exCh in segDir.glob('Ex*')}
        if verbose:
            print(f"Found cell classifications in the following channels: {list(self.classPaths.keys())}")
        
        # Grab CCF quantifications
        quantDir = self.rootDir.joinpath("image_cell_quantification")
        self.quantPaths = {exCh.name.split('_')[1]: exCh.joinpath("cell_count_by_region.csv") for exCh in quantDir.glob('Ex*')}
        self.ccfCellsPaths = {exCh.name.split('_')[1]: exCh.joinpath("transformed_cells.xml") for exCh in quantDir.glob('Ex*')}
        if verbose:
            print(f"Found atlas aligned quantifications in the following channels: {list(self.quantPaths.keys())}")
        
    def setLevel(self,level,verbose = True):
        # Method to update level and grab hierarchical volume for corresponding resolution level
        self.level = level
        if verbose:
            print(f"Grabbing volumes for level: {level}")
        self.getVol()
    
    def getVol(self):
        # Method to mount volumetric imaging data
        self.vols = {channel: da.from_zarr(str(chPath), self.level).squeeze() for channel,chPath in self.chPaths.items()}
    
    def orientVol(self, ch, plane = "coronal", returnLabels = False):
        # Method to orient requested channel volume to a particular plane. Return labels for internal methods, e.g. plotSlice
        if (plane.lower() == "horizontal") | (plane.lower() == "transverse"):
            printTxt = "Plotting horizontal axis, "
            axis = 0
            xLabel = "M/L"
            yLabel = "A/P"
        elif plane.lower() == "sagittal":
            printTxt = "Plotting sagittal axis, "
            axis = 2
            xLabel = "A/P"
            yLabel = "D/V"
        else:
            plane = "coronal"
            printTxt = "Plotting coronal axis, "
            axis = 1
            xLabel = "M/L"
            yLabel = "D/V"
        chVol = da.moveaxis(self.vols[ch],axis,0)
        
        if returnLabels:
            return chVol, xLabel, yLabel, printTxt
        else:
            return chVol
    
    def setColorMaps(self, base = "black",channelColors = {}):
        # Method to establish each channel's color map for future plotting. Modifies default colors via channelColors channel:color dictionary pairs
        colorSets = {"445":"turquoise","488":"lightgreen","561":"tomato","639":"white"} # default colors
        colormaps = {}
        
        # Modify color sets if channel colors are provided
        for ch, color in channelColors.items():
            if ch not in self.channels:
                raise ValueError(f"Trying to set color for channel {ch}, but channel was not found in dataset.")
            else:
                colorSets[ch] = color
        
        # Generate color maps for channels present in data
        for ch in self.channels:
            if ch not in colorSets.keys():
                print(f"No default color exists for the {ch} channel, setting to white.")
                colormaps[ch] = sns.blend_palette([base,'white'], as_cmap = True)
            else:
                colormaps[ch] = sns.blend_palette([base,colorSets[ch]], as_cmap = True)
        self.colormaps = colormaps
        
    def getInjectionSite(self, ch, level = 3, plane = 'sagittal', span = 60, showPlot = True):
        # Method to localize viral injection sites. 
        self.setLevel(level, showPlot)
        # For a given channel, find the center of mass in a span around the brightest point in the volume.
        chVol = self.orientVol(ch, plane = plane)  # Think about best orientation to save coordinates in
        posMax = np.argmax(chVol).compute() # Find brightest pixel in entire volume, then convert to index
        indxMax = np.unravel_index(posMax, chVol.shape)
        # Further process on volume centered at brightest point, size governed by span
        xSlice, ySlice, zSlice = slice(indxMax[0] - span,indxMax[0] + span), slice(indxMax[1] - span,indxMax[1] + span), slice(indxMax[2] - span,indxMax[2] + span)
        centerVol = chVol[xSlice,ySlice,zSlice]
        # Clip volume to signal for CoM calculation
        clipVals = np.quantile(centerVol,[.95,.995])
        centerVol = centerVol - clipVals[0] # Set everything below 95% to 0, clip to 95th percentile
        centerVol = centerVol.clip(0,clipVals[1] - clipVals[0])
        com = np.round(ndimage.center_of_mass(np.array(centerVol)))
        # Plot if requested
        if showPlot:
            plt.imshow(centerVol[com[0],:,:],cmap = self.colormaps[ch],vmax=1200)
            plt.plot(com[2],com[1],'or')
            
        coord = com - span + indxMax
        self.injectionSites[ch] = {"plane":plane, "level":level, "span":span, "coordinates":coord}
        # Edit later to include fitting function
        return centerVol
        
        
    def plotSlice(self,ch = [],plane = "coronal",section = [], extent = [], level = 3, vmin = 0, vmax = 600, alpha = 1, ticks = True, verbose = True):
        # Method to plot a particular slice
        
        # If no channel is provided, plot shortest wavelength
        if not ch:
            ch = min(self.channels)
        
        # Specify resolution level, and then retrieve properly oriented volume
        self.setLevel(level, verbose)
        [chVol, xLabel, yLabel, printTxt] = self.orientVol(ch, plane = plane, returnLabels = True)
        
        # Get data indices to be plotted
        if not section:
            sectionIndex = int(chVol.shape[0] / 2)
            section = sectionIndex * self.zarrMultiple[level]
        else: # otherwise convert microns to indices
            sectionIndex = int(section / self.zarrMultiple[level])
        if (not extent) | len(extent) != 4:
            extentIndices = np.array([0, chVol.shape[2], chVol.shape[1], 0])
            extent = extentIndices*self.zarrMultiple[level]
        else: #interpret extent requests as microns, convert to indices
            extentIndices = np.round(np.array(extent) / self.zarrMultiple[level])
        if verbose:
            print(printTxt + 'secion: ' + str(section) + ' (level ' + str(level) + ' index: ' + str(sectionIndex) + ')')
            
        # Plot data
        plt.imshow(chVol[sectionIndex,extentIndices[3]:extentIndices[2],extentIndices[0]:extentIndices[1]], cmap = self.colormaps[ch], 
                   vmin = vmin, vmax = vmax, extent = extent, alpha = alpha, interpolation='none')
        if ticks:
            plt.title(ch)
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
        else:
            plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)

    def plotPoint(self, cst, ch: list = [], span = 20, vmin = 0, vmax = 600):
        # Method to plot a given point in 3 planes, specified by variable cst (coronal, sagittal, transverse).
        
        # If no channel is provided, plot shortest wavelength
        if not ch:
            ch = min(self.channels)
        
        if span > 300:
            level = 1
        else:
            level = 0
        
        # Set up subplots
        nChannels = len(ch)
        planeDict = {0:"Coronal",1:"Sagittal",2:"Transverse"}
        extentDict = {0: [cst[1]-span, cst[1]+span, cst[2]+span,cst[2]-span], # M/L, D/V
                      1: [cst[0]-span, cst[0]+span, cst[2]+span,cst[2]-span], # A/P, D/V
                      2: [cst[1]-span, cst[1]+span, cst[0]+span,cst[0]-span], # M/L, A/P
                     }
        for chIndx, channel in enumerate(ch):
            for planeIndx in range(3):
                plt.subplot(nChannels,3,1+planeIndx+chIndx*3)
                self.plotSlice(ch=channel,plane = planeDict[planeIndx], section = cst[planeIndx], extent = extentDict[planeIndx],
                           level=0, vmin = vmin, vmax = vmax, printOutput = False, ticks = False)
                if chIndx == 0:
                    plt.title(planeDict[planeIndx])
        plt.tight_layout()
        plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
            

    def plotBlend(self,ch: list = [],plane = "coronal",section = [], extent = [], level = 3, alphaDict = [], vDict = [], ticks = True):
        # Method to plot blended channels
        
        # If no channels are provided, plot all. Default to longest wavelength first
        if not ch:
            ch = self.channels
            if ch[0] < ch[-1]:
                ch = ch[::-1]
            
        # If no alpha dict values are provided, use default
        if not alphaDict:
            defaultAlpha = 1/len(ch)
            alphaDict = {channel:defaultAlpha for channel in ch}
        
        # If no vmin / vmax dict values are provided, use default
        if not vDict:
            vDict = {channel:[0,600] for channel in ch}
        
        # Plot blended channels
        for channel in ch:
            self.plotSlice(ch=channel,plane = plane, section = section, extent =extent, level=level, alpha = alphaDict[channel],
                           vmin = vDict[channel][0], vmax = vDict[channel][1], printOutput = False, ticks = ticks)
        plt.title('')
        
    def getNGLink(self):
        # Method to print neuroglancer link of associated imaging data
        linkPath = self.rootDir.joinpath("neuroglancer_config.json")
        # linkPath =self.rootDir.joinpath("image_cell_segmentation/Ex_561_Em_593/visualization/neuroglancer_config.json")
        ngJSON = pd.read_json(linkPath, orient = 'index')
        print(ngJSON[0]["ng_link"])
        
    def getCellsCCF(self, ch: list):
        # Method to retrieve and format CCF transformed coordinates of segemented cells
        ccfDim = [528, 320, 456]
        locationDict = {}
        for channel in ch:
            locCellsDF = pd.read_xml(self.ccfCellsPaths[channel],xpath = "//CellCounter_Marker_File//Marker_Data//Marker_Type//Marker")
            locCells = locCellsDF.to_numpy() # Cells are output in [ML, DV, AP]
            # locCells[:,:] = locCells[:,[2,1,0]] # Rearrange indices to be AP, DV, ML
            for i, dim in enumerate(ccfDim): # Ensure cells fall within bounds of CCF annotation volume
                locCells[:,i] = locCells[:,i].clip(0,dim-1)
            locationDict[channel] = locCells
        return locationDict