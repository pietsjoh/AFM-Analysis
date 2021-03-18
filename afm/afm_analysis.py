import os
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
import matplotlib.pyplot as plt

import afm.postprocess_module as ppc
import afm.localmaxfinder_module as lmf
import afm.calculate_qdproperties_module as calc

class AFManalysis:
    def __init__(self):
        print("no path to the data was provided. Atleast the directory and filename should be provided when initializing.")
        print("Alternatively, manually set the filepath using set_filepath(), set_relative_data_directory() and set_absolute_data_directory().")

############################################################################################################################################################################

    def __init__(self, directory, fileName, pathToDirectory=None, useRel=True):
        if useRel:
            self.set_relative_data_directory(pathToDirectory)
        else:
            self.set_absolute_data_directory(pathToDirectory)
        self.set_filepath(directory, fileName, useRel)
        print()
        print(f"The path to the data file was extracted: {self.dataPath}")
        print("Use load_data() to get the data.")
        print()

############################################################################################################################################################################

    def set_relative_data_directory(self, relDataPath=None):
        if relDataPath is None:
            self.relDataPath = "../../dataAFM/"
        else:
            assert isinstance(relDataPath, str)
            self.relDataPath = relDataPath

############################################################################################################################################################################

    def set_absolute_data_directory(self, absDataPath):
        assert isinstance(absDataPath, str)
        assert os.path.isdir(absDataPath)
        self.absDataPath = absDataPath

############################################################################################################################################################################

    def set_filepath(self, directory, fileName, useRel=True):
        assert isinstance(directory, str) and isinstance(fileName, str)
        combineDirFile = f"{directory}/{fileName}"
        if useRel:
            assert hasattr(self, "relDataPath")
            parentPath = Path(__file__).parent
            relFilePath = self.relDataPath + combineDirFile
            self.dataPath = str((parentPath / relFilePath).resolve())
        else:
            self.dataPath = str((Path(self.absDataPath) / combineDirFile).resolve())
        assert os.path.exists(self.dataPath), "The file path does not exist."
        assert self.dataPath[-4:] == ".xyz", "Wrong file format. A .xyz file is expected."

############################################################################################################################################################################

    def load_data(self, imageDimensions, imageLines="default", zScale=10**9, filter=True, sigma=3):
        assert hasattr(self, "dataPath")
        assert len(imageDimensions) == 2 and isinstance(imageDimensions[0], (int, np.integer)) and isinstance(imageDimensions[1], (int, np.integer)) and imageDimensions[0] > 0 and imageDimensions[1] > 0, "Expected: physical image size in µm, Example: (1, 1) corresponds to a 1µm^2 quadratic image"

        startLoadData = time()
        print("-"*100)
        print()
        print(f"Loading input file {self.dataPath}")

        self.data = zScale*pd.read_csv(self.dataPath, delimiter="\t", header=None).to_numpy()
        z = self.data[:, 2]
        self.imageDimensions = imageDimensions

        if imageLines == "default":
            N = int(np.sqrt(z.size))
            assert np.isclose(N*N, z.size), "Image is not quadratic (different number of lines in x and y), use these numbers as input for imageLines instead."
            self.Nx = N
            self.Ny = N
        else:
            assert len(imageLines) == 2 and isinstance(imageLines[0], (int, np.integer)) and isinstance(imageLines[1], (int, np.integer)) and imageLines[0] > 0 and imageLines[1] > 0, "Expected: number of image lines, Example: (1024, 512) corresponds to 1024 lines in x and 512 lines in y"
            self.Nx, self.Ny = imageLines

        self.unprocessedZ= z.reshape((self.Nx, self.Ny))
        if filter:
            self.z = ppc.PostProcess2D.gaussian_filter(self.unprocessedZ, sigma)
            print()
            print(f"Applying Gaussian Filter (sigma={sigma}).")
        else:
            self.z = self.unprocessedZ
        self.meanZ = np.mean(self.z)

        timeLoadData = np.round(time() - startLoadData, 2)
        print()
        print(f"Finished loading data in {timeLoadData}s")
        print()

############################################################################################################################################################################

    def local_max(self, useCut=True, maxQDSize=100):
        assert hasattr(self, "z")
        assert hasattr(self, "imageDimensions")
        assert isinstance(maxQDSize, (int, np.integer, float, np.floating)) and maxQDSize > 0 and maxQDSize/1000 < self.imageDimensions[0] and maxQDSize/1000 < self.imageDimensions[1]

        if self.Nx == self.Ny and self.imageDimensions[0] == self.imageDimensions[1]:
            self.kernel = int(round(maxQDSize*self.Nx/(2*self.imageDimensions[0]*1000)))
        else:
            raise NotImplementedError("Currently only works for quadratic images")

        runLMF = lmf.LocalMaxFinder2D(self.z, useCut, self.kernel)
        runLMF.local_max_iterator()
        self.localMaxIndices = runLMF.localmax_ind
        self.numberOfQDs = runLMF.number_of_qds
        if useCut:
            self.newImageDimensions = (self.imageDimensions[0]*(self.Nx - 2*self.kernel)/self.Nx, self.imageDimensions[1]*(self.Ny - 2*self.kernel)/self.Ny)

############################################################################################################################################################################

    def density(self, densityScale=10**8):
        assert hasattr(self, "localMaxIndices")
        if hasattr(self, "newImageDimensions"):
            self.density = densityScale*self.localMaxIndices[0].size/(self.newImageDimensions[0]*self.newImageDimensions[1])
        else:
            assert hasattr(self, "imageDimensions")
            self.density = densityScale*self.localMaxIndices[0].size/(self.imageDimensions[0]*self.imageDimensions[1])

        densityScientificFormat = np.format_float_scientific(self.density, precision=2)
        print("-"*100)
        print()
        print(f"Quantum Dot density [cm^-2]: {densityScientificFormat}")
        print()

############################################################################################################################################################################

    def calculate_properties(self):
        assert hasattr(self, "kernel")
        print("-"*100)
        print()
        print("Calculating heights, diameters and aspect ratios for Quantum Dots.")
        print()
        startCalc = time()

        calcProp = calc.QDPropertiesCalculation(self.z, self.localMaxIndices)
        calcProp.rel_height_scale(self.kernel)
        lengthScale = 1000*self.imageDimensions[1]/self.Ny
        calcProp.generate_fwhm_list(size=self.kernel, lengthScale=lengthScale)
        self.heightList = calcProp.rel_z_peak_list
        self.diameterList = calcProp.FWHMList
        self.aspectRatioList = calcProp.aspectRatioList

        calcTime = np.round(time() - startCalc, 2)
        print()
        print(f"Finished calculating QD properties in {calcTime}s")
        print()

############################################################################################################################################################################

    def height_distribution(self):
        assert hasattr(self, "heightList")
        ppc.PostProcess2D.plot_distribution(self.heightList)
        plt.ylabel("Number of QDs")
        plt.xlabel("relative height [nm]")
        plt.show()

    def diameter_distribution(self):
        assert hasattr(self, "diameterList")
        ppc.PostProcess2D.plot_distribution(self.diameterList)
        plt.xlabel("Diameter at half maximum [nm]")
        plt.ylabel("Number of QDs")
        plt.show()

    def aspectratio_distribution(self):
        assert hasattr(self, "aspectRatioList")
        ppc.PostProcess2D.plot_distribution(self.aspectRatioList)
        plt.xlabel("Aspectratio (FWHM/height)")
        plt.ylabel("Number of QDs")
        plt.show()

############################################################################################################################################################################

    def plot_data(self, style="contour_imshow", useUnprocessed=False):
        if useUnprocessed:
            ppc.PostProcess2D.plot_data(data=self.unprocessedZ, dimensions=self.imageDimensions, style=style)
        else:
            ppc.PostProcess2D.plot_data(data=self.z, dimensions=self.imageDimensions, style=style)

    def plot_single_qd(self, index, heightScale=False):
        scaleX = 1000*self.imageDimensions[0]/self.Nx
        scaleY = 1000*self.imageDimensions[1]/self.Ny
        postproc = ppc.PostProcess2D(self.z, self.localMaxIndices)
        postproc.plot_single_qd(index, size=self.kernel, scales=(scaleX, scaleY), height_scale=heightScale)