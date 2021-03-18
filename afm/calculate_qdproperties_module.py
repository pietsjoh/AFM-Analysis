import numpy as np
from time import time
import scipy.stats as stats
import scipy.signal as sig
import warnings

import afm.postprocess_module as postprocess

class QDPropertiesCalculation:
    def __init__(self, z_data, localmax_ind):
        algoStart = time()
        assert isinstance(z_data, np.ndarray) and len(z_data.shape) == 2
        self.z_values = z_data
        self.Nx, self.Ny = self.z_values.shape
        assert isinstance(localmax_ind, tuple) and len(localmax_ind) == 2
        self.x_ind, self.y_ind = localmax_ind
        assert isinstance(self.x_ind, np.ndarray)
        assert isinstance(self.y_ind, np.ndarray)
        self.number_of_qds = self.x_ind.size
        self.abs_z_peak_list = self.z_values[self.x_ind, self.y_ind]
        self.ppc = postprocess.PostProcess2D(localmax_ind=(self.x_ind, self.y_ind))
        algoTime = round(time() - algoStart, 2)
        print(f"The absolute heights of the peaks have been calculated in {algoTime}s.")

#############################################################################################################################################################################

    def rel_height_scale(self, size, method="amean", methodScale=1):
        algoStart = time()
        if method == "amean":           ## arithmetic mean
            rel_scale = []
            for i in range(self.number_of_qds):
                X, Y = self.ppc.extract_qd_neighbourhood(size=size, index=i)
                Z = self.z_values[X.ravel(), Y.ravel()].reshape((2*size + 1, 2*size + 1))
                mean_z = np.mean(Z)
                rel_scale.append(mean_z)
        elif method == "median":        ## median
            rel_scale = []
            for i in range(self.number_of_qds):
                X, Y = self.ppc.extract_qd_neighbourhood(size=size, index=i)
                Z = self.z_values[X.ravel(), Y.ravel()].reshape((2*size + 1, 2*size + 1))
                median_z = np.median(Z)
                rel_scale.append(median_z)
        elif method == "gmean":         ## geometric mean
            rel_scale = []
            for i in range(self.number_of_qds):
                X, Y = self.ppc.extract_qd_neighbourhood(size=size, index=i)
                Z = self.z_values[X.ravel(), Y.ravel()].reshape((2*size + 1, 2*size + 1))
                gmean_z = stats.gmean(Z, axis=None)
                rel_scale.append(gmean_z)
        elif method == "hmean":         ## harmonic mean
            rel_scale = []
            for i in range(self.number_of_qds):
                X, Y = self.ppc.extract_qd_neighbourhood(size=size, index=i)
                Z = self.z_values[X.ravel(), Y.ravel()].reshape((2*size + 1, 2*size + 1))
                hmean_z = stats.hmean(Z, axis=None)
                rel_scale.append(hmean_z)
        elif method == "rms":           ## root-mean-square
            rel_scale = []
            for i in range(self.number_of_qds):
                X, Y = self.ppc.extract_qd_neighbourhood(size=size, index=i)
                Z = self.z_values[X.ravel(), Y.ravel()].reshape((2*size + 1, 2*size + 1))
                rms_z = np.sqrt(np.sum(Z*Z))/(2*size + 1)
                rel_scale.append(rms_z)
        else:
            raise NotImplementedError("use implemented methods: amean, gmean, hmean, rms or median")

        rel_scale = np.array(rel_scale)*methodScale
        self.rel_z_peak_list = self.abs_z_peak_list - rel_scale
        self.meanList = np.array(rel_scale)
        algoTime = round(time() - algoStart, 2)
        print(f"The relative heights of the peaks have been calculated in {algoTime}s (method = {method}, scale = {methodScale}).")

#############################################################################################################################################################################

    def fwhm(self, size, index):
        assert hasattr(self, "rel_z_peak_list")
        warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
        warnings.filterwarnings("ignore", message="some peaks have a width of 0")
        N = int(round(size/20))
        NRange = np.arange(size - N, size + N + 1, 1)
        Xorig, Yorig = self.ppc.extract_qd_neighbourhood(size, index)
        Zorig = self.z_values[Xorig.ravel(), Yorig.ravel()].reshape((2*size + 1, 2*size + 1))
        ZThresh = np.where(Zorig < self.meanList[index], self.meanList[index], Zorig)
        FWHMList = []
        for i in NRange:
            Z = ZThresh[:, i]
            FWHMList.append(sig.peak_widths(Z, np.array([size]), rel_height=0.5)[0])
        FWHMList = np.array(FWHMList)
        if np.isclose(np.mean(FWHMList), 0):
            FWHM = 0
        else:
            FWHM = np.nanmean(np.where(FWHMList != 0, FWHMList, np.nan)) 
        return FWHM

#############################################################################################################################################################################

    def generate_fwhm_list(self, size, lengthScale):
        algoStart = time()
        FWHMList = []
        for i in range(self.x_ind.size):
            FWHMList.append(self.fwhm(size, i))

        self.FWHMList = lengthScale*np.array(FWHMList)
        self.aspectRatioList = self.FWHMList/self.rel_z_peak_list
        algoTime = round(time() - algoStart, 2)
        print(f"Diameters (FWHM) and aspect ratios (diameter (FWHM) / relative height) have been calculated in {algoTime}s.")
        print()