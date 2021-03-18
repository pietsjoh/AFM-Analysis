import numpy as np
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.stats as stats
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage as ndi

class PostProcess2D:
    def __init__(self, data=None, localmax_ind=None):
        self.z_values = data
        if type(self.z_values) == np.ndarray:
            assert len(self.z_values.shape) == 2
            self.Nx, self.Ny = data.shape
            assert isinstance(self.Nx, (int, np.integer))
            assert isinstance(self.Ny, (int, np.integer))
        self.z_values_thresh = None
        self.z_values_cut = None
        if localmax_ind != None:
            assert isinstance(localmax_ind, tuple) and len(localmax_ind) == 2
            self.x_localmax_ind, self.y_localmax_ind = localmax_ind
            assert isinstance(self.x_localmax_ind, np.ndarray)
            assert isinstance(self.y_localmax_ind, np.ndarray)

############################################################################################################################################################################

    @staticmethod
    def plot_data(data, dimensions, style="contour_imshow"):
        assert isinstance(data, np.ndarray) and len(data.shape) == 2
        assert len(dimensions) == 2

        Nx, Ny = data.shape
        x_arr = np.linspace(0, Nx - 1, Nx)*(dimensions[0]/Nx)
        y_arr = np.linspace(0, Ny - 1, Ny)*(dimensions[1]/Ny)
        X, Y = np.meshgrid(x_arr, y_arr)

        if style == "contourf":
            plt.contourf(X, Y, data, cmap="RdGy")
            plt.colorbar()
            plt.show()
        elif style == "3d":
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X, Y, data, linewidth=0, antialiased=False, cmap="plasma")
            plt.show()
        elif style == "contour_imshow":
            plt.imshow(data, origin='lower', cmap='RdGy')
            plt.colorbar()
            plt.show()
        else:
            raise NotImplementedError("only 'contourf', '3d' and 'contour_imshow' styles are implemented")

############################################################################################################################################################################

    def threshold(self, fraction=0.5, level_thresh=True, use_cut=False):
        assert isinstance(self.z_values, np.ndarray) and len(self.z_values.shape) == 2
        assert np.amin(self.z_values) >= 0, "Use Gwyddion to 'shift minimum data value to zero', also use 'Level data by mean plane subtraction'"
        assert isinstance(fraction, (int, np.integer, float, np.floating))
        assert isinstance(level_thresh, bool)
        assert isinstance(use_cut, bool)

        start_threshold = time()

        if use_cut:
            assert isinstance(self.z_values_cut, np.ndarray), "Run cut_edges() first."
            z = self.z_values_cut
        else:
            z = self.z_values

        max_of_data = np.amax(z)
        threshold = fraction*max_of_data
        if level_thresh:
            self.z_values_thresh = np.where(z < threshold, threshold, z)
        else:
            self.z_values_thresh = np.where(z < threshold, 0, z)

        threshold_time = round(time() - start_threshold, 2)
        print("-"*100)
        print()
        print(f"Finished threshold() in {threshold_time}s")
        print()

############################################################################################################################################################################

    def cut_edges(self, size):
        assert isinstance(self.z_values, np.ndarray) and len(self.z_values.shape) == 2
        assert isinstance(size, (int, np.integer))

        start_cut_edges = time()

        self.z_values_cut = np.zeros((self.Nx, self.Ny))
        self.z_values_cut[size:-size, size:-size] = self.z_values[size:-size, size:-size]

        cut_edges_time = round(time() - start_cut_edges, 2)
        print("-"*100)
        print()
        print(f"Finished cut_edges() in {cut_edges_time}s")
        print()

############################################################################################################################################################################

    def extract_qd_neighbourhood(self, size, index):
        assert isinstance(size, (int, np.integer)) and size >= 0
        assert isinstance(index, (int, np.integer))

        y = self.y_localmax_ind[index]
        x = self.x_localmax_ind[index]
        x_val_near_peaks = np.arange(x - size, x + size + 1, 1)
        y_val_near_peaks = np.arange(y - size, y + size + 1, 1)
        X, Y = np.meshgrid(x_val_near_peaks, y_val_near_peaks)
        return X, Y

############################################################################################################################################################################

    def plot_single_qd(self, index, size, scales=(1, 1), height_scale=False):
        assert isinstance(self.z_values, np.ndarray) and len(self.z_values.shape) == 2
        assert isinstance(size, (int, np.integer))
        assert isinstance(index, (int, np.integer))

        X, Y = self.extract_qd_neighbourhood(size, index)
        Z = self.z_values[X.ravel(), Y.ravel()].reshape((2*size + 1, 2*size + 1))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X*scales[0], Y*scales[1], Z, linewidth=0, antialiased=False, cmap="plasma")
        if height_scale:
            Z_mean = np.mean(Z)*np.ones((2*size + 1, 2*size + 1))
            ax.plot_wireframe(X*scales[0], Y*scales[1], Z_mean, color="black")
        plt.show()

#############################################################################################################################################################################

    @staticmethod
    def plot_distribution(data):
        assert isinstance(data, np.ndarray) and len(data.shape) == 1

        maxData = np.amax(data)
        minData = np.amin(data)
        widthOfBins = 2*stats.iqr(data)/(data.size**(1/3))
        numberOfBins = int(np.ceil((maxData - minData)/widthOfBins))
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        plotScaleFactor = widthOfBins*(data.size)

        ## display text
        fig, ax = plt.subplots()
        textStr = "\n".join((
            r"$\mu=%.2f$" % (mean, ),
            r"$\sigma=%.2f$" % (std, )))
        props = dict(boxstyle="square", facecolor="white", alpha=0.5)
        fig.text(0.89, 0.69, textStr, horizontalalignment="center",
        verticalalignment="top", fontsize=11, bbox=props)
        ## plots histogram
        ax.hist(data, numberOfBins, color="blue", label="hist")

        ## plots kernel-density-approximation of the distribution using a gaussian kernel
        kernel = stats.gaussian_kde(data)
        xArray = np.linspace(minData, maxData, 1000)
        yArray = kernel(xArray)*plotScaleFactor
        ax.plot(xArray, yArray, color="black", label="gauss_kde")

        ## fits a gaussian to the data
        yArrayFit = plotScaleFactor*stats.norm.pdf(xArray, loc=mean, scale=std)
        ax.plot(xArray, yArrayFit, color="orange", label="gauss_fit")

        plt.subplots_adjust(right=0.78)
        plt.legend(loc="upper right", bbox_to_anchor=(1.33, 1))

    @staticmethod
    def interp_spline(X, Y, Z, upScale, downScale=-1, plot=False):
        x = X[0, :]
        y = Y[:, 0]
        spline = RectBivariateSpline(x, y, Z)
        x_min = np.amin(x)
        x_max = np.amax(x)
        y_min = np.amin(y)
        y_max = np.amax(y)

        if downScale > 0:
            x2 = np.linspace(x_min, x_max, downScale)
            y2 = np.linspace(y_min, y_max, downScale)
            X2, Y2 = np.meshgrid(x2, y2)
            Z2 = spline(x2, y2)

            spline2 = RectBivariateSpline(x2, y2, Z2)
            x3 = np.linspace(x_min, x_max, upScale)
            y3 = np.linspace(y_min, y_max, upScale)
            X3, Y3 = np.meshgrid(x3, y3)
            Z3 = spline2(x3, y3)

            if plot:
                fig, ax = plt.subplots(nrows=1, ncols=3, subplot_kw={"projection":"3d"})
                # ax[0].plot_surface(X, Y, Z, cmap=plt.cm.plasma)
                ax[0].plot_wireframe(X, Y, Z, color="k")
                ax[1].plot_wireframe(X2, Y2, Z2, color="k")
                ax[2].plot_wireframe(X3, Y3, Z3, color="k")
                plt.show()

            return X3, Y3, Z3
        else:
            x2 = np.linspace(x_min, x_max, upScale)
            x2 = np.linspace(x_min, x_max, upScale)
            y2 = np.linspace(y_min, y_max, upScale)
            X2, Y2 = np.meshgrid(x2, y2)
            Z2 = spline(x2, y2)

            if plot:
                fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection":"3d"})
                # ax[0].plot_surface(X, Y, Z, cmap=plt.cm.plasma)
                ax[0].plot_wireframe(X, Y, Z, color="k")
                ax[0].set_xlabel("x")
                ax[0].set_ylabel("y")
                ax[1].plot_wireframe(X2, Y2, Z2, color="k")
                ax[1].set_xlabel("x")
                ax[1].set_ylabel("y")
                plt.show()

            return X2, Y2, Z2

    @staticmethod
    def gaussian_filter(Z, sigma):
        assert isinstance(Z, np.ndarray) and len(Z.shape) == 2
        assert isinstance(sigma, (int, np.integer, float, np.floating)) and sigma > 0
        return ndi.gaussian_filter(Z, sigma)