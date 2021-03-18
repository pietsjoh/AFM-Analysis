import numpy as np
from time import time
from copy import deepcopy
import multiprocessing
import concurrent.futures as cf
from functools import partial

from afm.postprocess_module import PostProcess2D

class LocalMaxFinder2D:    
    def __init__(self, data, use_cut=False, kernel_size=50):
        assert isinstance(use_cut, bool)
        assert isinstance(data, np.ndarray) and len(data.shape) == 2
        assert isinstance(kernel_size, (int, np.integer)) and kernel_size > 0

        self.z_values = data
        self.Nx, self.Ny = data.shape
        self.max_kernel = kernel_size

        if use_cut:
            ppc = PostProcess2D(data=self.z_values)
            ppc.cut_edges(size=self.max_kernel)
            self.local_max_values = ppc.z_values_cut
        else:
            self.local_max_values = self.z_values

        self.localmax_ind = self.local_max_values.nonzero()
        self.number_of_qds = self.localmax_ind[0].size
        
        self.cpu_count = multiprocessing.cpu_count()
        self.multiple_counts = 0
        self.local_max_xindex_dump = None
        self.local_max_yindex_dump = None
        self.number_of_points = None
    
############################################################################################################################################################################

    def mp_prepare(self, part):
        assert isinstance(part, int) and 0 <= part <= (self.cpu_count - 1)

        x, y = self.localmax_ind
        self.number_of_points = int(np.floor(self.number_of_qds/self.cpu_count))
        if part == self.cpu_count - 1:
            x_part = x[part*self.number_of_points :]
            y_part = y[part*self.number_of_points :]
        else:
            x_part = x[part*self.number_of_points : part*self.number_of_points + self.number_of_points]
            y_part = y[part*self.number_of_points : part*self.number_of_points + self.number_of_points]

        return x_part, y_part

############################################################################################################################################################################
    
    def mp_lmc(self, part, kernel=1):
        assert isinstance(kernel, int) and kernel > 0
        
        peaks_x, peaks_y = self.mp_prepare(part)
        new_peaks = deepcopy(self.local_max_values)
        kill_count = 0
        self.multiple_counts = 0
        for x, y in zip(peaks_x, peaks_y):
            if 0 <= y - kernel and y + kernel <= self.Ny - 1 and 0 <= x - kernel and x + kernel <= self.Nx - 1:
                near_x = np.arange(x - kernel, x + kernel + 1, 1)
                near_y = np.arange(y - kernel, y + kernel + 1, 1)
            else:
                for i in np.arange(0, kernel + 1, 1):
                    if 0 <= y - kernel and y + kernel <= self.Ny - 1 and 0 <= x - kernel and x + i <= self.Nx - 1:
                        near_x = np.arange(x - kernel, x + i + 1, 1)
                        near_y = np.arange(y - kernel, y + kernel + 1, 1)
                    elif 0 <= y - kernel and y + kernel <= self.Ny - 1 and 0 <= x - i and x + kernel <= self.Nx - 1:
                        near_x = np.arange(x - i, x + kernel + 1, 1)
                        near_y = np.arange(y - kernel, y + kernel + 1, 1)
                    elif 0 <= y - kernel and y + i <= self.Ny - 1 and 0 <= x - kernel and x + kernel <= self.Nx - 1:
                        near_x = np.arange(x - kernel, x + kernel + 1, 1)
                        near_y = np.arange(y - kernel, y + i + 1, 1)
                    elif 0 <= y - i and y + kernel <= self.Ny - 1 and 0 <= x - kernel and x + kernel <= self.Nx - 1:
                        near_x = np.arange(x - kernel, x + kernel + 1, 1)
                        near_y = np.arange(y - i, y + kernel + 1, 1)
                    else:
                        for j in np.arange(0, kernel + 1, 1):
                            if y - i < 0 == y - i + 1 and x - j < 0 == x - j + 1:
                                near_x = np.arange(x - j + 1, x + kernel + 1, 1)
                                near_y = np.arange(y - i + 1, y + kernel + 1, 1)
                            elif y + i > self.Ny - 1 == y + i - 1 and x + j > self.Nx - 1 == x + j - 1:
                                near_x = np.arange(x - kernel, x + j, 1)
                                near_y = np.arange(y - kernel, y + i, 1)
                            elif y + i > self.Ny - 1 == y + i - 1 and x - j < 0 == x - j + 1:
                                near_x = np.arange(x - j + 1, x + kernel + 1, 1)
                                near_y = np.arange(y - kernel, y + i, 1)
                            elif y - i < 0 == y - i + 1 and x + j > self.Nx - 1 == x + j - 1:
                                near_x = np.arange(x - kernel, x + j, 1)
                                near_y = np.arange(y - i + 1, y + kernel + 1, 1)
            z_tmp = self.z_values[x, y]
            multiple_count_tmp = 0
            for nx in near_x:
                for ny in near_y:
                    if self.z_values[nx, ny] > z_tmp:
                        new_peaks[x, y] = 0
                        kill_count += 1
                        break
                    elif self.z_values[nx, ny] == z_tmp:
                        multiple_count_tmp += 1
                else:
                    continue
                break
            if multiple_count_tmp >= 2:
                self.multiple_counts +=1
        
        x, y = new_peaks.nonzero()
        if part == self.cpu_count - 1:
            xr = x[part*self.number_of_points : ]
            yr = y[part*self.number_of_points : ]
        else: 
            xr = x[part*self.number_of_points : part*self.number_of_points + self.number_of_points - kill_count]
            yr = y[part*self.number_of_points : part*self.number_of_points + self.number_of_points - kill_count]
        return xr, yr

############################################################################################################################################################################

    def mp_routine(self, kernel):
        start_time = time()

        self.local_max_xindex_dump = np.array([], dtype=np.int64)
        self.local_max_yindex_dump = np.array([], dtype=np.int64)
        with multiprocessing.Pool(processes=self.cpu_count) as pool:
            results = pool.map(partial(self.mp_lmc, kernel=kernel), range(self.cpu_count))
            pool.close()
            pool.join()
        for result in results:
            x, y = result
            self.local_max_xindex_dump = np.append(self.local_max_xindex_dump, x)
            self.local_max_yindex_dump = np.append(self.local_max_yindex_dump, y)
        
        self.localmax_ind = self.local_max_xindex_dump, self.local_max_yindex_dump
        self.number_of_qds = self.local_max_xindex_dump.size
        self.local_max_values = np.zeros((self.Nx, self.Ny))
        for i in range(self.number_of_qds):
            x_ind = self.local_max_xindex_dump[i]
            y_ind = self.local_max_yindex_dump[i]
            z = self.z_values[x_ind, y_ind]
            self.local_max_values[x_ind, y_ind] = z

        algo_time = round(time() - start_time, 1)
        print(f"kernel = {kernel}; time of local_max_finder (mp): {algo_time}s; number of QDs: {self.number_of_qds}; multiple counts: {self.multiple_counts}")

############################################################################################################################################################################

    def local_max_core(self, kernel=1):
        assert isinstance(kernel, int) and kernel > 0

        algo_start = time()
        peaks_x, peaks_y = self.localmax_ind
        self.multiple_counts = 0
        new_peaks = deepcopy(self.local_max_values)

        for x, y in zip(peaks_x, peaks_y):
            if 0 <= y - kernel and y + kernel <= self.Ny - 1 and 0 <= x - kernel and x + kernel <= self.Nx - 1:
                near_x = np.arange(x - kernel, x + kernel + 1, 1)
                near_y = np.arange(y - kernel, y + kernel + 1, 1)
            else:
                for i in np.arange(0, kernel + 1, 1):
                    if 0 <= y - kernel and y + kernel <= self.Ny - 1 and 0 <= x - kernel and x + i <= self.Nx - 1:
                        near_x = np.arange(x - kernel, x + i + 1, 1)
                        near_y = np.arange(y - kernel, y + kernel + 1, 1)
                    elif 0 <= y - kernel and y + kernel <= self.Ny - 1 and 0 <= x - i and x + kernel <= self.Nx - 1:
                        near_x = np.arange(x - i, x + kernel + 1, 1)
                        near_y = np.arange(y - kernel, y + kernel + 1, 1)
                    elif 0 <= y - kernel and y + i <= self.Ny - 1 and 0 <= x - kernel and x + kernel <= self.Nx - 1:
                        near_x = np.arange(x - kernel, x + kernel + 1, 1)
                        near_y = np.arange(y - kernel, y + i + 1, 1)
                    elif 0 <= y - i and y + kernel <= self.Ny - 1 and 0 <= x - kernel and x + kernel <= self.Nx - 1:
                        near_x = np.arange(x - kernel, x + kernel + 1, 1)
                        near_y = np.arange(y - i, y + kernel + 1, 1)
                    else:
                        for j in np.arange(0, kernel + 1, 1):
                            if y - i < 0 == y - i + 1 and x - j < 0 == x - j + 1:
                                near_x = np.arange(x - j + 1, x + kernel + 1, 1)
                                near_y = np.arange(y - i + 1, y + kernel + 1, 1)
                            elif y + i > self.Ny - 1 == y + i - 1 and x + j > self.Nx - 1 == x + j - 1:
                                near_x = np.arange(x - kernel, x + j, 1)
                                near_y = np.arange(y - kernel, y + i, 1)
                            elif y + i > self.Ny - 1 == y + i - 1 and x - j < 0 == x - j + 1:
                                near_x = np.arange(x - j + 1, x + kernel + 1, 1)
                                near_y = np.arange(y - kernel, y + i, 1)
                            elif y - i < 0 == y - i + 1 and x + j > self.Nx - 1 == x + j - 1:
                                near_x = np.arange(x - kernel, x + j, 1)
                                near_y = np.arange(y - i + 1, y + kernel + 1, 1)
            z_tmp = self.z_values[x, y]
            multiple_count_tmp = 0
            for nx in near_x:
                for ny in near_y:
                    if self.z_values[nx, ny] > z_tmp:
                        new_peaks[x, y] = 0
                        break
                    elif self.z_values[nx, ny] == z_tmp:
                        multiple_count_tmp += 1
                else:
                    continue
                break

            if multiple_count_tmp >= 2:
                self.multiple_counts +=1
        
        self.local_max_values = deepcopy(new_peaks)
        self.localmax_ind = self.local_max_values.nonzero()
        self.number_of_qds = self.localmax_ind[0].size
        algo_time = round(time() - algo_start, 1)
        print(f"kernel = {kernel}; time of local_max_finder (no mp): {algo_time}s; number of QDs: {self.number_of_qds}; multiple counts: {self.multiple_counts}")

#############################################################################################################################################################################

    def local_max_iterator(self, method="mp"):
        if self.max_kernel > 1:
            kernel_list = [1, self.max_kernel]
        else:
            kernel_list = list(range(1, self.max_kernel + 1))
        
        print("-"*100)
        print()
        print("start of local_max_iterator()")
        print()
        algo_start = time()
        if method == "mp":
            for i in kernel_list:
                self.mp_routine(i)
            algo_time = round(time() - algo_start, 1)
            print()
            print(f"Finished local_max_iterator() in {algo_time}s")
        elif method == "no_mp":
            for i in kernel_list:
                self.local_max_core(kernel=i)
            algo_time = round(time() - algo_start, 1)
            print()
            print(f"Finished local_max_iterator() in {algo_time}s")
        elif method == "fastest":
            self.mp_routine(kernel_list[0])
            for i in kernel_list[1:-1]:
                self.local_max_core(kernel=i)
            self.mp_routine(kernel_list[-1])
            self.localmax_ind = self.local_max_values.nonzero()
            algo_time = round(time() - algo_start, 1)
            print()
            print(f"Finished local_max_iterator() in {algo_time}s")
            print()
        else:
            raise NotImplementedError("Use one of the provided methods: 'fastest', 'mp' or 'no_mp'")


# run tests to find out, which combination is the fastest