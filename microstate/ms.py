import mat73
import numpy as np
import cupy as cp
import mne
import multiprocessing
from multiprocessing import Pool
from scipy.io import savemat, loadmat

from configuration.config import PATH_CONFIG


class Microstate:
    def __init__(self, data, cuda=False):
        self.data = Microstate.substract_mean(data)
        self.n_t = self.data.shape[0]
        self.n_ch = self.data.shape[1]
        # self.gfp = np.std(self.data, axis=1)
        # self.gfp_peaks = Microstate.locmax(self.gfp)
        if cuda:
            self.gfp = cp.std(self.data, axis=1)
            self.gfp_peaks = Microstate.locmax_cuda(self.gfp)
            self.gfp_values = self.gfp[self.gfp_peaks]
            self.n_gfp = self.gfp_peaks.shape[0]
            # self.sum_gfp2 = np.sum(self.gfp_values**2)
            self.sum_gfp2 = cp.sum(self.gfp_values**2)
        else:
            self.gfp = np.std(self.data, axis=1)
            self.gfp_peaks = Microstate.locmax(self.gfp)
            self.gfp_values = self.gfp[self.gfp_peaks]
            self.n_gfp = self.gfp_peaks.shape[0]
            self.sum_gfp2 = np.sum(self.gfp_values**2)

    @staticmethod
    def locmax(x):
        dx = np.diff(x)
        zc = np.diff(np.sign(dx))
        m = 1 + np.where(zc == -2)[0]
        return m
    
    @staticmethod
    def locmax_cuda(x):
        dx = cp.diff(x)
        zc = cp.diff(cp.sign(dx))
        m = 1 + cp.where(zc == -2)[0]
        return m

    @staticmethod
    def substract_mean(x):
        return x - x.mean(axis=1, keepdims=True)

    @staticmethod
    def assign_labels_kmeans(data, maps, n_ch, gfp, gfp_peaks=None):
        c = np.dot(data, maps.T)
        if isinstance(gfp_peaks, np.ndarray):
            c /= (n_ch * np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
        else:
            c /= (n_ch * np.outer(gfp, np.std(maps, axis=1)))
        l = np.argmax(c ** 2, axis=1)

        return l, c
    

    # def assign_labels_kmeans_cuda(data, maps, n_ch, gfp, gfp_peaks=None):
    #     data = cp.asnumpy(data)
    #     maps = cp.asnumpy(maps)
    #     gfp = cp.asnumpy(gfp)


    #     c = np.dot(data, maps.T)
    #     if isinstance(gfp_peaks, np.ndarray):
    #         c /= (n_ch * np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
    #     else:
    #         c /= (n_ch * np.outer(gfp, np.std(maps, axis=1)))
    #     l = np.argmax(c ** 2, axis=1)
        
    #     l = cp.asarray(l)
    #     c = cp.asarray(c)
        
    #     return l, c
    

    def assign_labels_kmeans_cuda(data, maps, n_ch, gfp, gfp_peaks=None):
        # Convert data and maps to GPU arrays
        data = cp.asarray(data)
        maps = cp.asarray(maps)
        
        # Ensure gfp is a CuPy array
        gfp = cp.asarray(gfp)
        
        # Check if gfp_peaks is provided, and convert it to a CuPy array if necessary
        if gfp_peaks is not None:
            gfp_peaks = cp.asarray(gfp_peaks)
        
        # Compute the matrix product of data and maps.T
        c = cp.dot(data, maps.T)
        
        # If GFP peaks are provided, normalize accordingly
        if gfp_peaks is not None:
            c /= (n_ch * cp.outer(gfp[gfp_peaks], cp.std(maps, axis=1)))
        else:
            c /= (n_ch * cp.outer(gfp, cp.std(maps, axis=1)))
        
        # Find the labels by taking the argmax of the squared values
        l = cp.argmax(c ** 2, axis=1)
        return l, c
    

    def fit_back(self, maps, threshold=None):
        c = np.dot(self.data, maps.T) / (self.n_ch * np.outer(self.gfp, maps.std(axis=1)))
        c = abs(c)
        c_max = np.max(c, axis=1)
        c_max_index = np.argmax(c, axis=1)
        if threshold:
            c_threshold_index = np.where(c_max > threshold)[0]
            l = c_max_index[c_threshold_index]
        else:
            l = c_max_index
        return l

    def fit_back_peaks(self, maps, threshold=None):
        c = np.dot(self.data[self.gfp_peaks], maps.T) / (
                    self.n_ch * np.outer(self.gfp[self.gfp_peaks], maps.std(axis=1)))
        c = abs(c)
        c_max = np.max(c, axis=1)
        # c_mean = np.mean(c_max, axis=0)
        c_max_index = np.argmax(c, axis=1)
        res = []
        # add
        # clean_info_path = PATH_CONFIG['root'] + f"/derivatives/HAPPE_CUSTOMIZE/sub-YZISM{PATH_CONFIG['subject']}/sleep/ses-sleep0/clean_info/baseline.mat"
        # clean_info = loadmat(clean_info_path)['info'][0][0]['bad_epochs'][0, ::]
        # clean_epochs_index = np.where(clean_info != 0)[0]

        # clean_data_path = PATH_CONFIG['root'] + f"/derivatives/HAPPE_CUSTOMIZE/sub-YZISM{PATH_CONFIG['subject']}/sleep/ses-sleep0/clean_data/baseline.set"
        # clean_data = mne.io.read_epochs_eeglab(clean_data_path)
        # drop_index = []
        # for i in clean_epochs_index:
        #     drop_index += list(range(i*2*250, (i+1)*2*250))
        # total_epochs = np.arange(clean_data.get_data().shape[0] * 2 * 250)
        # # 获取未被删除的 epochs 索引
        # kept_epochs = np.setdiff1d(total_epochs, drop_index)

        # # 构建完整索引 → 当前索引（去除丢弃的）
        # full_to_kept_map = {i: kept_epochs[i] for i in range(len(kept_epochs))}
        # gfp_peaks_adjusted = np.array([kept_epochs[i] for i in self.gfp_peaks])

        if threshold:
            c_threshold_index = np.where(c_max > threshold)[0]
            l = c_max_index[c_threshold_index]
            gfp_peaks = self.gfp_peaks[c_threshold_index]
        else:
            l = c_max_index
            gfp_peaks = self.gfp_peaks
            # gfp_peaks = gfp_peaks_adjusted
        for i in range(0, len(gfp_peaks) - 1):
            med_point = (gfp_peaks[i] + gfp_peaks[i + 1]) // 2
            res += [l[i] for j in range(med_point - gfp_peaks[i])] + [l[i + 1] for j in
                                                                      range(gfp_peaks[i + 1] - med_point)]
        return np.asarray(res)

    def gev(self, maps):
        n_maps = len(maps)
        c = np.dot(self.data[self.gfp_peaks], maps.T)
        c /= (self.n_ch * np.outer(self.gfp[self.gfp_peaks], np.std(maps, axis=1)))
        l = np.argmax(c ** 2, axis=1)
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = l == k
            gev[k] = np.sum(self.gfp_values[r] ** 2 * c[r, k] ** 2) / self.sum_gfp2
        return gev, np.sum(gev)

    def wcss(self, maps):
        c = np.dot(self.data[self.gfp_peaks], maps.T) / (self.n_ch * np.outer(self.gfp[self.gfp_peaks], maps.std(axis=1)))
        c = abs(c)
        c_max = np.max(c, axis=1)
        c_max_index = np.argmax(c, axis=1)
        l = c_max_index
        gfp_peaks = self.gfp_peaks


    def kl_criterion(self, wcss):
        kl_values = []
        for k in range(1, len(wcss)-1):
            numerator = np.abs(wcss[k+1] - 2 * wcss[k] + wcss[k-1])
            denominator = wcss[k]
            kl_value = numerator / denominator
            kl_values.append(kl_value)
        return np.array(kl_values)



    def kmeans(self, n_maps, maxerr=1e-6, maxiter=1000):
        np.random.seed()
        rndi = np.random.permutation(self.n_gfp)[:n_maps]
        data_gfp = self.data[self.gfp_peaks, :]
        sum_v2 = np.sum(data_gfp ** 2)
        maps = data_gfp[rndi, :]
        maps /= np.sqrt(np.sum(maps ** 2, axis=1, keepdims=True))
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        while ((np.abs((var0 - var1) / var0) > maxerr) & (n_iter < maxiter)):
            l_peaks, c = Microstate.assign_labels_kmeans(data_gfp, maps, self.n_ch, self.gfp, self.gfp_peaks)
            for k in range(n_maps):
                vt = data_gfp[l_peaks == k, :]
                sk = np.dot(vt.T, vt)
                evals, evecs = np.linalg.eig(sk)
                v = np.real(evecs[:, np.argmax(np.abs(evals))])
                maps[k, :] = v / np.sqrt(np.sum(v ** 2))
            var1 = var0
            var0 = sum_v2 - np.sum(np.sum(maps[l_peaks, :] * data_gfp, axis=1) ** 2)
            var0 /= (self.n_gfp * (self.n_ch - 1))
            n_iter += 1
        l, _ = Microstate.assign_labels_kmeans(self.data, maps, self.n_ch, self.gfp)
        var = np.sum(self.data ** 2) - np.sum(np.sum(maps[l, :] * self.data, axis=1) ** 2)
        var /= (self.n_t * (self.n_ch - 1))
        cv = var * (self.n_ch - 1) ** 2 / (self.n_ch - n_maps - 1.) ** 2
        return maps, l, cv


    def kmeans_cuda(self, n_maps, maxerr=1e-6, maxiter=1000):
        # Use CuPy's random seed and permutation
        # self.data = cp.asarray(self.data)

        cp.random.seed()
        rndi = cp.random.permutation(self.n_gfp)[:n_maps]

        data_gfp = cp.asarray(self.data[self.gfp_peaks, :])
        sum_v2 = cp.sum(data_gfp ** 2)
        
        maps = data_gfp[rndi, :]
        maps /= cp.sqrt(cp.sum(maps ** 2, axis=1, keepdims=True))
        
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        
        # Main loop
        while (cp.abs((var0 - var1) / var0) > maxerr) & (n_iter < maxiter):
            l_peaks, c = Microstate.assign_labels_kmeans_cuda(data_gfp, maps, self.n_ch, self.gfp, self.gfp_peaks)
            for k in range(n_maps):
                vt = data_gfp[l_peaks == k, :]
                sk = cp.dot(vt.T, vt)
                
                evals, evecs = cp.linalg.eigh(sk)
                v = cp.real(evecs[:, cp.argmax(cp.abs(evals))])
                maps[k, :] = v / cp.sqrt(cp.sum(v ** 2))
            
            var1 = var0
            var0 = sum_v2 - cp.sum(cp.sum(maps[l_peaks, :] * data_gfp, axis=1) ** 2)
            var0 /= (self.n_gfp * (self.n_ch - 1))
            n_iter += 1

        # self.data = cp.asarray(self.data)
        l, _ = Microstate.assign_labels_kmeans_cuda(self.data, maps, self.n_ch, self.gfp)
        
        
        var = cp.sum(self.data ** 2) - cp.sum(cp.sum(maps[l, :] * self.data, axis=1) ** 2)
        var /= (self.n_t * (self.n_ch - 1))
        
        cv = var * (self.n_ch - 1) ** 2 / (self.n_ch - n_maps - 1.) ** 2
        
        l = cp.asnumpy(l)
        maps = cp.asnumpy(maps)
        cv = cp.asnumpy(cv).tolist()
        # self.data = cp.asnumpy(self.data)
        # Move everything back to CPU and return results
        
        return maps, l, cv


    def wrap_kmeans(self, para):
        return self.kmeans_cuda(para)

    def kmeans_repetition(self, n_repetition, n_maps, n_pool=11):
        l_list = []
        cv_list = []
        maps_list = []
        # pool = Pool(n_pool)
        multi_res = []
        # tasks = [(n_maps,) for _ in range(n_repetition)]
        # self.data = cp.asarray(self.data)
        for i in range(n_repetition):
            multi_res.append(self.wrap_kmeans(n_maps))
            # multi_res.append(pool.apply_async(self.wrap_kmeans, ([n_maps],)))
        # pool.close()
        # pool.join()
        # results = pool.starmap(self.wrap_kmeans, tasks)

        # 关闭池并等待进程结束
        # pool.close()
        # pool.join()

        for i in range(n_repetition):
            # temp = multi_res[i].get()
            temp = multi_res[i]
            maps_list.append(temp[0])
            l_list.append(temp[1])
            cv_list.append(temp[2])

        k_opt = np.argmin(cv_list)
        return maps_list[k_opt], cv_list[k_opt]

    def microstate(self, max_maps, n_repetition, n_pool=11, is_single=None):
        maps_list = []
        cv_list = []
        if is_single is not None:
            temp = self.kmeans_repetition(n_repetition, is_single, n_pool)
            maps_list.append(temp[0].tolist())
            cv_list.append(temp[1])
        else:
            for n_maps in range(4, max_maps + 1):
                print('n_maps:', n_maps)
                temp = self.kmeans_repetition(n_repetition, n_maps, n_pool)
                maps_list.append(temp[0].tolist())
                cv_list.append(temp[1])
        return maps_list, cv_list


