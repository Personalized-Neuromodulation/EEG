import numpy as np
from scipy import stats
import itertools
from operator import itemgetter
from multiprocessing import Pool
from collections import OrderedDict
import copy

class MeanMicrostate:
    def __init__(self, data, n_k, n_ch, n_condition):
        self.data = data
        self.n_k = n_k
        self.n_ch = n_ch
        self.n_condition = n_condition
        data_concatenate = np.zeros((1, n_ch))
        for i in range(n_condition):
            data_concatenate = np.concatenate((data_concatenate, data[i]), axis=0)

        self.data_concatenate = data_concatenate[1::, :]

    def label_two_microstates(self, microstates, mean_microstates, polarity=False):
        similarity_matrix = np.zeros((self.n_k, self.n_k))
        for i in range(self.n_k):
            for j in range(self.n_k):
                similarity_matrix[i, j] = stats.pearsonr(microstates[i], mean_microstates[j])[0]

        comb = [zip(perm, [i for i in range(self.n_k)]) for perm in itertools.permutations([j for j in range(self.n_k)], self.n_k)]
        res = []
        for i, item in enumerate(comb):
            s = 0
            comb_list = []
            sign = []
            for item_j in item:
                comb_list.append(item_j)
                similarity = similarity_matrix[item_j[0], item_j[1]]
                if similarity < 0:
                    sign.append(-1)
                else:
                    sign.append(1)
                s = s + similarity if polarity else s + abs(similarity)
            res.append((s/len(comb_list), comb_list, sign))
        sorted_res = sorted(res, key=itemgetter(0), reverse=True)
        return sorted_res[0]

    def label_microstates(self, mul_microstates, mean_microstates, polarity=False):
        label = []
        sign = []
        similarity = []
        for microstates in mul_microstates:
            s = self.label_two_microstates(microstates, mean_microstates, polarity)
            for index, item in enumerate(s[1]):
                label.append(item[0])
            sign.extend(s[2])
            similarity.append(s[0])
        return label, sign, np.mean(similarity), np.std(similarity)

    # def reorder_microstates(self, mul_microstates, mean_microstates, polarity=False):
    #     res = []
    #     for microstates in mul_microstates:
    #         s = self.label_two_microstates(microstates, mean_microstates, polarity)
    #         sorted_index = [i[0] for i in s[1]]
    #         sign = np.repeat(s[2], self.n_ch).reshape(-1, self.n_ch)
    #         microstate_updated = np.asarray(microstates)[sorted_index] * sign
    #         res.append(microstate_updated.tolist())
    #     return res

    def update_mean_microstates(self, label, sign, polarity=False):
        label = np.asarray(label)
        mean_microsate_updated = np.zeros((self.n_k, self.n_ch))
        for i in range(self.n_k):
            index = np.argwhere(label == i).reshape(-1)
            maps = self.data_concatenate[index, :]
            # if not polarity:
            temp = np.asarray(sign)[index].reshape(self.n_condition, 1)
            temp = np.repeat(temp, self.n_ch, axis=1)
            maps = maps * temp
            mean_microsate_updated[i, :] = np.mean(maps, axis=0)
        return mean_microsate_updated

    def mean_microstates(self, n_runs=200, maxiter=1000):
        n_data_concatenate = len(self.data_concatenate)
        maps_list = []
        label_list = []
        mean_similarity_list = []
        std_similarity_list = []
        for run in range(n_runs):
            # print(run)
            mean_similarity_run = []
            std_similarity_run = []
            label_run = []
            maps_run = []
            rndi = np.random.permutation(n_data_concatenate)[:self.n_k]
            # maps = Microstate.normalization(self.data_concatenate[rndi, :], axis=1)

            maps = self.data_concatenate[rndi, :]
            label, sign, mean_similarity, std_similarity = self.label_microstates(self.data, maps, False)
            iter_num = 0
            while iter_num < maxiter:
                iter_num += 1
                maps_updated = self.update_mean_microstates(label, sign, False)
                label_updated, sign, mean_similarity, std_similarity = self.label_microstates(self.data, maps_updated, False)
                if label == label_updated:
                    maps_list.append(maps_updated)
                    label_list.append(label)
                    mean_similarity_list.append(mean_similarity)
                    std_similarity_list.append(std_similarity)
                    break
                else:
                    mean_similarity_run.append(mean_similarity)
                    std_similarity_run.append(std_similarity)
                    label_run.append(label_updated)
                    maps_run.append(maps_updated)
                    label = label_updated
            else:
                index = np.argmax(np.asarray(mean_similarity_run))
                mean_similarity_list.append(mean_similarity_run[index])
                std_similarity_list.append(std_similarity_run[index])
                label_list.append(label_run[index])
                maps_list.append(maps_run[index])
        index = np.argmax(mean_similarity_list)
        return maps_list[index], label_list[index], mean_similarity_list[index], std_similarity_list[index]

    @staticmethod
    def reorder_microstate(maps, ms_template=None, order=None, sign=None):
        n_k = maps.shape[0]
        n_ch = maps.shape[1]
        if ms_template is not None:
            ms = MeanMicrostate(maps, n_k, n_ch, 0)
            s = ms.label_two_microstates(maps, ms_template)
            sorted_index = [i[0] for i in s[1]]
            sign = np.repeat(s[2], ms.n_ch).reshape(-1, ms.n_ch)
            reorder_maps = np.asarray(maps)[sorted_index] * sign
        else:
            reorder_maps = copy.deepcopy(maps)
            for index, j in enumerate(order):
                reorder_maps[index, :] = maps[j] * sign[index]

        return reorder_maps