import numpy as np
from itertools import combinations
from utils.io.file import to_string
from utils.math.numerics import nCr
import lzma


class MicrostateLRD:
    def __init__(self, sequence, n_microstate):
        self.sequence = sequence
        self.n_microstate = n_microstate
        self.n_sequence = len(sequence)

    def partition_state(self, k):
        comb = combinations([i for i in range(self.n_microstate)], k)
        length = nCr(self.n_microstate, k)
        res = []
        for index, item in enumerate(comb):
            if k % 2 == 0:
                if index == length / 2:
                    break
                else:
                    res.append(item)
            else:
                res.append(item)
        return res


    def embed_random_walk(self, data, k):
        partitions = self.partition_state(k)
        np_sequence = np.asarray(data)
        res = {}
        for item in partitions:
            temp_x = np.ones(len(data)) * -1
            for state in item:
                temp_x[np.where(np_sequence == state)[0]] = 1
            res[to_string(item)] = temp_x
        return res

    @staticmethod
    def detrend(embed_sequence, window_size):
        shape = (embed_sequence.shape[0] // window_size, window_size)
        temp = np.lib.stride_tricks.as_strided(embed_sequence, shape)
        window_size_index = np.arange(window_size)
        res = np.zeros(shape[0])
        for index, y in enumerate(temp):
            coeff = np.polyfit(window_size_index, y, 1)
            y_hat = np.polyval(coeff, window_size_index)
            res[index] = np.sqrt(np.mean((y - y_hat) ** 2))
        return res

    @staticmethod
    def dfa(embed_sequence, segment_range, segment_density):
        y = np.cumsum(embed_sequence - np.mean(embed_sequence))
        scales = (2 ** np.arange(segment_range[0], segment_range[1], segment_density)).astype(np.int)
        f = np.zeros(len(scales))
        for index, window_size in enumerate(scales):
            f[index] = np.sqrt(np.mean(MicrostateLRD.detrend(y, window_size) ** 2))
        coeff = np.polyfit(np.log2(scales), np.log2(f), 1)
        return {'slope': coeff[0], 'fluctuation': f.tolist(), 'scales': scales.tolist()}

    @staticmethod
    def shanon_entropy(x, nx, ns):
        p = np.zeros(ns)
        for t in range(nx):
            p[x[t]] += 1.0
        p /= nx
        return -np.sum(p[p > 0] * np.log2(p[p > 0]))

    @staticmethod
    def shanon_joint_entropy(x, y, nx, ny, ns):
        n = min(nx, ny)
        p = np.zeros((ns, ns))
        for t in range(n):
            p[x[t], y[t]] += 1.0
        p /= n
        return -np.sum(p[p > 0] * np.log2(p[p > 0]))

    @staticmethod
    def shanon_joint_entropy_k(x, nx, ns, k):
        p = np.zeros(tuple(k * [ns]))
        for t in range(nx - k):
            p[tuple(x[t:t + k])] += 1.0
        p /= (nx - k)
        h = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        return h

    def mutual_information(self, lag):
        lag = min(self.n_sequence, lag)
        res = np.zeros(lag)
        for time_lag in range(lag):
            nmax = self.n_sequence - time_lag
            h = self.shanon_entropy(self.sequence[:nmax], nmax, self.n_microstate)
            h_lag = self.shanon_entropy(self.sequence[time_lag:time_lag + nmax], nmax, self.n_microstate)
            h_h_lag = self.shanon_joint_entropy(self.sequence[:nmax], self.sequence[time_lag:time_lag + nmax], nmax,
                                                nmax, self.n_microstate)
            res[time_lag] = h + h_lag - h_h_lag
        return res

    def partial_mutual_information(self, lag):
        p = np.zeros(lag)
        a = self.mutual_information(2)
        p[0], p[1] = a[0], a[1]
        for k in range(2, lag):
            h1 = MicrostateLRD.shanon_joint_entropy_k(self.sequence, self.n_sequence, self.n_microstate,
                                                                      lag)
            h2 = MicrostateLRD.shanon_joint_entropy_k(self.sequence, self.n_sequence, self.n_microstate,
                                                                      lag - 1)
            h3 = MicrostateLRD.shanon_joint_entropy_k(self.sequence, self.n_sequence, self.n_microstate,
                                                                      lag + 1)
            p[k] = 2 * h1 - h2 - h3
        return p

    def excess_entropy_rate(self, kmax):
        h = np.zeros(kmax)
        for k in range(kmax):
            h[k] = MicrostateLRD.shanon_joint_entropy_k(self.sequence, self.n_sequence,
                                                                        self.n_microstate, k + 1)
        ks = np.arange(1, kmax + 1)
        entropy_rate, excess_entropy = np.polyfit(ks, h, 1)
        return entropy_rate, excess_entropy, ks


    def lempel_ziv_markov_chain(self):
        compress_seq = lzma.compress(bytes(self.sequence))
        return len(compress_seq)

    def seq_dyanmics(self, window, step, partition):
        n_step = 0
        res = {'lempel_ziv': [], 'dfa': {}}
        for i in self.partition_state(partition):
            res['dfa'][to_string(i)] = []
        while (window+step*n_step) <= self.n_sequence:
            seq = self.sequence[step*n_step:window+step*n_step]
            seq_size = self.lempel_ziv(seq)
            seq_emd = self.embed_random_walk(seq, partition)
            segment_range = [2, int(np.log2(len(seq)))]
            segment_density = 0.25
            for key, value in seq_emd.items():
                res['dfa'][key].append(self.dfa(value, segment_range, segment_density))
            res['lempel_ziv'].append(seq_size)
            n_step += 1
        return res