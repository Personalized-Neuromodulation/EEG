"""
滤波、去趋势、ECG伪迹去除、坏道检测等预处理工具
"""
import numpy as np
import mne
from scipy.signal import detrend, filtfilt, iirfilter
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs
import concurrent.futures
import neurokit2 as nk
import os
from natsort import natsorted

# ------------------ 通用工具函数 ------------------
def find_eeg_files(folder_path):
    """递归查找指定目录下的所有 .bdf 或 .edf 文件"""
    file_path_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in natsorted(files):
            if file.endswith('.bdf') or file.endswith('.edf'):
                file_path_list.append(os.path.join(root, file))
    return file_path_list


# ------------------ 滤波与去趋势 ------------------
def filter_detrend(raw):
    """滤波、平均参考、基线校正"""
    raw.load_data()
    eeg_picks = mne.pick_types(raw.info, eeg=True, emg=False, ecg=False, eog=False, misc=False)
    emg_picks = mne.pick_types(raw.info, emg=True, ecg=False, eog=False, misc=False)
    ecg_picks = mne.pick_types(raw.info, emg=False, ecg=True, eog=False, misc=False)
    eog_picks = mne.pick_types(raw.info, emg=False, ecg=False, eog=True, misc=False)

    if len(eeg_picks) > 0:
        raw.filter(l_freq=0.5, h_freq=40, picks=eeg_picks, method='fir', fir_design='firwin')
        if raw.info['sfreq'] > 100:
            raw.notch_filter(freqs=50, picks=eeg_picks, method='fir', fir_design='firwin')

    if len(eog_picks) > 0:
        raw.filter(l_freq=0.5, h_freq=15, picks=eog_picks, method='fir', fir_design='firwin')
        if raw.info['sfreq'] > 100:
            raw.notch_filter(freqs=50, picks=eog_picks, method='fir', fir_design='firwin')

    if len(emg_picks) > 0:
        h_freq = min(100, raw.info['sfreq']/2-1)
        raw.filter(l_freq=15, h_freq=h_freq, picks=emg_picks, method='fir', fir_design='firwin')
        if 200 > raw.info['sfreq'] > 100:
            raw.notch_filter(freqs=50, picks=emg_picks, method='fir', fir_design='firwin')
        if raw.info['sfreq'] > 200:
            raw.notch_filter(freqs=[50,100], picks=emg_picks, method='fir', fir_design='firwin')

    if len(ecg_picks) > 0:
        raw.filter(l_freq=0.5, h_freq=35, picks=ecg_picks, method='fir', fir_design='firwin')

    raw = detrend_data_multithreaded(raw)
    return raw


def detrend_data_multithreaded(raw):
    """多线程去趋势"""
    original_annotations = raw.annotations.copy()
    full_data = raw.get_data()
    n_channels = full_data.shape[0]

    def detrend_channel(args):
        channel_idx, channel_data = args
        return channel_idx, detrend(channel_data, type='linear')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [(i, full_data[i, :].copy()) for i in range(n_channels)]
        results = list(executor.map(detrend_channel, tasks))

    for channel_idx, channel_data in results:
        full_data[channel_idx, :] = channel_data

    new_raw = mne.io.RawArray(full_data, raw.info)
    new_raw.set_annotations(original_annotations)
    print("**************滤波、平均重参考、去趋势完成**************")
    return new_raw


# ------------------ ECG 伪迹去除 ------------------
class MultiChannelECGArtifactRemover:
    """多通道ECG伪迹去除"""
    def __init__(self, fs):
        self.fs = fs
        self.pre_samples = int(0.5 * fs)
        self.post_samples = int(0.5 * fs)
        self.channel_templates = {}
        self.r_peaks = None

    def remove_ecg_from_raw(self, raw, ecg_data):
        self.r_peaks = self._detect_r_peaks(ecg_data)
        clean_raw = raw.copy()
        all_ch_names = [ch for i, ch in enumerate(raw.ch_names)
                        if raw.get_channel_types()[i] != 'ecg']
        for ch_name in all_ch_names:
            ch_data = raw.get_data(picks=[ch_name]).flatten()
            template, pre, post = self.create_ecg_template(ch_data, self.r_peaks)
            cleaned_data = self.subtract_template(ch_data, self.r_peaks, template, pre, post)
            clean_raw._data[raw.ch_names.index(ch_name)] = cleaned_data
        return clean_raw

    def subtract_template(self, signal, r_peaks, template, pre, post):
        corrected_signal = signal.copy()
        for r in r_peaks:
            if r - pre < 0 or r + post > len(signal):
                continue
            corrected_signal[r - pre:r + post] -= template
        return corrected_signal

    def create_ecg_template(self, signal, r_peaks):
        pre = self.pre_samples
        post = self.post_samples
        segments = []
        for r in r_peaks:
            if r - pre < 0 or r + post > len(signal):
                continue
            segment = signal[r - pre:r + post]
            segments.append(segment)
        segments = np.array(segments)
        template = np.median(segments, axis=0)
        return template, pre, post

    def _detect_r_peaks(self, ecg_reference):
        ecg_cleaned = nk.ecg_clean(ecg_reference, sampling_rate=self.fs, method="neurokit")
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.fs)
        return rpeaks["ECG_R_Peaks"]


# ------------------ 坏通道检测 ------------------
class CleanRawData:
    """检测坏通道"""
    def __init__(self):
        self.flat_duration = 5.0
        self.flat_threshold = 1e-6
        self.noise_threshold = 4
        self.hp_cutoff = 20.0
        self.lp_cutoff = 30.0
        self.corr_threshold = 0.8
        self.n_neighbors = 3

    def detect_bad_channels(self, raw):
        flat_bad = self.detect_flat_channels(raw)
        hf_noise_bad = self.detect_hf_noise_channels(raw)
        low_corr_bad = self.detect_low_correlation_channels(raw)
        all_bad = list(set(flat_bad + hf_noise_bad + low_corr_bad))
        print(f"检测到平坦通道: {flat_bad}")
        print(f"检测到高频噪声通道: {hf_noise_bad}")
        print(f"检测到低相关性通道: {low_corr_bad}")
        print(f"总共检测到坏通道: {len(all_bad)}个")
        return all_bad

    def detect_flat_channels(self, raw, win_size=1.0):
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        n_channels, n_times = data.shape
        win_samples = int(win_size * sfreq)
        n_windows = int(n_times / win_samples)
        bad_channels = []
        for i in range(n_channels):
            flat_count = 0
            max_flat_count = 0
            for win in range(n_windows):
                start = win * win_samples
                end = min((win + 1) * win_samples, n_times)
                segment = data[i, start:end]
                seg_range = np.max(segment) - np.min(segment)
                if seg_range < self.flat_threshold:
                    flat_count += 1
                    if flat_count > max_flat_count:
                        max_flat_count = flat_count
                else:
                    flat_count = 0
            max_flat_duration = max_flat_count * win_size
            if max_flat_duration >= self.flat_duration:
                bad_channels.append(raw.ch_names[i])
        return bad_channels

    def detect_hf_noise_channels(self, raw):
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        n_channels = data.shape[0]
        noisiness = np.zeros(n_channels)
        for i in range(n_channels):
            ch_data = data[i, :]
            hp_b, hp_a = iirfilter(N=4, Wn=self.hp_cutoff, btype='highpass', fs=sfreq, ftype='butter')
            lp_b, lp_a = iirfilter(N=4, Wn=self.lp_cutoff, btype='lowpass', fs=sfreq, ftype='butter')
            hp_data = filtfilt(hp_b, hp_a, ch_data)
            lp_data = filtfilt(lp_b, lp_a, ch_data)
            mad_hp = stats.median_abs_deviation(hp_data, scale='normal')
            mad_lp = stats.median_abs_deviation(lp_data, scale='normal')
            noisiness[i] = mad_hp / mad_lp if mad_lp > 0 else np.inf
        median_noise = np.median(noisiness)
        mad_noise = stats.median_abs_deviation(noisiness, scale='normal')
        znoise = (noisiness - median_noise) / mad_noise
        bad_channels = [raw.ch_names[i] for i in range(n_channels) if znoise[i] > self.noise_threshold]
        return bad_channels

    def detect_low_correlation_channels(self, raw):
        eeg_raw = raw.copy().pick_types(eeg=True)
        if len(eeg_raw.ch_names) == 0:
            return []
        data = eeg_raw.get_data()
        ch_names = eeg_raw.ch_names
        ch_pos = []
        valid_positions = []
        for i, ch in enumerate(eeg_raw.info['chs']):
            loc = ch['loc'][:3]
            if not np.isnan(loc).any() and np.linalg.norm(loc) > 1e-6:
                valid_positions.append(i)
                ch_pos.append(loc)
        ch_pos = np.array(ch_pos)
        if len(ch_pos) < 2:
            return []
        nbrs = NearestNeighbors(n_neighbors=min(self.n_neighbors+1, len(ch_pos)))
        nbrs.fit(ch_pos)
        _, indices = nbrs.kneighbors(ch_pos)
        correlations = np.zeros(len(eeg_raw.ch_names))
        for i, orig_idx in enumerate(valid_positions):
            ch_data = data[orig_idx, :]
            neighbor_indices = indices[i, 1:]
            neighbor_indices = neighbor_indices.astype(int)
            if len(neighbor_indices) == 0:
                correlations[orig_idx] = 0
                continue
            neighbor_orig_indices = [valid_positions[j] for j in neighbor_indices]
            neighbor_data = data[neighbor_orig_indices, :].T
            model = RANSACRegressor(min_samples=min(3, len(neighbor_orig_indices)), stop_probability=0.98)
            try:
                model.fit(neighbor_data, ch_data)
                pred_data = model.predict(neighbor_data)
                corr = np.corrcoef(pred_data, ch_data)[0, 1]
                correlations[orig_idx] = np.abs(corr)
            except:
                correlations[orig_idx] = 0
        print(correlations)
        bad_channels = [ch_names[i] for i in range(len(ch_names)) if correlations[i] < self.corr_threshold]
        return bad_channels

