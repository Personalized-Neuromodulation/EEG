"""
预处理主类 preprocess_EEG
"""
import os
import numpy as np
import mne
import logging
import gc
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from datetime import datetime
from channel_utils import handle_bipolar_channels, channel_type_detection
from preprocessing_utils import filter_detrend, MultiChannelECGArtifactRemover, CleanRawData
from ica_faster import ICA_EEG, FASTER_EEG


def detect_stim_intervals(raw, start_prefix='Start of stimulation', end_prefix='End of stimulation'):
    stim_intervals = []
    start_events = {}
    end_events = {}
    for ann in raw.annotations:
        desc = str(ann['description'])
        if desc.startswith(start_prefix):
            match = re.search(r'\[(\d+),\s*(\d+)\]', desc)
            if match:
                event_id = tuple(map(int, match.groups()))
                start_events[event_id] = ann['onset']
        elif desc.startswith(end_prefix):
            match = re.search(r'\[(\d+),\s*(\d+)\]', desc)
            if match:
                event_id = tuple(map(int, match.groups()))
                end_events[event_id] = ann['onset'] + ann['duration']
    for e_id in sorted(set(start_events.keys()) | set(end_events.keys())):
        start = start_events.get(e_id)
        end = end_events.get(e_id)
        if start and end and start < end:
            stim_intervals.append((start-2, end+2))
    n_samples = len(raw.times)
    sfreq = raw.info['sfreq']
    non_stim_mask = np.ones(n_samples, dtype=bool)
    non_stim_mask_time = np.ones(int(raw.times.shape[-1]/sfreq), dtype=bool)
    for start, end in stim_intervals:
        start_idx = int(start * sfreq)
        end_idx = int(end * sfreq)
        non_stim_mask[start_idx:min(end_idx, n_samples)] = False
        non_stim_mask_time[int(start):min(int(end), int(raw.times.shape[-1]/sfreq))] = False
    non_stim_data = raw.get_data()[:, non_stim_mask]
    non_stim_raw = mne.io.RawArray(non_stim_data, raw.info.copy())
    return non_stim_raw, non_stim_mask, non_stim_mask_time


class preprocess_EEG:
    def __init__(self, file_paths, only_include_EEG_channels=True):
        if len(file_paths) <= 1 or isinstance(file_paths, str):
            if isinstance(file_paths, str):
                self.file_path = file_paths
                file_list = [file_paths]
            else:
                self.file_path = file_paths[0]
                file_list = file_paths
            if os.path.basename(self.file_path).endswith('.bdf'):
                self.raw = mne.io.read_raw_bdf(self.file_path, preload=True)
            elif os.path.basename(self.file_path).endswith('.edf'):
                self.raw = mne.io.read_raw_edf(self.file_path, preload=True)
            elif os.path.basename(self.file_path).endswith('.fif'):
                self.raw = mne.io.read_raw_fif(self.file_path, preload=True)
            else:
                raise ValueError('文件格式错误,请使用bdf或edf格式')
        else:
            if "preprocessed_merge_RAW.edf" in os.listdir(os.path.dirname(file_paths[0])):
                print("已存在预处理合并文件，跳过合并步骤")
                self.raw = mne.io.read_raw_edf(os.path.dirname(file_paths[0]) + '/preprocessed_merge_RAW.edf', preload=True)
            else:
                self.raw = self.merge_bdf_files(file_paths)
                save_path = os.path.dirname(file_paths[0])
                #mne.export.export_raw(save_path + '/preprocessed_merge_RAW.edf', self.raw, fmt='edf', overwrite=True)
                #print(f"Preprocessed data saved to {save_path}/preprocessed_merge_RAW.edf")

        self.original_raw = self.raw.copy()
        self.original_sfreq = self.raw.info['sfreq']
        self.orgin_channel_list = self.raw.info['ch_names']
        self.only_include_EEG_channels = only_include_EEG_channels
        print(f"总时长：{self.raw.times[-1]}s")
        self.raw = handle_bipolar_channels(self.raw)
        self.raw = channel_type_detection(self.raw)
        self.raw = filter_detrend(self.raw)
        self.raw.load_data()
        self.resample(new_fs=128)
        self.get_non_eeg_channels()

    def remove_ecg(self):
        logging.info("正在去除心电伪迹...")
        if self.ecg_reference is not None:
            ecg_remover = MultiChannelECGArtifactRemover(self.raw.info['sfreq'])
            self.raw = ecg_remover.remove_ecg_from_raw(self.raw, self.ecg_reference)
            print("含有心电参考通道，self.raw所有通道的心电伪迹去除完成**********************")
        else:
            print("没有心电参考通道，心电伪迹去除失败**********************")

    def merge_channels(self, eeg_raw, non_eeg_raw):
        if non_eeg_raw is None:
            return eeg_raw
        merged = eeg_raw.copy()
        merged.add_channels([non_eeg_raw], force_update_info=True)
        return merged

    def get_non_eeg_channels(self):
        channel_types = self.raw.get_channel_types()
        non_eeg_channels = [ch for i, ch in enumerate(self.raw.ch_names)
                            if channel_types[i] != 'eeg' and channel_types[i] != 'misc']
        if non_eeg_channels:
            ecg_channels = [ch for i, ch in enumerate(self.raw.ch_names) if channel_types[i] == 'ecg']
            if ecg_channels:
                self.ecg_reference = self.raw.get_data(picks=ecg_channels[0])[0, :]
                print(f"使用ECG通道，可去除心电伪迹: {ecg_channels[0]}")
            else:
                self.ecg_reference = None
                print("未找到ECG通道")
            self.remove_ecg()
            self.no_eeg_raw = self.raw.copy().pick_channels(non_eeg_channels)
            emg_channels = [ch for i, ch in enumerate(self.no_eeg_raw.ch_names) if channel_types[i] == 'emg']
            if emg_channels:
                self.emg = self.no_eeg_raw.get_data(picks=emg_channels[0])[0, :]
                print(f"EMG通道: {emg_channels}")
            else:
                self.emg = None
            eog_channels = [ch for i, ch in enumerate(self.no_eeg_raw.ch_names) if channel_types[i] == 'eog']
            if eog_channels:
                self.eog = self.no_eeg_raw.get_data(picks=eog_channels[0])[0, :]
                print(f"EOG通道: {eog_channels}")
            else:
                self.eog = None
        else:
            self.no_eeg_raw = None
            self.ecg_reference = None
            self.emg = None
            self.eog = None
            print("未找到非EEG通道")
        all_non_eeg_channels_shhs = [ch for i, ch in enumerate(self.raw.ch_names) if channel_types[i] != 'eeg']
        self.all_no_eeg_raw = self.raw.copy().pick_channels(all_non_eeg_channels_shhs)

    def plot_emg(self, save_path=r"D:\analysis\long_stim_bdf\test\emg.pdf", dpi=300):
        plt.figure(figsize=(12, 8))
        f, t, Sxx = spectrogram(self.emg * 1e6, fs=self.original_sfreq,
                                nperseg=int(self.original_sfreq)*4,
                                noverlap=int(self.original_sfreq)*2,
                                window='hann')
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot2grid((6, 1), (1, 0), rowspan=5)
        freq_mask = (f >= 15) & (f <= 100)
        f_sub = f[freq_mask]
        Sxx_sub = Sxx[freq_mask, :]
        Sxx_db = 10 * np.log10(Sxx_sub)
        im = ax.pcolormesh(t/60, f_sub, Sxx_db, shading='auto', cmap="viridis", vmin=-40, vmax=20)
        ax.set_yticks([15, 30, 50, 70, 100])
        ax.set_ylabel('Frequency(Hz)', loc='center', labelpad=30)
        ax.set_title('EMG')
        ax.set_xlabel('Time (min)')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"时频图已保存至: {save_path}")
        plt.close()

    def resample(self, new_fs=250):
        self.raw.resample(new_fs, npad="auto")
        self.fs = new_fs
        print(f"******************重采样完成，新采样率：{self.fs}Hz**************")

    def merge_bdf_files(self, file_paths):
        all_raws = []
        all_annotations = []
        cumulative_onset = 0
        first_raw = None
        for i, file_path in enumerate(file_paths):
            if file_path.endswith(".edf"):
                continue
            print(f"\n处理文件 {i+1}/{len(file_paths)}: {file_path}")
            raw = mne.io.read_raw_bdf(file_path, preload=True, units="uV")
            raw.load_data()
            if i == 0:
                first_raw = raw
            original_duration = raw.times[-1]
            print(f"  原始持续时间: {original_duration:.3f}秒")
            print(f"  采样频率: {raw.info['sfreq']:.1f}Hz")
            print(f"  通道数: {len(raw.ch_names)}")
            end_marker_time = None
            if raw.annotations is not None and len(raw.annotations) > 0:
                for idx, desc in enumerate(raw.annotations.description):
                    desc_lower = desc.lower()
                    if "recording ends" in desc_lower:
                        end_marker_time = raw.annotations.onset[idx]
                        end_marker_duration = raw.annotations.duration[idx]
                        end_time = end_marker_time + end_marker_duration
                        print(f"  找到'Recording ends'标记: 在 {end_marker_time:.3f}s, 持续 {end_marker_duration:.3f}s")
                        if end_time < original_duration - 0.1:
                            print(f"  裁剪 {end_time:.3f}s 之后的数据，共去除 {original_duration - end_time:.3f}s")
                            raw.crop(tmax=end_time)
                            if len(raw.annotations) > 0:
                                keep_mask = raw.annotations.onset <= end_time
                                if np.any(keep_mask):
                                    raw.set_annotations(mne.Annotations(
                                        onset=raw.annotations.onset[keep_mask],
                                        duration=raw.annotations.duration[keep_mask],
                                        description=[raw.annotations.description[j] for j in range(len(raw.annotations)) if keep_mask[j]]
                                    ))
                                else:
                                    raw.set_annotations(None)
                            print(f"  裁剪后持续时间: {raw.times[-1]:.3f}s")
                        break
                if end_marker_time is None:
                    print(f"  未找到'Recording ends'标记，保留完整文件")
            else:
                print(f"  文件无annotations，保留完整文件")
            orig_annot = raw.annotations
            if i == 0:
                all_raws.append(raw)
                all_annotations.append(orig_annot)
                cumulative_onset = raw.times[-1]
                print(f"  首个文件结束时间: {cumulative_onset:.3f}秒")
                continue
            current_duration = raw.times[-1]
            modified_annot = None
            if orig_annot is not None and len(orig_annot) > 0:
                new_onset = orig_annot.onset + cumulative_onset
                modified_annot = mne.Annotations(onset=new_onset, duration=orig_annot.duration, description=orig_annot.description)
                print(f"  调整 {len(modified_annot)} 个Annotations")
            else:
                print(f"  没有Annotations需要调整")
                modified_annot = orig_annot
            all_annotations.append(modified_annot)
            all_raws.append(raw)
            cumulative_onset += current_duration
            print(f"  累积时间偏移量: {cumulative_onset:.3f}秒")
        print(f"\n开始拼接 {len(all_raws)} 个文件...")
        if len(all_raws) > 1:
            for idx, raw in enumerate(all_raws):
                print(f"  文件{idx+1}: {raw.times[-1]:.3f}秒, {len(raw.ch_names)}通道, {raw.info['sfreq']:.1f}Hz")
            combined_raw = mne.concatenate_raws(all_raws, preload=False, on_mismatch='ignore')
            combined_raw.load_data()
            if combined_raw.annotations is not None and len(combined_raw.annotations) > 0:
                boundary_count = 0
                for desc in combined_raw.annotations.description:
                    if 'boundary' in desc.lower():
                        boundary_count += 1
                if boundary_count > 0:
                    print(f"  注意: 拼接后检测到 {boundary_count} 个边界标记")
                    remove_boundary_indices = []
                    for idx, desc in enumerate(combined_raw.annotations.description):
                        if 'boundary' in desc.lower():
                            remove_boundary_indices.append(idx)
                    if remove_boundary_indices:
                        keep_indices = [i for i in range(len(combined_raw.annotations)) if i not in remove_boundary_indices]
                        if keep_indices:
                            new_annot = mne.Annotations(
                                onset=combined_raw.annotations.onset[keep_indices],
                                duration=combined_raw.annotations.duration[keep_indices],
                                description=[combined_raw.annotations.description[i] for i in keep_indices]
                            )
                            combined_raw.set_annotations(new_annot)
                        else:
                            combined_raw.set_annotations(None)
                        print(f"  已移除 {len(remove_boundary_indices)} 个边界标记")
            print(f"拼接完成，总持续时间: {combined_raw.times[-1]:.3f}秒")
        else:
            combined_raw = all_raws[0]
            print(f"只有一个文件，无需拼接，持续时间: {combined_raw.times[-1]:.3f}秒")
        del all_raws
        combined_raw.set_meas_date(first_raw.info['meas_date'])
        return combined_raw

    def set_annotations(self, processing_info, epoch_duration):
        data_duration = self.eeg_raw.n_times / self.eeg_raw.info['sfreq']
        bad_epochs_relative = processing_info.get('bad_epochs_indices', [])
        valid_onsets = []
        valid_durations = []
        valid_descriptions = []
        if bad_epochs_relative:
            for i, onset_relative in enumerate(bad_epochs_relative):
                if onset_relative <= data_duration:
                    remaining_duration = data_duration - onset_relative
                    actual_duration = min(epoch_duration, remaining_duration)
                    if actual_duration > 0:
                        valid_onsets.append(onset_relative)
                        valid_durations.append(actual_duration)
                        valid_descriptions.append(f'bad_epoch_{i}')
                else:
                    print(f"  跳过超出范围的epoch: 相对时间={onset_relative:.2f}s, 数据段结束于{data_duration:.2f}s")
        if self.eeg_raw.annotations is not None:
            combined_onset = np.concatenate([self.eeg_raw.annotations.onset, valid_onsets])
            combined_duration = np.concatenate([self.eeg_raw.annotations.duration, valid_durations])
            combined_description = np.concatenate([self.eeg_raw.annotations.description, valid_descriptions])
            from mne import Annotations
            all_annotations = Annotations(onset=combined_onset, duration=combined_duration,
                                          description=combined_description,
                                          orig_time=self.eeg_raw.annotations.orig_time)
            self.eeg_raw.set_annotations(all_annotations)

    def process_main(self, useICA=False, useFASTER=True, remove_bad_channels=False, stim_analysis=False):
        self.eeg_raw = self.raw.copy().pick_types(eeg=True)
        if self.all_no_eeg_raw:
            logging.info(f"EEG通道{self.eeg_raw.info['ch_names']}")
            logging.info(f"非EEG通道{self.all_no_eeg_raw.info['ch_names']}")
        else:
            logging.info(f"EEG通道{self.eeg_raw.info['ch_names']}")
            logging.info("无非EEG通道")
        full_data = np.zeros((len(self.eeg_raw.ch_names), self.eeg_raw._data.shape[1]))
        eeg_raw_good = self.eeg_raw.copy()
        good_channels = self.eeg_raw.info['ch_names'].copy()
        global_bad_channels, bad_channels = [], []
        if stim_analysis:
            print("获取非刺激部分数据")
            non_stim_raw, non_stim_mask, non_stim_mask_time = detect_stim_intervals(self.eeg_raw)
        else:
            non_stim_raw = self.eeg_raw.copy()
        if remove_bad_channels:
            cleaner = CleanRawData()
            cleaner.flat_duration = 5.0
            cleaner.flat_threshold = 1e-6
            cleaner.noise_threshold = 4
            cleaner.corr_threshold = 0.8
            cleaner.n_neighbors = 3
            bad_channels = cleaner.detect_bad_channels(non_stim_raw)
            if bad_channels:
                logging.info(f"CleanRawData检测到坏通道: {bad_channels}, 先移除进行后续处理")
                bad_channel_data = self.eeg_raw.copy().pick(bad_channels).get_data()
                bad_idx = [self.eeg_raw.ch_names.index(ch) for ch in bad_channels]
                full_data[bad_idx] = bad_channel_data[:, :eeg_raw_good._data.shape[1]]
                good_channels = [ch for ch in self.eeg_raw.ch_names if ch not in bad_channels]
                eeg_raw_good = self.eeg_raw.copy().pick(good_channels)
        if stim_analysis:
            logging.info("进行含有刺激marker数据分析处理")
            if useICA:
                logging.info("应用ICA")
                ica = ICA_EEG(eeg_raw_good, stim_analysis=True, non_stim_mask=non_stim_mask)
                eeg_raw_good = ica.clean_raw
            if useFASTER:
                logging.info("应用FASTER")
                faster = FASTER_EEG(eeg_raw_good, stim_analysis=True, epoch_duration=1.0, max_bad_frac=0.3)
                eeg_raw_good, global_bad_channels = faster.clean_raw, faster.global_bad_channels
        else:
            logging.info("进行不含刺激marker实验数据分析处理")
            if useICA:
                logging.info("应用ICA")
                ica = ICA_EEG(eeg_raw_good, stim_analysis=False)
                eeg_raw_good = ica.clean_raw
            if useFASTER:
                logging.info("应用FASTER")
                faster = FASTER_EEG(eeg_raw_good, stim_analysis=False, epoch_duration=1.0, max_bad_frac=0.5)
                eeg_raw_good, global_bad_channels, processing_info = faster.clean_raw, faster.global_bad_channels, faster.processing_info
                self.set_annotations(processing_info, epoch_duration=1.0)
        good_idx = [self.eeg_raw.ch_names.index(ch) for ch in good_channels]
        full_data[good_idx] = eeg_raw_good.get_data()
        self.eeg_raw._data = full_data
        repaired_bad_channels = list(set(global_bad_channels + bad_channels))
        if len(repaired_bad_channels) != 0:
            self.eeg_raw.info['bads'] = repaired_bad_channels
            self.eeg_raw.interpolate_bads(method='spline', reset_bads=True)
            logging.info(f"完成坏通道的插值重建: {repaired_bad_channels}")
        self.raw = self.eeg_raw.copy()
        self.raw.add_channels([self.all_no_eeg_raw], force_update_info=True)
        return self.raw
