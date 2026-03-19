"""
ICA 和 FASTER 处理模块
"""
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne_faster import find_bad_epochs, find_bad_channels_in_epochs
import logging
import gc
import re

class ICA_EEG:
    def __init__(self, raw_data, stim_analysis, non_stim_mask=None, n_components=0.99, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.raw_data = raw_data
        self.fastica = ICA(n_components=self.n_components, method='fastica', random_state=self.random_state)
        if stim_analysis:
            self.clean_raw = self.apply_ica_stim(non_stim_mask)
        else:
            self.clean_raw = self.apply_ica()

    def apply_ica(self):
        raw_ica = self.raw_data.copy()
        self.fastica.fit(raw_ica)
        ica_labels = label_components(raw_ica, self.fastica, method="iclabel")
        exclude_idx = []
        threshold = 0.5
        artifact_types = ['eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise']
        for i, (label, proba) in enumerate(zip(ica_labels['labels'], ica_labels['y_pred_proba'])):
            if label in artifact_types and proba > threshold:
                exclude_idx.append(i)
        if exclude_idx:
            self.fastica.exclude = exclude_idx
            raw_clean = self.fastica.apply(raw_ica.copy())
        else:
            raw_clean = raw_ica
        del raw_ica, ica_labels
        gc.collect()
        return raw_clean

    def apply_ica_stim(self, non_stim_mask):
        raw = self.raw_data.copy()
        original_annots = raw.annotations.copy()
        non_stim_raw = mne.io.RawArray(raw.get_data()[:, non_stim_mask], raw.info.copy())
        self.fastica.fit(non_stim_raw)
        ic_labels = label_components(non_stim_raw, self.fastica, method="iclabel")
        labels = ic_labels['labels']
        probabilities = ic_labels['y_pred_proba']
        artefact_threshold = 0.7
        artefact_components = []
        for idx, (label, prob) in enumerate(zip(labels, probabilities)):
            if prob > artefact_threshold and label != 'brain':
                artefact_components.append(idx)
        print(f"标记的伪迹成分: {artefact_components}")
        denoised_non_stim = self.fastica.apply(non_stim_raw.copy(), exclude=artefact_components)
        denoised_data = raw.get_data().copy()
        denoised_data[:, non_stim_mask] = denoised_non_stim.get_data()
        denoised_raw = mne.io.RawArray(denoised_data, raw.info)
        denoised_raw.set_annotations(original_annots)
        del raw, non_stim_raw, denoised_non_stim
        gc.collect()
        return denoised_raw


class FASTER_EEG:
    def __init__(self, raw, stim_analysis=False, epoch_duration=1.0, max_bad_frac=0.2):
        self.epoch_duration = epoch_duration
        self.max_bad_frac = max_bad_frac
        if stim_analysis:
            print("FASTER处理中启用刺激分析模式")
            self.clean_raw, self.global_bad_channels = self.faster_remove_bad_epoch_stim(raw)
        else:
            print("FASTER处理中禁用刺激分析模式")
            self.clean_raw, self.global_bad_channels, self.processing_info = self.faster_remove_bad_epoch(raw)

    def faster_remove_bad_epoch(self, raw):
        raw_clean = raw.copy()
        sfreq = raw.info['sfreq']
        processing_info = {
            'parameters': {
                'bad_channel_threshold': self.max_bad_frac,
                'epoch_duration': self.epoch_duration,
                'sampling_frequency': sfreq
            },
            'bad_epochs_indices': [],
            'repaired_epochs_indices': [],
            'global_bad_channels': [],
            'epoch_details': {}
        }
        n_times_per_epoch = int(self.epoch_duration * sfreq)
        events = mne.make_fixed_length_events(raw_clean, duration=self.epoch_duration, overlap=0.0)
        epochs = mne.Epochs(raw_clean, events, tmin=0, tmax=self.epoch_duration - 1/sfreq,
                            baseline=None, preload=True, reject=None, flat=None)
        n_epochs, n_channels = len(epochs), len(epochs.ch_names)
        processing_info['parameters']['total_epochs'] = n_epochs
        processing_info['parameters']['total_channels'] = n_channels
        print(f"FASTER处理: {n_epochs}个epoch, {n_channels}个通道")
        from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs
        global_bad_channels_step1 = find_bad_channels(epochs, thres=3)
        processing_info['global_bad_channels_step1'] = global_bad_channels_step1
        bad_epochs_detected = find_bad_epochs(epochs, thres=5.0)
        processing_info['bad_epochs_detected'] = bad_epochs_detected
        bad_channels_per_epoch = find_bad_channels_in_epochs(epochs, thres=3)
        processing_info['bad_channels_per_epoch'] = bad_channels_per_epoch
        channel_bad_count = {ch: 0 for ch in epochs.ch_names}
        for epoch_bad_chs in bad_channels_per_epoch:
            for ch_name in epoch_bad_chs:
                channel_bad_count[ch_name] += 1
        epoch_based_global_bads = [ch for ch, count in channel_bad_count.items() if count / n_epochs > 0.3]
        global_bad_channels = list(set(global_bad_channels_step1 + epoch_based_global_bads))
        processing_info['global_bad_channels'] = global_bad_channels
        processing_info['epoch_based_global_bads'] = epoch_based_global_bads
        print(f"全局坏通道: {global_bad_channels}")
        print(f"检测到的坏epoch: {len(bad_epochs_detected)}个")

        all_epoch_data = epochs.get_data()
        for epoch_idx in range(n_epochs):
            epoch_info = {
                'bad_channels': bad_channels_per_epoch[epoch_idx],
                'is_bad_epoch_detected': epoch_idx in bad_epochs_detected,
                'action_taken': 'none',
                'repaired_channels': []
            }
            all_bad_channels = list(set(epoch_info['bad_channels'] + global_bad_channels))
            bad_ratio = len(all_bad_channels) / n_channels
            if bad_ratio > self.max_bad_frac or epoch_info['is_bad_epoch_detected']:
                epoch_info['action_taken'] = 'marked_bad'
                epoch_info['bad_ratio'] = bad_ratio
                processing_info['bad_epochs_indices'].append(epoch_idx)
            elif epoch_info['bad_channels']:
                epoch_info['action_taken'] = 'repaired'
                epoch_info['repaired_channels'] = epoch_info['bad_channels']
                epoch_info['bad_ratio'] = bad_ratio
                processing_info['repaired_epochs_indices'].append(epoch_idx)
                try:
                    epoch_data_repaired = self._repair_single_epoch(all_epoch_data[epoch_idx], epochs.info, epoch_info['bad_channels'])
                    all_epoch_data[epoch_idx] = epoch_data_repaired
                except Exception as e:
                    print(f"Error repairing epoch {epoch_idx}: {e}")
            processing_info['epoch_details'][f'epoch_{epoch_idx}'] = epoch_info

        clean_data = all_epoch_data.transpose(1, 0, 2).reshape(n_channels, -1)
        if clean_data.shape[1] != raw_clean.n_times:
            clean_data_adjusted = np.zeros((n_channels, raw_clean.n_times))
            min_length = min(clean_data.shape[1], raw_clean.n_times)
            clean_data_adjusted[:, :min_length] = clean_data[:, :min_length]
            clean_data = clean_data_adjusted
        clean_raw = mne.io.RawArray(clean_data, raw.info)
        if raw.annotations is not None:
            clean_raw.set_annotations(raw.annotations.copy())

        if processing_info['bad_epochs_indices']:
            bad_epoch_onsets = [epoch_idx * self.epoch_duration for epoch_idx in processing_info['bad_epochs_indices']]
            annotations = mne.Annotations(onset=bad_epoch_onsets,
                                          duration=[self.epoch_duration]*len(bad_epoch_onsets),
                                          description=['bad_epoch'] * len(bad_epoch_onsets))
            clean_raw.set_annotations(annotations)

        n_bad = len(processing_info['bad_epochs_indices'])
        n_repaired = len(processing_info['repaired_epochs_indices'])
        print(f"\nFASTER处理完成: 标记为bad的epoch: {n_bad}个, 已修复的epoch: {n_repaired}个, 全局坏通道: {global_bad_channels}")
        return clean_raw, global_bad_channels, processing_info

    def _repair_single_epoch(self, epoch_data, info, bad_channels):
        if not bad_channels:
            return epoch_data
        temp_epoch = mne.EpochsArray(data=epoch_data[np.newaxis, :, :], info=info.copy(),
                                      events=np.array([[0, 0, 1]]), tmin=0)
        temp_epoch.info['bads'] = bad_channels
        temp_epoch.interpolate_bads(reset_bads=True)
        repaired_data = temp_epoch.get_data()[0]
        return repaired_data

    def faster_remove_bad_epoch_stim(self, raw, stim_start_prefix='Start of stimulation', stim_end_prefix='End of stimulation'):
        original_annotations = raw.annotations.copy() if raw.annotations is not None else None
        epochs = mne.make_fixed_length_epochs(raw, duration=self.epoch_duration, overlap=0.0, preload=True, verbose=True)
        n_epochs, n_channels, n_times = epochs.get_data().shape
        print(f"创建 {n_epochs} 个 epoch, 每个 epoch {n_times} 个时间点")
        stim_intervals = self.identify_stim_intervals(raw, stim_start_prefix, stim_end_prefix)
        stim_mask = self.create_stim_mask(epochs, stim_intervals)
        non_stim_mask = ~stim_mask
        epoch_data = epochs.get_data()
        bad_channels_per_epoch = [[] for _ in range(n_epochs)]
        bad_epochs = np.zeros(n_epochs, dtype=bool)
        if np.any(non_stim_mask):
            non_stim_epochs = epochs[non_stim_mask]
            non_stim_bad_epochs = find_bad_epochs(non_stim_epochs, thres=3)
            non_stim_bad_channels = find_bad_channels_in_epochs(non_stim_epochs, thres=3)
            non_stim_indices = np.where(non_stim_mask)[0]
            for i, epoch_idx in enumerate(non_stim_indices):
                bad_epochs[epoch_idx] = i in non_stim_bad_epochs
                bad_channels_per_epoch[epoch_idx] = non_stim_bad_channels[i]
        channel_counts = {}
        for epoch_idx in np.where(non_stim_mask)[0]:
            for chan in bad_channels_per_epoch[epoch_idx]:
                channel_counts[chan] = channel_counts.get(chan, 0) + 1
        n_non_stim_epochs = np.sum(non_stim_mask)
        global_bad_channels = [chan for chan, count in channel_counts.items()
                               if n_non_stim_epochs > 0 and count / n_non_stim_epochs > 0.2]
        logging.info(f"全局坏通道 (基于非刺激期): {global_bad_channels}")
        logging.info(f"开始处理非刺激期数据，共 {n_non_stim_epochs} 个 epoch")
        cleaned_data = []
        for epoch_idx in range(n_epochs):
            if stim_mask[epoch_idx]:
                cleaned_data.append(epoch_data[epoch_idx])
                continue
            if bad_epochs[epoch_idx]:
                bads = bad_channels_per_epoch[epoch_idx]
                logging.info(f"process epoch {epoch_idx} 通道: {bads}")
                if len(bads) > 1:
                    epoch = epochs[epoch_idx]
                    epoch.info['bads'] = list(set(bads))
                    epoch.interpolate_bads()
                    cleaned_data.append(epoch.get_data()[0])
                else:
                    cleaned_data.append(epoch_data[epoch_idx])
            else:
                cleaned_data.append(epoch_data[epoch_idx])
        clean_data = np.concatenate(cleaned_data, axis=1)
        clean_raw = mne.io.RawArray(clean_data, epochs.info)
        if original_annotations is not None:
            clean_raw.set_annotations(original_annotations)
        del raw, epochs
        gc.collect()
        return clean_raw, global_bad_channels

    def identify_stim_intervals(self, raw, start_prefix='Stim_Start', end_prefix='Stim_End'):
        stim_intervals = []
        start_events = {}
        end_events = {}
        pattern = re.compile(r'\[(\d+),\s*(\d+)\]')
        for ann in raw.annotations:
            desc = str(ann['description'])
            if desc.startswith(start_prefix):
                match = pattern.search(desc)
                if match:
                    event_id = tuple(map(int, match.groups()))
                    start_events[event_id] = ann['onset']
            elif desc.startswith(end_prefix):
                match = pattern.search(desc)
                if match:
                    event_id = tuple(map(int, match.groups()))
                    end_events[event_id] = ann['onset'] + ann['duration']
        for event_id in set(start_events.keys()) | set(end_events.keys()):
            start = start_events.get(event_id)
            end = end_events.get(event_id)
            if start and end and start < end:
                stim_intervals.append((start, end))
        return self.merge_intervals(stim_intervals)

    def merge_intervals(self, intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        expanded_stim = []
        for start, end in merged:
            expanded_start = max(0, start - 2)
            expanded_end = end + 2
            expanded_stim.append((expanded_start, expanded_end))
        return expanded_stim

    def create_stim_mask(self, epochs, stim_intervals):
        if not stim_intervals:
            return np.zeros(len(epochs), dtype=bool)
        sfreq = epochs.info['sfreq']
        n_epochs = len(epochs)
        event_samples = epochs.events[:, 0]
        epoch_starts = event_samples / sfreq
        epoch_ends = epoch_starts + (epochs.tmax - epochs.tmin)
        stim_starts = np.array([s for s, _ in stim_intervals])
        stim_ends = np.array([e for _, e in stim_intervals])
        overlaps = np.zeros(n_epochs, dtype=bool)
        for stim_start, stim_end in zip(stim_starts, stim_ends):
            start_overlap = np.logical_and(epoch_starts < stim_end, epoch_ends > stim_start)
            overlaps = np.logical_or(overlaps, start_overlap)
        return overlaps
