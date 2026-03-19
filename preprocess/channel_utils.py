"""
通道类型分类、双极导联处理等工具
"""
import re
import numpy as np
import mne
from mne.channels import make_standard_montage, make_dig_montage

# 标准10-20系统电极列表
standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", 
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

class ChannelTypeClassifier:
    """通道类型分类器"""
    def __init__(self, standard_1020_list, similarity_threshold=75):
        self.standard_1020 = standard_1020_list
        self.similarity_threshold = similarity_threshold

    def split_channel_name(self, channel_name):
        """使用非字母数字字符拆分通道名称"""
        parts = re.split(r'[^a-zA-Z0-9]+', channel_name)
        parts = [part.upper() for part in parts if part]
        return parts

    def is_eeg_channel(self, channel_name):
        parts = self.split_channel_name(channel_name)
        if any('EEG' in part.upper() for part in parts):
            return True
        for part in parts:
            if part in self.standard_1020:
                return True
        combined = ''.join(parts).upper()
        ref_patterns = ['A1', 'A2', 'REF', 'GND', 'CZ']
        for ref in ref_patterns:
            if combined.endswith(ref):
                electrode = combined[:-len(ref)]
                if electrode in self.standard_1020:
                    return True
        if combined in self.standard_1020:
            return True
        for electrode in self.standard_1020:
            if len(electrode) >= 2:
                if combined == electrode:
                    return True
                if combined.startswith(electrode):
                    remaining = combined[len(electrode):]
                    if remaining in ref_patterns:
                        return True
                if combined.endswith(electrode):
                    if len(combined) > len(electrode):
                        preceding_char = combined[len(combined) - len(electrode) - 1]
                        if preceding_char.isalpha():
                            continue
                    return True
        return False

    def is_eog_channel(self, channel_name):
        parts = self.split_channel_name(channel_name)
        if parts and parts[0] == 'EOG':
            return True
        if any('EOG' in part for part in parts):
            return True
        eog_keywords = ['EYE', 'OCULAR', 'OCULOGRAM', 'HEOG', 'VEOG']
        if any(keyword in part for part in parts for keyword in eog_keywords):
            return True
        return False

    def is_ecg_channel(self, channel_name):
        parts = self.split_channel_name(channel_name)
        if parts and parts[0] == 'ECG':
            return True
        if any('ECG' in part for part in parts):
            return True
        if any('EKG' in part for part in parts):
            return True
        ecg_keywords = ['CARDIO', 'ELECTROCARDIOGRAM', 'HEART']
        if any(keyword in part for part in parts for keyword in ecg_keywords):
            return True
        return False

    def is_emg_channel(self, channel_name):
        parts = self.split_channel_name(channel_name)
        if parts and parts[0] == 'EMG':
            return True
        if any('EMG' in part for part in parts):
            return True
        emg_keywords = ['MUSCULAR', 'ELECTROMYOGRAM', 'MUSCLE']
        if any(keyword in part for part in parts for keyword in emg_keywords):
            return True
        return False

    def classify_channel(self, channel_name):
        """分类通道类型"""
        parts = self.split_channel_name(channel_name)
        if parts:
            first_part = parts[0]
            if first_part == 'ECG':
                return 'ecg'
            elif first_part == 'EMG':
                return 'emg'
            elif first_part == 'EOG':
                return 'eog'
            elif first_part == 'EEG':
                return 'eeg'
        if self.is_ecg_channel(channel_name):
            return "ecg"
        if self.is_emg_channel(channel_name):
            return "emg"
        if self.is_eog_channel(channel_name):
            return "eog"
        if self.is_eeg_channel(channel_name):
            return "eeg"
        return "misc"


def handle_bipolar_channels(raw):
    """
    处理双极导联通道，为它们分配位置
    """
    ch_types = raw.get_channel_types()
    montage = make_standard_montage("standard_1020")
    std_ch_pos = montage.get_positions()['ch_pos']
    custom_ch_pos = {}

    for ch in raw.ch_names:
        if ch in std_ch_pos:
            custom_ch_pos[ch] = std_ch_pos[ch]

    bipolar_chs = []
    for i, ch_name in enumerate(raw.ch_names):
        if ch_types[i] == 'eeg':
            if re.search(r'[-_]', ch_name):
                bipolar_chs.append(ch_name)

    if bipolar_chs:
        print(f"检测到双导联通道: {bipolar_chs}")
        for ch in bipolar_chs:
            parts = re.split(r'[-_]', ch)
            if len(parts) >= 2:
                first_ch = parts[0]
                second_ch = parts[1] if len(parts) > 1 else None
                if first_ch in std_ch_pos and second_ch in std_ch_pos:
                    pos1 = np.array(std_ch_pos[first_ch])
                    pos2 = np.array(std_ch_pos[second_ch])
                    midpoint = (pos1 + pos2) / 2
                    custom_ch_pos[ch] = midpoint
                    print(f"为双导联 {ch} 使用 {first_ch} 和 {second_ch} 的中点位置")
                elif first_ch in std_ch_pos:
                    custom_ch_pos[ch] = std_ch_pos[first_ch]
                    print(f"为双导联 {ch} 使用 {first_ch} 的位置")
                elif second_ch in std_ch_pos:
                    custom_ch_pos[ch] = std_ch_pos[second_ch]
                    print(f"为双导联 {ch} 使用 {second_ch} 的位置")
                else:
                    if any(x in ch.upper() for x in ['FP', 'AF', 'F']):
                        center_pos = np.array([0, 0.08, 0.05])
                    elif any(x in ch.upper() for x in ['C', 'CP']):
                        center_pos = np.array([0, 0, 0.1])
                    elif any(x in ch.upper() for x in ['P', 'PO', 'O']):
                        center_pos = np.array([0, -0.08, 0.05])
                    elif any(x in ch.upper() for x in ['T']):
                        center_pos = np.array([-0.07, 0, 0.05])
                    else:
                        center_pos = np.array([0, 0, 0.1])
                    custom_ch_pos[ch] = center_pos
                    print(f"为双导联 {ch} 使用推断的默认位置")
            else:
                center_pos = np.array([0, 0, 0.1])
                custom_ch_pos[ch] = center_pos
                print(f"为双导联 {ch} 使用默认中心位置")

    if custom_ch_pos:
        new_montage = make_dig_montage(ch_pos=custom_ch_pos, coord_frame='head')
        raw.set_montage(new_montage, on_missing="warn")
    return raw


def channel_type_detection(raw, similarity_threshold=75):
    """检测并设置通道类型"""
    classifier = ChannelTypeClassifier(standard_1020, similarity_threshold)
    channel_types = {}
    for ch_name in raw.info['ch_names']:
        channel_type = classifier.classify_channel(ch_name)
        channel_types[ch_name] = channel_type
    raw.set_channel_types(channel_types)
    print("通道类型分类结果:")
    for ch_name, ch_type in channel_types.items():
        print(f"  {ch_name}: {ch_type}")
    return raw
