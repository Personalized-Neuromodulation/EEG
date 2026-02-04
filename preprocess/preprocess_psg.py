
"""
预处理EEG信号：(兼容处理刺激区间的数据)
1.读取bdf/edf文件 (如有需要则合并所有bdf为一个文件)
2.滤波、重参考、去除ECG伪迹、去趋势
3.重采样(可选)
4.ICA(可选)
5.FASTER修复或者标记坏epoch

"""


import mne
from datetime import datetime, timedelta
import numpy as np
import mne,os,re
from scipy.signal import detrend
import concurrent.futures
import logging,time
from mne.preprocessing import ICA
import gc
import numpy as np
from scipy.signal import welch
from scipy import signal
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs
from scipy.signal import welch, butter, filtfilt,iirfilter
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import datetime
from scipy.signal import spectrogram
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import neurokit2 as nk
from mne_icalabel import label_components 
import scipy.stats as stats
from mne.channels import make_standard_montage, make_dig_montage
from sklearn.linear_model import RANSACRegressor
import glob
from natsort import natsorted
import random
from BIDS import BIDSProcessor
random.seed(42)
np.random.seed(42)


# def setup_logging(base_dir):
#     """设置日志记录器"""
#     # 创建日志目录
#     log_dir = os.path.join(base_dir, "analysis_logs")
#     os.makedirs(log_dir, exist_ok=True)
    
#     # 创建带时间戳的日志文件名
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     log_filename = os.path.join(log_dir, f"processing_{timestamp}.log")
    
#     # 配置日志记录器
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_filename),
#             logging.StreamHandler()
#         ]
#     )
#     return log_dir
# # base_dir = r'D:\analysis\long_stim_bdf\test_preprocess'
# # log_dir = setup_logging(base_dir)
# # print(f"日志文件保存在: {log_dir}")

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

def handle_bipolar_channels(raw):
    """
    处理双极导联通道的完整方案
    
    参数:
        raw: mne.io.Raw 对象
    
    返回:
        更新后的 Raw 对象
    """
    # 获取所有通道类型
    ch_types = raw.get_channel_types()
    # 获取标准蒙太奇
    montage = make_standard_montage("standard_1020")
    std_ch_pos = montage.get_positions()['ch_pos']
    
    # 创建自定义位置字典
    custom_ch_pos = {}
    
    # 添加标准通道位置
    for ch in raw.ch_names:
        if ch in std_ch_pos:
            custom_ch_pos[ch] = std_ch_pos[ch]
    
    # 处理双极导联位置
    # 找出双极导联通道（使用正则表达式匹配多种分隔符）
    bipolar_chs = []
    for i, ch_name in enumerate(raw.ch_names):
        if ch_types[i] == 'eeg':
            # 使用正则表达式匹配连字符或下划线分隔的双极导联
            if re.search(r'[-_]', ch_name):
                bipolar_chs.append(ch_name)
    
    # 如果没有双极导联，直接返回
    if bipolar_chs:   
        print(f"检测到双导联通道: {bipolar_chs}")
        
        for ch in bipolar_chs:
            # 使用正则表达式分割多种分隔符
            parts = re.split(r'[-_]', ch)
            
            if len(parts) >= 2:
                first_ch = parts[0]
                second_ch = parts[1] if len(parts) > 1 else None
                
                # 策略1: 如果两个电极都有位置，使用中点位置
                if first_ch in std_ch_pos and second_ch in std_ch_pos:
                    pos1 = np.array(std_ch_pos[first_ch])
                    pos2 = np.array(std_ch_pos[second_ch])
                    midpoint = (pos1 + pos2) / 2
                    custom_ch_pos[ch] = midpoint
                    print(f"为双导联 {ch} 使用 {first_ch} 和 {second_ch} 的中点位置")
                
                # 策略2: 只有第一个电极有位置
                elif first_ch in std_ch_pos:
                    custom_ch_pos[ch] = std_ch_pos[first_ch]
                    print(f"为双导联 {ch} 使用 {first_ch} 的位置")
                
                # 策略3: 只有第二个电极有位置
                elif second_ch in std_ch_pos:
                    custom_ch_pos[ch] = std_ch_pos[second_ch]
                    print(f"为双导联 {ch} 使用 {second_ch} 的位置")
                
                # 策略4: 都没有位置，使用默认位置
                else:
                    # 根据通道名称判断大致位置
                    if any(x in ch.upper() for x in ['FP', 'AF', 'F']):
                        center_pos = np.array([0, 0.08, 0.05])  # 前部
                    elif any(x in ch.upper() for x in ['C', 'CP']):
                        center_pos = np.array([0, 0, 0.1])  # 中央
                    elif any(x in ch.upper() for x in ['P', 'PO', 'O']):
                        center_pos = np.array([0, -0.08, 0.05])  # 后部
                    elif any(x in ch.upper() for x in ['T']):
                        center_pos = np.array([-0.07, 0, 0.05])  # 颞部
                    else:
                        center_pos = np.array([0, 0, 0.1])  # 默认
                    
                    custom_ch_pos[ch] = center_pos
                    print(f"为双导联 {ch} 使用推断的默认位置")
            else:
                # 使用默认中心位置
                center_pos = np.array([0, 0, 0.1])
                custom_ch_pos[ch] = center_pos
                print(f"为双导联 {ch} 使用默认中心位置")
    
    # 使用 make_dig_montage 创建蒙太奇
    if custom_ch_pos:  # 确保有位置信息
        new_montage = make_dig_montage(
            ch_pos=custom_ch_pos,
            coord_frame='head'  # 使用头部坐标系
        )
        
        # 应用新蒙太奇
        raw.set_montage(new_montage, on_missing="warn")
    
    return raw

def channel_type_detection(raw, similarity_threshold=75):
    """优化后的通道类型检测函数"""
    classifier = ChannelTypeClassifier(standard_1020, similarity_threshold)
    channel_types = {}
    
    for ch_name in raw.info['ch_names']:
        channel_type = classifier.classify_channel(ch_name)
        channel_types[ch_name] = channel_type
    
    # 应用通道类型
    raw.set_channel_types(channel_types)
    
    # 打印分类结果
    print("通道类型分类结果:")
    for ch_name, ch_type in channel_types.items():
        print(f"  {ch_name}: {ch_type}")
    
    return raw

def detect_stim_intervals(raw, start_prefix='Start of stimulation', end_prefix='End of stimulation'):
    # 1. 识别刺激片段
    stim_intervals = []
    start_events = {}
    end_events = {}

    for ann in raw.annotations:
        desc = str(ann['description'])
        
        # 匹配带编号的开始标记
        if desc.startswith(start_prefix):
            match = re.search(r'\[(\d+),\s*(\d+)\]', desc)
            if match:
                event_id = tuple(map(int, match.groups()))
                start_events[event_id] = ann['onset']
        
        # 匹配带编号的结束标记
        elif desc.startswith(end_prefix):
            match = re.search(r'\[(\d+),\s*(\d+)\]', desc)
            if match:
                event_id = tuple(map(int, match.groups()))
                end_events[event_id] = ann['onset'] + ann['duration']
    
    # 创建刺激间隔
    for e_id in sorted(set(start_events.keys()) | set(end_events.keys())):
        start = start_events.get(e_id)
        end = end_events.get(e_id)
        
        if start and end and start < end:
            # if end - start < 2 *60:
            #     stim_intervals.append((start-60, end+10*60)) #sham
            # else:
            stim_intervals.append((start-2, end+2))  ### ###########长时程刺激去掉60秒，短时程刺激去掉2秒
    
    # 2. 创建非刺激掩码
    n_samples = len(raw.times)
    sfreq = raw.info['sfreq']
    non_stim_mask = np.ones(n_samples, dtype=bool)#15844000
    non_stim_mask_time = np.ones(int(raw.times.shape[-1]/sfreq), dtype=bool)
    
    for start, end in stim_intervals:
        start_idx = int(start * sfreq)
        end_idx = int(end * sfreq)
        non_stim_mask[start_idx:min(end_idx, n_samples)] = False
        non_stim_mask_time[int(start):min(int(end), int(raw.times.shape[-1]/sfreq))] = False
    
    # 3. 提取非刺激数据
    non_stim_data = raw.get_data()[:, non_stim_mask] #(22, 10968959)
    non_stim_raw = mne.io.RawArray(non_stim_data, raw.info.copy())

    return non_stim_raw, non_stim_mask,non_stim_mask_time

def filter_detrend(raw):
        """滤波、平均参考、基线校正"""
        raw.load_data()
        
        # 识别通道类型
        eeg_picks = mne.pick_types(raw.info, eeg=True,emg=False, ecg=False, eog=False, misc=False)
        emg_picks = mne.pick_types(raw.info, emg=True, ecg=False, eog=False, misc=False)
        ecg_picks = mne.pick_types(raw.info, emg=False, ecg=True, eog=False, misc=False)
        eog_picks = mne.pick_types(raw.info, emg=False, ecg=False, eog=True, misc=False)
        # 对EEG通道应用特定滤波
        # 对EEG通道应用特定滤波
        if len(eeg_picks) > 0:
            # 0.5-40Hz带通滤波
            raw.filter(
                l_freq=0.5, 
                h_freq=40, 
                picks=eeg_picks, 
                method='fir', 
                fir_design='firwin'
            )
            
            # 50Hz陷波滤波
            if raw.info['sfreq'] > 100:
                raw.notch_filter(
                    freqs=50, #people 
                    picks=eeg_picks, 
                    method='fir', 
                    fir_design='firwin'
                )
        
        # 对其他通道应用特定滤波
        if len(eog_picks) > 0:
            # 0.5-40Hz带通滤波
            raw.filter(
                l_freq=0.5, 
                h_freq=15, 
                picks=eog_picks, 
                method='fir', 
                fir_design='firwin'
            )

            # 50Hz陷波滤波
            if raw.info['sfreq'] > 100:
                raw.notch_filter(
                    freqs=50, 
                    picks=eog_picks,
                    method='fir',
                    fir_design='firwin'
                )

        if len(emg_picks) > 0:
            # 15-100Hz带通滤波
            raw.filter(
                l_freq=15, 
                h_freq=min(100, raw.info['sfreq']/2-1), 
                picks=emg_picks, 
                method='fir', 
                fir_design='firwin'
            )
            
            # 50Hz和100Hz陷波滤波
            if 200 > raw.info['sfreq'] > 100:
                raw.notch_filter(
                    freqs=50, 
                    picks=emg_picks, 
                    method='fir', 
                    fir_design='firwin'
                )
            if raw.info['sfreq'] > 200:
                raw.notch_filter(
                    freqs=[50,100], 
                    picks=emg_picks, 
                    method='fir', 
                    fir_design='firwin'
                )

        if len(ecg_picks) > 0:
            #心电带通滤波
            raw.filter(
                l_freq=0.5, 
                h_freq=35, 
                picks=ecg_picks, 
                method='fir', 
                fir_design='firwin'
            )
        
        
        raw = detrend_data_multithreaded(raw)
        return raw

def detrend_data_multithreaded(raw):
        """
        使用多线程并行处理所有通道的去趋势，并保留Annotations信息
        参数: 原始Raw对象，包含数据、Info和Annotations
        返回:处理后的新Raw对象，包含原始Annotations
        """
        # 在数据修改前保存Annotations
        original_annotations = raw.annotations.copy()
        
        full_data = raw.get_data()
        n_channels = full_data.shape[0]
        
        # 定义单通道去趋势函数
        def detrend_channel(args):
            channel_idx, channel_data = args
            return channel_idx, detrend(channel_data, type='linear')
        # 准备线程池
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 准备任务参数 (通道索引, 通道数据)
            tasks = [(i, full_data[i, :].copy()) for i in range(n_channels)]
            # 并行执行去趋势处理
            results = list(executor.map(detrend_channel, tasks))
     
        for channel_idx, channel_data in results:
            full_data[channel_idx, :] = channel_data
        # 创建新的Raw对象
        new_raw = mne.io.RawArray(full_data, raw.info)
        # 恢复Annotations信息 (包括onset, duration, description等)
        new_raw.set_annotations(original_annotations)
        print("**************滤波、平均重参考、去趋势完成**************")
        return new_raw

class ChannelTypeClassifier:
    def __init__(self, standard_1020_list, similarity_threshold=75):
        self.standard_1020 = standard_1020_list
        self.similarity_threshold = similarity_threshold
        
        # 已知的特定数据集映射
        # self.known_datasets = {
        #     'sleep_edf': {
        #         'EEG Fpz-Cz': 'eeg',
        #         'EEG Pz-Oz': 'eeg',
        #         'EOG horizontal': 'eog',
        #         'Resp oro-nasal': 'misc',
        #         'EMG submental': 'emg',
        #         'Temp rectal': 'misc',
        #         'Event marker': 'misc'
        #     }
        # }
    
    def split_channel_name(self, channel_name):
        """使用非字母数字字符拆分通道名称"""
        # 使用正则表达式按非字母数字字符拆分
        parts = re.split(r'[^a-zA-Z0-9]+', channel_name)
        # 过滤空字符串并转换为大写
        parts = [part.upper() for part in parts if part]
        return parts

    def is_eeg_channel(self, channel_name):
        """判断是否为EEG通道"""
        parts = self.split_channel_name(channel_name)
        
        # 1. 检查是否包含EEG关键词
        if any('EEG' in part.upper() for part in parts):
            return True
        
        # 2. 检查每个部分是否在10-20系统电极列表中
        for part in parts:
            if part in self.standard_1020:
                return True
        
        # 3. 处理没有连字符的情况
        # 将部分连接起来处理（如F3A2, C4A1等）
        combined = ''.join(parts).upper()
        
        # 常见参考电极模式
        ref_patterns = ['A1', 'A2', 'REF', 'GND', 'CZ']
        
        # 3.1 尝试从末尾匹配参考电极
        for ref in ref_patterns:
            if combined.endswith(ref):
                electrode = combined[:-len(ref)]
                if electrode in self.standard_1020:
                    return True
        
        # 3.2 检查是否直接匹配标准电极
        if combined in self.standard_1020:
            return True
        
        # 3.3 检查是否由电极+参考电极组成
        # 这里的关键是：电极名称通常以字母开头，后跟数字
        # 而SpO2、EOG、ECG等通道不遵循这个模式
        for electrode in self.standard_1020:
            # 避免将"O2"错误匹配到"SpO2"
            # 检查combined是否以电极开头或以电极+参考电极结尾
            if len(electrode) >= 2:  # 电极至少2个字符
                # 检查combined是否恰好等于电极
                if combined == electrode:
                    return True
                
                # 检查combined是否以电极开头，后面跟着参考电极
                if combined.startswith(electrode):
                    remaining = combined[len(electrode):]
                    if remaining in ref_patterns:
                        return True
                
                # 检查combined是否以电极结尾，前面可能有其他内容
                if combined.endswith(electrode):
                    # 避免将"SpO2"中的"O2"误判
                    # 检查电极前面的字符
                    if len(combined) > len(electrode):
                        preceding_char = combined[len(combined) - len(electrode) - 1]
                        # 如果前面的字符是字母，可能不是独立电极
                        if preceding_char.isalpha():
                            # 检查前面是否可能是电极的一部分
                            # 例如："SpO2"中的'O2'，前面是'p'，这是字母
                            # 而"F3A2"中的'A2'，前面是'3'，这是数字
                            continue
                    return True
        
        return False
    
    def is_eog_channel(self, channel_name):
        """判断是否为EOG通道"""
        parts = self.split_channel_name(channel_name)
        
        # 检查是否以EOG开头
        if parts and parts[0] == 'EOG':
            return True
            
        # 检查是否包含EOG关键词
        if any('EOG' in part for part in parts):
            return True
            
        # 检查其他EOG相关关键词
        eog_keywords = ['EYE', 'OCULAR', 'OCULOGRAM', 'HEOG', 'VEOG']
        if any(keyword in part for part in parts for keyword in eog_keywords):
            return True
            
        return False
    
    def is_ecg_channel(self, channel_name):
        """判断是否为ECG通道"""
        parts = self.split_channel_name(channel_name)
        
        # 检查是否以ECG开头
        if parts and parts[0] == 'ECG':
            return True
            
        # 检查是否包含ECG关键词
        if any('ECG' in part for part in parts):
            return True
            
        # 检查EKG变体
        if any('EKG' in part for part in parts):
            return True
            
        # 检查其他ECG相关关键词
        ecg_keywords = ['CARDIO', 'ELECTROCARDIOGRAM', 'HEART']
        if any(keyword in part for part in parts for keyword in ecg_keywords):
            return True
            
        return False
    
    def is_emg_channel(self, channel_name):
        """判断是否为EMG通道"""
        parts = self.split_channel_name(channel_name)
        
        # 检查是否以EMG开头
        if parts and parts[0] == 'EMG':
            return True
            
        # 检查是否包含EMG关键词
        if any('EMG' in part for part in parts):
            return True
            
        # 检查其他EMG相关关键词
        emg_keywords = ['MUSCULAR', 'ELECTROMYOGRAM', 'MUSCLE']
        if any(keyword in part for part in parts for keyword in emg_keywords):
            return True
            
        return False
    
    def check_known_datasets(self, channel_name):
        """检查是否是已知数据集的特定通道"""
        for dataset, mapping in self.known_datasets.items():
            if channel_name in mapping:
                return mapping[channel_name]
        return None
    
    def classify_channel(self, channel_name):
        """分类通道类型"""
        # 0. 首先检查是否是已知数据集的特定通道
        # known_type = self.check_known_datasets(channel_name)
        # if known_type:
        #     return known_type
        
        # 1. 检查是否以ECG、EMG、EOG开头
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
        
        # 2. 检查ECG模式（优先级高于EEG）
        if self.is_ecg_channel(channel_name):
            return "ecg"
        
        # 3. 检查EMG模式
        if self.is_emg_channel(channel_name):
            return "emg"
        
        # 4. 检查EOG模式
        if self.is_eog_channel(channel_name):
            return "eog"
        
        # 5. 检查EEG模式
        if self.is_eeg_channel(channel_name):
            return "eeg"
        
        # 6. 默认标记为其他
        return "misc"

class MultiChannelECGArtifactRemover:
    def __init__(self, fs):
        """
        多通道EEG信号ECG伪迹去除类
        
        参数:
            fs: 采样率
            segment_length: 心搏分析窗长 (秒)
            n_templates: 使用的主模板数量
        """
        self.fs = fs
        self.pre_samples = int(0.5 * fs)  # QRS波前200ms
        self.post_samples = int(0.5 * fs)  # QRS波后600ms
        self.channel_templates = {}  # 存储每个通道的模板
        self.r_peaks = None  # 存储检测到的R波位置
    
    def remove_ecg_from_raw(self, raw, ecg_data):
        """
        从多通道EEG Raw对象中去除ECG伪迹
        
        参数:
            raw: MNE Raw对象，包含EEG通道和ECG通道
            ecg_data: ECG
            
        返回:
            clean_raw: 去除ECG伪迹后的Raw对象
        """
        
        
        # 检测R波位置
        self.r_peaks = self._detect_r_peaks(ecg_data)
        
        # 获取所有EEG通道名称
        channel_types = raw.get_channel_types()
        # 找出所有非EEG通道
        all_ch_names = [ch for i, ch in enumerate(raw.ch_names) 
                        if channel_types[i] != 'ecg']
        # all_ch_names = [ch for ch in raw.ch_names if ch != 'ECG']
        
        # 创建干净的Raw对象副本
        clean_raw = raw.copy()
        
        # 处理每个EEG通道
        for ch_name in all_ch_names:
          
            # 获取通道数据
            ch_data = raw.get_data(picks=[ch_name]).flatten()
            
            # 去除该通道的ECG伪迹
            # r_peaks = detect_r_peaks(ecg_filtered, fs)
            template, pre, post = self.create_ecg_template(ch_data, self.r_peaks)
            cleaned_data = self.subtract_template(ch_data, self.r_peaks, template, pre, post)
        
            # 将处理后的数据放回Raw对象
            clean_raw._data[raw.ch_names.index(ch_name)] = cleaned_data
        
        return clean_raw
    
    def subtract_template(self,signal, r_peaks, template, pre, post):
        # 复制信号
        corrected_signal = signal.copy()

        # 遍历每个R峰
        for r in r_peaks:
            if r - pre < 0 or r + post > len(signal):
                continue
            corrected_signal[r - pre:r + post] -= template

        return corrected_signal
    
    def create_ecg_template(self, signal, r_peaks):
        pre = self.pre_samples  # time before R-peak
        post = self.post_samples # time after R-peak
        segments = []

        for r in r_peaks:
            if r - pre < 0 or r + post > len(signal):
                continue
            segment = signal[r - pre:r + post]
            segments.append(segment)

        segments = np.array(segments)

        # Robust mean: use median to reduce noise influence
        template = np.median(segments, axis=0)

        return template, pre, post

    def _detect_r_peaks(self, ecg_reference):
        # Use biosppy ECG module to get R-peaks
        # out = signals.ecg.ecg(signal=ecg_reference, sampling_rate=self.fs, show=False)
        # r_peaks = out['rpeaks']
        """使用 NeuroKit2 检测 R 波位置"""
        # 轻度滤波 ECG 信号
        ecg_cleaned = nk.ecg_clean(ecg_reference, sampling_rate=self.fs, method="neurokit")
        
        # 检测 R 波峰值
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.fs)
        
        return rpeaks["ECG_R_Peaks"]
        # return r_peaks
    
class CleanRawData:
    """
    class CleanRawData: EEGlabel中检测和标记bad_channels方法
    """
    def __init__(self):
        """
        初始化Clean RawData通道拒绝参数
        """
        # 平坦通道检测参数
        self.flat_duration = 5.0  # 平坦信号持续时间阈值（秒）
        self.flat_threshold = 1e-6  # 信号幅度变化阈值
        
        # 高频噪声检测参数
        self.noise_threshold = 4  # 最大可接受高频噪声标准差阈值
        self.hp_cutoff = 20.0  # 高通滤波器截止频率（Hz）
        self.lp_cutoff = 30.0  # 低通滤波器截止频率（Hz）
        
        # 相关性检测参数
        self.corr_threshold = 0.8  # 最小可接受相关性阈值
        self.n_neighbors = 3  # 邻近通道数量

    def detect_bad_channels(self, raw):
        """
        检测所有类型的坏通道
        """

        # 检测平坦通道
        flat_bad = self.detect_flat_channels(raw)
        print(f"检测到平坦通道: {flat_bad}")
        
        # 检测高频噪声通道
        hf_noise_bad = self.detect_hf_noise_channels(raw)
        print(f"检测到高频噪声通道: {hf_noise_bad}")
        
        # 检测低相关性通道(含有刺激时相关性非常低)
        low_corr_bad = self.detect_low_correlation_channels(raw)
        print(f"检测到低相关性通道: {low_corr_bad}")
      
        all_bad = list(set(flat_bad + hf_noise_bad + low_corr_bad))
        print(f"总共检测到坏通道: {len(all_bad)}个")
        
        return all_bad

    def detect_flat_channels(self, raw, win_size=1.0):
        """
        检测平坦通道 - 检测持续平坦信号超过5秒的通道
        """

        # 获取数据和参数
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        n_channels, n_times = data.shape
        
        # 计算窗口样本数
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
                
                # 计算窗口内信号范围
                a = np.max(segment)
                b = np.min(segment)
                seg_range = np.max(segment) - np.min(segment)
                
                # 检查是否平坦
                if seg_range < self.flat_threshold:
                    flat_count += 1
                    # 记录最大连续平坦段
                    if flat_count > max_flat_count:
                        max_flat_count = flat_count
                else:
                    flat_count = 0
            
            # 计算持续平坦时间（秒）
            max_flat_duration = max_flat_count * win_size
            
            # 如果超过5秒平坦信号
            if max_flat_duration >= self.flat_duration:
                bad_channels.append(raw.ch_names[i])
        
        return bad_channels

    def detect_hf_noise_channels(self, raw):
        """
        检测高频噪声通道 - 使用noisiness指标
        
        参数:
            raw: mne.io.Raw 对象
            
        返回:
            高频噪声通道名称列表
        """
        # 获取数据和参数
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        n_channels = data.shape[0]
        
        # 为所有通道计算noisiness
        noisiness = np.zeros(n_channels)
        
        for i in range(n_channels):
            # 获取通道数据
            ch_data = data[i, :]
            
            # 设计高通滤波器（20Hz）
            hp_b, hp_a = iirfilter(
                N=4, 
                Wn=self.hp_cutoff, 
                btype='highpass', 
                fs=sfreq,
                ftype='butter'
            )
            
            # 设计低通滤波器（40Hz）
            lp_b, lp_a = iirfilter(
                N=4, 
                Wn=self.lp_cutoff, 
                btype='lowpass', 
                fs=sfreq,
                ftype='butter'
            )
            
            # 应用高通滤波器（提取高频分量）
            hp_data = filtfilt(hp_b, hp_a, ch_data)
            
            # 应用低通滤波器（提取神经信号）
            lp_data = filtfilt(lp_b, lp_a, ch_data)
            
            # 计算MAD（Median Absolute Deviation）
            mad_hp = stats.median_abs_deviation(hp_data, scale='normal')
            mad_lp = stats.median_abs_deviation(lp_data, scale='normal')
            
            # 计算noisiness
            if mad_lp > 0:
                noisiness[i] = mad_hp / mad_lp
            else:
                noisiness[i] = np.inf  # 分母为零时设为极大值
        
        # 计算znoise标准化值
        median_noise = np.median(noisiness)
        mad_noise = stats.median_abs_deviation(noisiness, scale='normal')
        znoise = (noisiness - median_noise) / mad_noise
        
        # 检测高频噪声通道
        bad_channels = []
        for i in range(n_channels):
            if znoise[i] > self.noise_threshold:
                bad_channels.append(raw.ch_names[i])
        
        return bad_channels

    def detect_low_correlation_channels(self, raw):
        """
        检测低相关性通道 - 使用RANSAC方法与邻近通道比较
        
        参数:
            raw: mne.io.Raw 对象
            
        返回:
            低相关性通道名称列表
        """
        # 1. 只选择EEG通道进行分析
        eeg_raw = raw.copy().pick_types(eeg=True)
        if len(eeg_raw.ch_names) == 0:
            print("警告: 没有EEG通道可用")
            return []
        
        # 获取数据和电极位置
        data = eeg_raw.get_data()
        ch_names = eeg_raw.ch_names
        
        # 2. 检查电极位置是否可用
        ch_pos = []
        valid_positions = []
        invalid_channels = []
        
        for i, ch in enumerate(eeg_raw.info['chs']):
            loc = ch['loc'][:3]  # 位置信息 (x, y, z)
            # 检查位置是否有效
            if np.isnan(loc).any() or np.linalg.norm(loc) < 1e-6:
                invalid_channels.append((i, ch_names[i]))
            else:
                valid_positions.append(i)
                ch_pos.append(loc)
        
        ch_pos = np.array(ch_pos)
        # print(f"发现 {len(invalid_channels)} 个EEG通道位置无效")
        
        # 3. 尝试修复无效位置
        if invalid_channels:
            try:
                # 获取标准10-20系统模板
                montage = make_standard_montage('standard_1020')
                pos_dict = montage.get_positions()['ch_pos']
                
                fixed_count = 0
                for idx, ch_name in invalid_channels:
                    if ch_name in pos_dict:
                        eeg_raw.info['chs'][idx]['loc'][:3] = pos_dict[ch_name]
                        print(f"使用标准模板修复通道 {ch_name} 的位置")
                        fixed_count += 1
                        # 添加到有效位置
                        valid_positions.append(idx)
                        ch_pos = np.vstack([ch_pos, pos_dict[ch_name]])
                
                print(f"成功修复 {fixed_count} 个EEG通道位置")
            except Exception as e:
                print(f"使用标准模板失败: {str(e)}")
        
        # 4. 检查是否有足够有效通道
        n_valid = len(valid_positions)
        if n_valid < 2:
            print("警告: 有效通道不足，无法进行相关性分析")
            return []
        
        # 5. 计算邻近通道
        nbrs = NearestNeighbors(n_neighbors=min(self.n_neighbors+1, n_valid))
        nbrs.fit(ch_pos)
        _, indices = nbrs.kneighbors(ch_pos)
        
        # 6. 为每个通道计算相关性
        correlations = np.zeros(len(eeg_raw.ch_names))
        
        for i, orig_idx in enumerate(valid_positions):
            # 获取当前通道数据
            ch_data = data[orig_idx, :]
            
            # 获取邻近通道索引（排除自身）
            neighbor_indices = indices[i, 1:]
            neighbor_indices = neighbor_indices.astype(int)
            
            # 检查是否有邻近通道
            if len(neighbor_indices) == 0:
                correlations[orig_idx] = 0
                continue
            
            # 获取邻近通道的原始索引
            neighbor_orig_indices = [valid_positions[j] for j in neighbor_indices]
            
            # 获取邻近通道数据
            neighbor_data = data[neighbor_orig_indices, :].T
            
            # 创建RANSAC回归模型
            model = RANSACRegressor(
                min_samples=min(3, len(neighbor_orig_indices)), 
                stop_probability=0.98
            )
            
            # 训练模型并预测
            try:
                model.fit(neighbor_data, ch_data)
                pred_data = model.predict(neighbor_data)
                
                # 计算预测值与实际值的相关系数
                corr = np.corrcoef(pred_data, ch_data)[0, 1]
                correlations[orig_idx] = np.abs(corr)
            except:
                correlations[orig_idx] = 0
        
        # 7. 检测低相关性通道
        print(correlations)
        bad_channels = []
        for i in range(len(eeg_raw.ch_names)):
            if correlations[i] < self.corr_threshold:
                bad_channels.append(ch_names[i])
        
        return bad_channels

class ICA_EEG:
    def __init__(self,raw_data, stim_analysis,non_stim_mask=None,n_components=0.99, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.raw_data = raw_data
        self.fastica = ICA(n_components=self.n_components, method='fastica', random_state=self.random_state)
        if stim_analysis:
            self.clean_raw = self.apply_ica_stim(non_stim_mask)
        else:
            self.clean_raw = self.apply_ica()

    def apply_ica(self):
        """
        快速ICA去伪迹
        """
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
        
        # 清理内存
        del raw_ica, ica_labels
        gc.collect()
        return raw_clean

    def apply_ica_stim(self, non_stim_mask):
        """
        快速ICA去伪迹（带刺激分析）
        """
        raw = self.raw_data.copy()
        original_annots = raw.annotations.copy()
        #获取非刺激部分数据
        non_stim_raw = mne.io.RawArray(raw.get_data()[:, non_stim_mask], raw.info.copy())
        self.fastica.fit(non_stim_raw)
        print("使用ICLabel标记成分...")
        ic_labels = label_components(non_stim_raw, self.fastica,method="iclabel")
        labels = ic_labels['labels']
        probabilities = ic_labels['y_pred_proba']
        print("标记伪迹成分...")
        artefact_threshold = 0.7  # 概率阈值
        artefact_components = []
        
        for idx, (label, prob) in enumerate(zip(labels, probabilities)):
            # 只考虑高概率的伪迹
            if prob > artefact_threshold and label != 'brain':
                artefact_components.append(idx)
     
        print(f"标记的伪迹成分: {artefact_components}")
        
        # 7. 仅对非刺激部分应用ICA
        denoised_non_stim = self.fastica.apply(non_stim_raw.copy(), exclude=artefact_components)
        # 8. 重建完整数据
        denoised_data = raw.get_data().copy()  # 创建原始数据的副本
        # 8非刺激部分应用ICA结果
        denoised_data[:, non_stim_mask] = denoised_non_stim.get_data() #(18, 3948000)
        # 9. 创建新的Raw对象
        denoised_raw = mne.io.RawArray(denoised_data, raw.info)
        denoised_raw.set_annotations(original_annots)
        del raw,non_stim_raw,denoised_non_stim
        gc.collect()
        return denoised_raw

class FASTER_EEG:
    def __init__(self, raw, stim_analysis=False,epoch_duration=1.0, max_bad_frac=0.2):
        self.epoch_duration = epoch_duration
        self.max_bad_frac = max_bad_frac
        if stim_analysis:
            print("FASTER处理中启用刺激分析模式")
            self.clean_raw, self.global_bad_channels = self.faster_remove_bad_epoch_stim(raw)
        else:
            print("FASTER处理中禁用刺激分析模式")
            self.clean_raw, self.global_bad_channels, self.processing_info = self.faster_remove_bad_epoch(raw)

    def faster_remove_bad_epoch(self, raw):
        """
        FASTER处理函数 - 整合三个FASTER步骤
        """
        # 创建副本避免修改原始数据
        raw_clean = raw.copy()
        sfreq = raw.info['sfreq']
        
        # 初始化处理信息记录
        processing_info = {
            'parameters': {
                'bad_channel_threshold': self.max_bad_frac,
                'epoch_duration': self.epoch_duration,
                'sampling_frequency': sfreq
            },
            'bad_epochs_indices': [],      # 标记为bad的epoch索引
            'repaired_epochs_indices': [], # 修复过的epoch索引
            'global_bad_channels': [],     # 全局坏通道列表
            'epoch_details': {}            # 每个epoch的详细处理记录
        }
        
        # 1. 创建固定长度epoch
        n_times_per_epoch = int(self.epoch_duration * sfreq)
        events = mne.make_fixed_length_events(raw_clean, duration=self.epoch_duration, overlap=0.0)
        
        # 创建Epochs对象
        epochs = mne.Epochs(
            raw_clean, 
            events, 
            tmin=0, 
            tmax=self.epoch_duration - 1/sfreq,
            baseline=None, 
            preload=True,
            reject=None, 
            flat=None
        )
        
        n_epochs, n_channels = len(epochs), len(epochs.ch_names)
        processing_info['parameters']['total_epochs'] = n_epochs
        processing_info['parameters']['total_channels'] = n_channels
        
        print(f"FASTER处理: {n_epochs}个epoch, {n_channels}个通道")
        
        # 2. 使用FASTER的三个步骤检测
        from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs
        # 第一步: 检测全局坏通道 (基于整个数据)
        print("步骤1: 检测全局坏通道...")
        global_bad_channels_step1 = find_bad_channels(epochs, thres=3)
        processing_info['global_bad_channels_step1'] = global_bad_channels_step1
        
        # 第二步: 检测坏epoch
        # print("步骤2: 检测坏epoch...")
        bad_epochs_detected = find_bad_epochs(epochs, thres=5.0)
        processing_info['bad_epochs_detected'] = bad_epochs_detected
        
        # 第三步: 检测每个epoch中的坏通道
        print("步骤3: 检测每个epoch中的坏通道...")
        bad_channels_per_epoch = find_bad_channels_in_epochs(epochs, thres=3)
        processing_info['bad_channels_per_epoch'] = bad_channels_per_epoch
        
        # 3. 计算全局坏通道（合并第一步结果和在超过30%的epoch中都被标记为坏的通道）
        channel_bad_count = {ch: 0 for ch in epochs.ch_names}
        
        for epoch_bad_chs in bad_channels_per_epoch:
            for ch_name in epoch_bad_chs:
                channel_bad_count[ch_name] += 1
        
        # 找出在超过30%的epoch中被标记为坏的通道
        epoch_based_global_bads = [
            ch for ch, count in channel_bad_count.items() 
            if count / n_epochs > 0.3
        ]
        
        # 合并两种方法检测到的全局坏通道
        global_bad_channels = list(set(global_bad_channels_step1 + epoch_based_global_bads))
        processing_info['global_bad_channels'] = global_bad_channels
        processing_info['epoch_based_global_bads'] = epoch_based_global_bads
        
        print(f"全局坏通道: {global_bad_channels}")
        print(f"检测到的坏epoch: {len([])}个")
        print(bad_epochs_detected)

        # 4. 处理每个epoch
        all_epoch_data = epochs.get_data()  # 获取所有epoch的数据
        
        for epoch_idx in range(n_epochs):
            epoch_info = {
                'bad_channels': bad_channels_per_epoch[epoch_idx],
                'is_bad_epoch_detected': epoch_idx in bad_epochs_detected,
                'action_taken': 'none',
                'repaired_channels': []
            }
            
            # 计算当前epoch的坏通道比例（包括全局坏通道）
            all_bad_channels = list(set(epoch_info['bad_channels'] + global_bad_channels))
            bad_ratio = len(all_bad_channels) / n_channels
            
            # 判断处理策略
            if bad_ratio > self.max_bad_frac or epoch_info['is_bad_epoch_detected']:
                # 坏通道比例超过阈值或FASTER检测为坏epoch → 标记为bad
                epoch_info['action_taken'] = 'marked_bad'
                epoch_info['bad_ratio'] = bad_ratio
                processing_info['bad_epochs_indices'].append(epoch_idx)
                
            elif epoch_info['bad_channels']:
                # 坏通道比例未超阈值但有坏通道 → 进行重建
                epoch_info['action_taken'] = 'repaired'
                epoch_info['repaired_channels'] = epoch_info['bad_channels']
                epoch_info['bad_ratio'] = bad_ratio
                processing_info['repaired_epochs_indices'].append(epoch_idx)
                
                # 使用插值法重建坏通道
                try:
                    epoch_data_repaired = self._repair_single_epoch(
                        all_epoch_data[epoch_idx], 
                        epochs.info, 
                        epoch_info['bad_channels']
                    )
                    all_epoch_data[epoch_idx] = epoch_data_repaired
                except Exception as e:
                    print(f"Error repairing epoch {epoch_idx}: {e}")
                    # epoch_info['error'] = str(e)
            
            # 记录当前epoch的处理信息
            processing_info['epoch_details'][f'epoch_{epoch_idx}'] = epoch_info
        
        # 5. 重建连续数据
        clean_data = all_epoch_data.transpose(1, 0, 2).reshape(n_channels, -1)
        
        # 确保数据长度与原始数据一致
        if clean_data.shape[1] != raw_clean.n_times:
            clean_data_adjusted = np.zeros((n_channels, raw_clean.n_times))
            min_length = min(clean_data.shape[1], raw_clean.n_times)
            clean_data_adjusted[:, :min_length] = clean_data[:, :min_length]
            clean_data = clean_data_adjusted
        
        # 6. 创建新的Raw对象
        clean_raw = mne.io.RawArray(clean_data, raw.info)
        
        # 恢复原始注释
        if raw.annotations is not None:
            clean_raw.set_annotations(raw.annotations.copy())
        
        # 7. 为标记为bad的epoch添加注释（相对时间）
        if processing_info['bad_epochs_indices']:
            # 计算每个bad epoch的起始时间（相对于raw开始）
            sfreq = raw.info['sfreq']
            bad_epoch_onsets = []
            bad_epoch_durations = []
            
            for epoch_idx in processing_info['bad_epochs_indices']:
                onset = epoch_idx * self.epoch_duration  # 相对于raw开始的秒数
                bad_epoch_onsets.append(onset)
                bad_epoch_durations.append(self.epoch_duration)
            
            # 创建注释
            annotations = mne.Annotations(
                onset=bad_epoch_onsets,
                duration=bad_epoch_durations,
                description=['bad_epoch'] * len(bad_epoch_onsets),
                orig_time=None  # 相对时间
            )
            
     
            clean_raw.set_annotations(annotations)
        
        # 8. 打印简要处理结果
        n_bad = len(processing_info['bad_epochs_indices'])
        n_repaired = len(processing_info['repaired_epochs_indices'])
        n_clean = n_epochs - n_bad - n_repaired
        
        print(f"\nFASTER处理完成:")
        print(f"  - 标记为bad的epoch: {n_bad}个")
        print(f"  - 已修复的epoch: {n_repaired}个")
        print(f"  - 干净epoch: {n_clean}个")
        print(f"  - 全局坏通道: {global_bad_channels}")
        
        return clean_raw, global_bad_channels, processing_info

    def _repair_single_epoch(self, epoch_data, info, bad_channels):
        """
        使用插值法修复单个epoch中的坏通道
        
        参数:
            epoch_data: 单个epoch的数据 (n_channels, n_times)
            info: mne.Info对象
            bad_channels: 需要修复的坏通道列表
        
        返回:
            repaired_data: 修复后的epoch数据
        """
        if not bad_channels:
            return epoch_data
        
        # 创建临时的EpochsArray对象
        temp_epoch = mne.EpochsArray(
            data=epoch_data[np.newaxis, :, :],
            info=info.copy(),
            events=np.array([[0, 0, 1]]),
            tmin=0
        )
        
        # 设置坏通道
        temp_epoch.info['bads'] = bad_channels
        
        # 进行插值修复
        temp_epoch.interpolate_bads(reset_bads=True)
        
        # 获取修复后的数据
        repaired_data = temp_epoch.get_data()[0]
        
        return repaired_data

    def faster_remove_bad_epoch_stim(self, raw,stim_start_prefix='Start of stimulation', stim_end_prefix='End of stimulation',):
        """
        FASTER 处理，保留时间连续性，区分刺激区和非刺激区
        """
        # 1. 保存原始注释
        original_annotations = raw.annotations.copy() if raw.annotations is not None else None
        epochs = mne.make_fixed_length_epochs(
                    raw,
                    duration=self.epoch_duration,
                    overlap=0.0,
                    preload=True,
                    verbose=True
                )
        
        # 4. 获取 epoch 数据
        epoch_data = epochs.get_data()
        n_epochs, n_channels, n_times = epoch_data.shape
        print(f"创建 {n_epochs} 个 epoch, 每个 epoch {n_times} 个时间点")
        
        # 4. 分离刺激期和非刺激期 epoch
        stim_intervals = self.identify_stim_intervals(raw, stim_start_prefix, stim_end_prefix)
        stim_mask = self.create_stim_mask(epochs, stim_intervals)
        non_stim_mask = ~stim_mask
        
        epoch_data = epochs.get_data()
        n_epochs, n_channels, n_times = epoch_data.shape
        
        # 5. 仅对非刺激期应用 FASTER 检测
        from mne_faster import find_bad_epochs, find_bad_channels_in_epochs
        bad_channels_per_epoch = [[] for _ in range(n_epochs)]
        bad_epochs = np.zeros(n_epochs, dtype=bool)
        # 只在非刺激期检测坏 epoch 和坏通道
        if np.any(non_stim_mask):
            non_stim_epochs = epochs[non_stim_mask]
            
            # 检测非刺激期的坏 epoch
            non_stim_bad_epochs = find_bad_epochs(non_stim_epochs,thres=3)     
            # 检测非刺激期的坏通道
            non_stim_bad_channels = find_bad_channels_in_epochs(non_stim_epochs,thres=3)
            # 映射回原始索引
            non_stim_indices = np.where(non_stim_mask)[0]
            for i, epoch_idx in enumerate(non_stim_indices):
                bad_epochs[epoch_idx] = i in non_stim_bad_epochs
                bad_channels_per_epoch[epoch_idx] = non_stim_bad_channels[i]
        
        # 6. 计算全局坏通道 (仅基于非刺激期)
        channel_counts = {}
        for epoch_idx in np.where(non_stim_mask)[0]:
            for chan in bad_channels_per_epoch[epoch_idx]:
                channel_counts[chan] = channel_counts.get(chan, 0) + 1
        
        n_non_stim_epochs = np.sum(non_stim_mask)
        global_bad_channels = [
            chan for chan, count in channel_counts.items()
            if n_non_stim_epochs > 0 and count / n_non_stim_epochs > 0.2
        ]
        logging.info(f"全局坏通道 (基于非刺激期): {global_bad_channels}")
        
        # 7. 处理每个 epoch
        logging.info(f"开始处理非刺激期数据，共 {n_non_stim_epochs} 个 epoch")
        cleaned_data = []
        for epoch_idx in range(n_epochs):
            # 刺激期 epoch - 直接保留
            if stim_mask[epoch_idx]:
                cleaned_data.append(epoch_data[epoch_idx])
                continue
            
            # 非刺激期 epoch - 应用处理
            if bad_epochs[epoch_idx]:
                # 坏 epoch 处理策略
                bads = bad_channels_per_epoch[epoch_idx]
                logging.info(f"process epoch {epoch_idx} 通道: {bads}")
                if len(bads) > 1: 
                    epoch = epochs[epoch_idx]
                    epoch.info['bads'] = list(set(bads)) #global_bad_channels
                    
                    # 重建坏通道
                    epoch.interpolate_bads()
                    cleaned_data.append(epoch.get_data()[0])
                else:
                    # 坏通道比例小于阈值，直接保留
                    cleaned_data.append(epoch_data[epoch_idx])
            else:
                # 好 epoch，直接添加
                cleaned_data.append(epoch_data[epoch_idx])
        
        # 8. 重建连续数据
        clean_data = np.concatenate(cleaned_data, axis=1)
        clean_raw = mne.io.RawArray(clean_data, epochs.info)
        # 9. 恢复原始注释
        if original_annotations is not None:
            clean_raw.set_annotations(original_annotations)
        del raw, epochs
        gc.collect() #清理内存
        return clean_raw, global_bad_channels

    def identify_stim_intervals(self, raw, start_prefix='Stim_Start', end_prefix='Stim_End'):
        """识别刺激时间段"""
        stim_intervals = []
        start_events = {}
        end_events = {}     
        # 预编译正则表达式提高效率
        pattern = re.compile(r'\[(\d+),\s*(\d+)\]')  
        for ann in raw.annotations:
            desc = str(ann['description'])
            # 匹配开始标记
            if desc.startswith(start_prefix):
                match = pattern.search(desc)
                if match:
                    event_id = tuple(map(int, match.groups()))
                    start_events[event_id] = ann['onset']
            # 匹配结束标记
            elif desc.startswith(end_prefix):
                match = pattern.search(desc)
                if match:
                    event_id = tuple(map(int, match.groups()))
                    end_events[event_id] = ann['onset'] + ann['duration']
        
        # 创建刺激间隔
        for event_id in set(start_events.keys()) | set(end_events.keys()):
            start = start_events.get(event_id)
            end = end_events.get(event_id)
            
            if start and end and start < end:
                stim_intervals.append((start, end))
        
        # 合并重叠区间
        return self.merge_intervals(stim_intervals)

    def merge_intervals(self, intervals):
        """合并重叠的时间区间"""
        if not intervals:
            return []
        
        # 按开始时间排序
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        current_start, current_end = intervals[0]
        
        for start, end in intervals[1:]:
            if start <= current_end:
                # 扩展当前区间
                current_end = max(current_end, end)
            else:
                # 保存当前区间并开始新区间
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # 添加最后一个区间
        merged.append((current_start, current_end))
        # 对每个合并后的区间扩展8s
        expanded_stim = []
        for start, end in merged:
            # 第一个值减8（确保不小于0）
            expanded_start = max(0, start - 2)  
            # 第二个值加8
            expanded_end = end + 2
            expanded_stim.append((expanded_start, expanded_end))
        return expanded_stim
    
    def create_stim_mask(self, epochs, stim_intervals):
        """高效创建刺激期掩码 (使用向量化操作)"""
        if not stim_intervals:
            return np.zeros(len(epochs), dtype=bool)
        
        sfreq = epochs.info['sfreq']
        n_epochs = len(epochs)
        
        # 获取所有事件的采样点
        event_samples = epochs.events[:, 0]
        
        # 计算所有 epoch 的开始和结束时间
        epoch_starts = event_samples / sfreq
        epoch_ends = epoch_starts + (epochs.tmax - epochs.tmin)
        
        # 提取刺激区间
        stim_starts = np.array([s for s, _ in stim_intervals])
        stim_ends = np.array([e for _, e in stim_intervals])
        
        # 使用向量化操作检查重叠
        # 初始化重叠标志
        overlaps = np.zeros(n_epochs, dtype=bool)
        
        # 检查每个刺激区间
        for stim_start, stim_end in zip(stim_starts, stim_ends):
            # 找到与当前刺激区间重叠的 epoch
            start_overlap = np.logical_and(epoch_starts < stim_end, epoch_ends > stim_start)
            overlaps = np.logical_or(overlaps, start_overlap)
        
        return overlaps
    

class preprocess_EEG:
    """
    class preprocess_EEG:预处理EEG数据
    """
    def __init__(self, file_paths, only_include_EEG_channels=True):
        
        # 读取bdf或edf文件
        if len(file_paths) <= 1:
            self.file_path = file_paths[0]
            if os.path.basename(self.file_path).endswith('.bdf'):
                self.raw = mne.io.read_raw_bdf(self.file_path, preload=True)
            elif os.path.basename(self.file_path).endswith('.edf'):
                self.raw = mne.io.read_raw_edf(self.file_path, preload=True)
            elif os.path.basename(self.file_path).endswith('.fif'):
                self.raw = mne.io.read_raw_fif(self.file_path, preload=True)
            else:
                raise ValueError('文件格式错误,请使用bdf或edf格式')
            meas_date = self.raw.info['meas_date']
        elif type(file_paths) is str:
            self.file_path = file_paths
            if os.path.basename(self.file_path).endswith('.bdf'):
                self.raw = mne.io.read_raw_bdf(self.file_path, preload=True)
            elif os.path.basename(self.file_path).endswith('.edf'):
                self.raw = mne.io.read_raw_edf(self.file_path, preload=True) 
            elif os.path.basename(self.file_path).endswith('.fif'):
                self.raw = mne.io.read_raw_fif(self.file_path, preload=True)
            else:
                raise ValueError('文件格式错误,请使用bdf或edf格式')
            meas_date = self.raw.info['meas_date']
        else:
            if "preprocessed_merge_RAW.edf" in os.listdir(os.path.dirname(file_paths[0])):
                print("已存在预处理合并文件，跳过合并步骤")
                self.raw = mne.io.read_raw_edf(os.path.dirname(file_paths[0]) + '/preprocessed_merge_RAW.edf', preload=True)
                meas_date = self.raw.info['meas_date'] 
            else:
                self.raw = self.merge_bdf_files(file_paths) #合并多个bdf文件
                meas_date = self.raw.info['meas_date'] 
                save_path = os.path.dirname(file_paths[0])
                mne.export.export_raw(save_path + '/preprocessed_merge_RAW.edf', self.raw, fmt='edf',overwrite=True)
                print(f"Preprocessed data saved to {save_path}/preprocessed_merge_RAW.edf")

        self.original_raw = self.raw.copy()  #保存原始数据
        self.original_sfreq = self.raw.info['sfreq']
        self.orgin_channel_list = self.raw.info['ch_names']  #原始通道
        self.only_include_EEG_channels = only_include_EEG_channels

        #裁剪
        print(f"总时长：{self.raw.times[-1]}s")
        # target_utc = datetime.datetime(meas_date.year, meas_date.month, meas_date.day, 20, 0, 0, tzinfo=meas_date.tzinfo)
        # start_time = (target_utc - meas_date).total_seconds()
        # if start_time < 0:
        #     start_time =0
        # if meas_date is None:
        #     raise ValueError("RAW文件没有包含测量日期信息(meas_date)，无法进行绝对时间裁剪。")
        # print(f"裁剪开始时间：{start_time}")
        # # total_duration = self.raw.times[-1]
        # # crop_duration = min(10 * 60 * 60, total_duration - start_time)
        # # if start_time + crop_duration > total_duration:
        # #     crop_duration = total_duration - start_time
        # #     print(f"调整裁剪时长以避免超出数据范围")
        # # self.raw = self.raw.crop(tmin=start_time, tmax=start_time + crop_duration)
        # print(f"裁剪后时长：{self.raw.times[-1]}s")
        self.raw = handle_bipolar_channels(self.raw) #标准化EEG数据的通道信息,包括通道类型和参考电极
        self.raw = channel_type_detection(self.raw)  #通道类型检测和修正 
        self.raw = filter_detrend(self.raw) # EEG通道和其他通道分别滤波
    
        
        self.raw.load_data()

        self.resample(new_fs = 128)  #重采样
        self.get_non_eeg_channels()  # 获取非EEG通道,同时去除心电伪迹

        #保存滤波处理后的数据
        # save_path = os.path.join(os.path.dirname(file_paths[0]),"resample_512.edf")
        # mne.export.export_raw(save_path, self.raw, fmt='edf', overwrite=True)
         
        ## 绘制EMG的时频图
        # self.plot_emg(save_path=r"D:\analysis\long_stim_bdf\test\emg.pdf")

    def remove_ecg(self):
        """去除心电伪迹"""
        logging.info("正在去除心电伪迹...")
     
        if self.ecg_reference is not None:
            ecg_remover = MultiChannelECGArtifactRemover(self.raw.info['sfreq'])
            # a=self.no_eeg_raw.get_data(picks="ECG")
            # self.ecg_reference =self.no_eeg_raw.get_data(picks="ECG")[0,:]
            self.raw = ecg_remover.remove_ecg_from_raw(self.raw,self.ecg_reference)
            print("含有心电参考通道，self.raw所有通道的心电伪迹去除完成**********************")
        else:
            print("没有心电参考通道，心电伪迹去除失败**********************")
    
    def merge_channels(self, eeg_raw, non_eeg_raw):
        """使用MNE内置方法合并（需要MNE 1.0+）"""
        if non_eeg_raw is None:
            return eeg_raw
        

            # MNE的add_channels方法可以自动处理
        merged = eeg_raw.copy()
        
        # 需要确保两个raw对象的时间对齐
        # 这里假设它们从同一时间开始
        merged.add_channels([non_eeg_raw], force_update_info=True)
        
        return merged

    def get_non_eeg_channels(self):
        """使用通道类型匹配方法获取非EEG通道数据"""
        # 获取所有通道类型
        channel_types = self.raw.get_channel_types()
        
        # 找出所有非EEG通道
        non_eeg_channels = [ch for i, ch in enumerate(self.raw.ch_names) 
                        if channel_types[i] != 'eeg' and channel_types[i] != 'misc']
        
        if non_eeg_channels:
            # 使用通道类型获取ECG参考信号
            ecg_channels = [ch for i, ch in enumerate(self.raw.ch_names) 
                        if self.raw.get_channel_types()[i] == 'ecg']
            
            if ecg_channels:
                # 如果有多个ECG通道，选择第一个
                self.ecg_reference = self.raw.get_data(picks=ecg_channels[0])[0, :]
                print(f"使用ECG通道，可去除心电伪迹: {ecg_channels[0]}")
                
            else:
                self.ecg_reference = None
                print("未找到ECG通道")
            # 所有通道去除心电伪迹，self.raw为去除心电伪迹后的数据
            self.remove_ecg()
            self.no_eeg_raw = self.raw.copy().pick_channels(non_eeg_channels)
        
            emg_channels = [ch for i, ch in enumerate(self.no_eeg_raw.ch_names) 
                        if self.no_eeg_raw.get_channel_types()[i] == 'emg']
            
            if emg_channels:
                # 如果有多个EMG通道，选择第一个
                self.emg = self.no_eeg_raw.get_data(picks=emg_channels[0])[0, :]
                print(f"EMG通道: {emg_channels}")
            else:
                self.emg = None
                print("未找到EMG通道")
                
            # 获取EOG通道数据
            eog_channels = [ch for i, ch in enumerate(self.no_eeg_raw.ch_names) 
                        if self.no_eeg_raw.get_channel_types()[i] == 'eog']
            
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
        
        # SHHS非eeg通道处理
        all_non_eeg_channels_shhs = [ch for i, ch in enumerate(self.raw.ch_names) 
                if channel_types[i] != 'eeg']
        self.all_no_eeg_raw = self.raw.copy().pick_channels(all_non_eeg_channels_shhs)

    def plot_emg(self, save_path=r"D:\analysis\long_stim_bdf\test\emg.pdf", dpi=300): 
       
        plt.figure(figsize=(12, 8))
        f, t, Sxx = spectrogram(
            self.emg * 1e6,
            fs=self.original_sfreq,
            nperseg=int(self.original_sfreq)*4,
            noverlap=int(self.original_sfreq)*2,  # 50%重叠
            window='hann',
            # scaling='density'
        )
        
        # 创建图形
        fig = plt.figure(figsize=(12, 8))
        
        # 创建网格布局
        ax = plt.subplot2grid((6, 1), (1, 0), rowspan=5)
        
        # 提取所需频率范围 (15-100Hz)
        freq_mask = (f >= 15) & (f <= 100)
        f_sub = f[freq_mask]
        Sxx_sub = Sxx[freq_mask, :]
        
        # 转换为分贝 (dB)
        Sxx_db = 10 * np.log10(Sxx_sub)  # 添加小值避免log(0)
        
        
        # 绘制时频图
        im = ax.pcolormesh(
            t/60,  # 转换为分钟
            f_sub, 
            Sxx_db,
            shading='auto',
            cmap="viridis",
            vmin=-40,
            vmax=20
        )
        
        # 设置y轴刻度和标签
        ax.set_yticks([15, 30, 50, 70, 100])
        ax.set_ylabel('Frequency(Hz)', loc='center', labelpad=30)
        ax.set_title(f'EMG')
        ax.set_xlabel('Time (min)')  # 添加x轴标签
        
        # 设置轴线样式
        ax.spines['right'].set_color('none')
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_color('none')
        ax.spines['top'].set_linewidth(2)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')
        
        plt.tight_layout()
    
        # 保存为PNG文件
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"时频图已保存至: {save_path}")
        
        # 关闭图形以释放内存
        plt.close()
        
    def resample(self, new_fs=250):
        """重采样"""
        self.raw.resample(new_fs, npad="auto")
        self.fs = new_fs
        print(f"******************重采样完成，新采样率：{self.fs}Hz**************")

    def merge_bdf_files(self, file_paths):
        """
        按顺序拼接多个BDF文件，去除每个文件中"Recording ends"之后的数据
        然后调整Annotations为绝对时间
        
        参数:
            file_paths (list): 按顺序排列的BDF文件路径列表
        
        返回:
            combined_raw: 拼接后的Raw对象
            first_raw: 第一个文件的原始Raw对象
        """
        if len(file_paths) < 1:
            raise ValueError("至少需要一个文件进行拼接")
        
        # 初始化变量
        all_raws = []        # 存储所有处理后的Raw对象
        all_annotations = []  # 存储所有调整后的Annotations
        cumulative_onset = 0   # 累积时间偏移量（秒）
        first_raw = None
        
        for i, file_path in enumerate(file_paths):
            if file_path.endswith(".edf"):
                continue
            print(f"\n处理文件 {i+1}/{len(file_paths)}: {file_path}")
            
            # 1. 读取当前BDF文件
            raw = mne.io.read_raw_bdf(file_path, preload=True, units="uV")
            raw.load_data()
            
            if i == 0:
                first_raw = raw
                
            original_duration = raw.times[-1]
            print(f"  原始持续时间: {original_duration:.3f}秒")
            print(f"  采样频率: {raw.info['sfreq']:.1f}Hz")
            print(f"  通道数: {len(raw.ch_names)}")
            
            # 2. 查找"Recording ends"标记
            end_marker_time = None
            if raw.annotations is not None and len(raw.annotations) > 0:
                for idx, desc in enumerate(raw.annotations.description):
                    desc_lower = desc.lower()
                    if "recording ends" in desc_lower:
                        end_marker_time = raw.annotations.onset[idx]
                        end_marker_duration = raw.annotations.duration[idx]
                        end_time = end_marker_time + end_marker_duration
                        print(f"  找到'Recording ends'标记: 在 {end_marker_time:.3f}s, 持续 {end_marker_duration:.3f}s")
                        
                        # 3. 裁剪Recording ends之后的数据
                        if end_time < original_duration - 0.1:  # 确保有数据可裁剪
                            print(f"  裁剪 {end_time:.3f}s 之后的数据，共去除 {original_duration - end_time:.3f}s")
                            raw.crop(tmax=end_time)
                            # 更新annotations，移除被裁剪部分的标记
                            if len(raw.annotations) > 0:
                                # 只保留onset在裁剪时间内的annotations
                                keep_mask = raw.annotations.onset <= end_time
                                if np.any(keep_mask):
                                    # 创建新的Annotations对象
                                    raw.set_annotations(mne.Annotations(
                                        onset=raw.annotations.onset[keep_mask],
                                        duration=raw.annotations.duration[keep_mask],
                                        description=[raw.annotations.description[j] for j in range(len(raw.annotations)) 
                                                if keep_mask[j]]
                                    ))
                                else:
                                    # 如果没有annotation在裁剪时间内，清空annotations
                                    raw.set_annotations(None)
                            
                            print(f"  裁剪后持续时间: {raw.times[-1]:.3f}s")
                        else:
                            print(f"  'Recording ends'标记已在文件末尾，无需裁剪")
                        break
                if end_marker_time is None:
                    print(f"  未找到'Recording ends'标记，保留完整文件")
            else:
                print(f"  文件无annotations，保留完整文件")
            
            # 4. 获取当前文件的annotations
            orig_annot = raw.annotations
            
            # 5. 如果是第一个文件，保留所有原始Annotations
            if i == 0:
                # 存储原始Raw和Annotations
                all_raws.append(raw)
                all_annotations.append(orig_annot)
                
                # 记录第一个文件的总时间作为偏移基准
                cumulative_onset = raw.times[-1]
                print(f"  首个文件结束时间: {cumulative_onset:.3f}秒")
                continue
            
            # 6. 后续文件处理
            # 6.1 计算这个文件的持续时间（秒）
            current_duration = raw.times[-1]
            
            # 6.2 如果文件有Annotations，调整onset时间为绝对时间
            modified_annot = None
            if orig_annot is not None and len(orig_annot) > 0:
                # 创建新的onset数组，加上累积偏移量
                new_onset = orig_annot.onset + cumulative_onset
                
                # 创建新的Annotations对象
                modified_annot = mne.Annotations(
                    onset=new_onset,
                    duration=orig_annot.duration,
                    description=orig_annot.description
                )
                if len(modified_annot) > 0:
                    print(f"  调整 {len(modified_annot)} 个Annotations")
                    print(f"    第一个事件: {orig_annot.onset[0]:.3f}s → {modified_annot.onset[0]:.3f}s")
            else:
                print(f"  没有Annotations需要调整")
                modified_annot = orig_annot
            
            # 6.3 存储调整后的Annotations
            all_annotations.append(modified_annot)
            
            # 6.4 存储当前Raw对象
            all_raws.append(raw)
            
            # 6.5 更新累积时间偏移量（加上当前文件持续时间）
            cumulative_onset += current_duration
            print(f"  累积时间偏移量: {cumulative_onset:.3f}秒")
        
        # 7. 将所有处理后的Raw文件拼接成一个
        print(f"\n开始拼接 {len(all_raws)} 个文件...")
        
        if len(all_raws) > 1:
            # 检查所有文件的基本信息
            for idx, raw in enumerate(all_raws):
                print(f"  文件{idx+1}: {raw.times[-1]:.3f}秒, {len(raw.ch_names)}通道, {raw.info['sfreq']:.1f}Hz")
            
            # 使用on_mismatch='ignore'避免因微小差异导致的错误
            combined_raw = mne.concatenate_raws(
                all_raws, 
                preload=False,
                on_mismatch='ignore'  # 忽略小的不匹配
            )
            
            # 转换为uV
            combined_raw.load_data()
            
            # 检查是否有BAD/EDGE边界标记
            if combined_raw.annotations is not None and len(combined_raw.annotations) > 0:
                boundary_count = 0
                for desc in combined_raw.annotations.description:
                    if 'boundary' in desc.lower():
                        boundary_count += 1
                
                if boundary_count > 0:
                    print(f"  注意: 拼接后检测到 {boundary_count} 个边界标记")
                    
                    # 可以选择移除边界标记
                    remove_boundary_indices = []
                    for idx, desc in enumerate(combined_raw.annotations.description):
                        if 'boundary' in desc.lower():
                            remove_boundary_indices.append(idx)
                    
                    if remove_boundary_indices:
                        # 保留非边界标记
                        keep_indices = [i for i in range(len(combined_raw.annotations)) 
                                    if i not in remove_boundary_indices]
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
        del all_raws  # 释放内存
        combined_raw.set_meas_date(first_raw.info['meas_date'])
        return combined_raw

    def set_annotations(self, processing_info,epoch_duration):
        data_duration = self.eeg_raw.n_times / self.eeg_raw.info['sfreq']  # 当前数据段的总时长
        bad_epochs_relative = processing_info.get('bad_epochs_indices', [])
            # 过滤掉超出数据范围的onset时间，并转换为绝对时间
        valid_onsets = []
        valid_durations = []
        valid_descriptions = []
        if bad_epochs_relative:

            for i, onset_relative in enumerate(bad_epochs_relative):
                # 检查相对onset是否在当前数据段时间范围内
                if onset_relative <= data_duration:
                    # 确保duration不超过数据末尾
                    remaining_duration = data_duration - onset_relative
                    actual_duration = min(epoch_duration, remaining_duration)
                    
                    if actual_duration > 0:
                        # 关键：转换为绝对时间（相对于原始数据起始）
                        onset_absolute = onset_relative 
                        
                        valid_onsets.append(onset_absolute)
                        valid_durations.append(actual_duration)
                        valid_descriptions.append(f'bad_epoch_{i}')
                else:
                    print(f"  跳过超出范围的epoch: 相对时间={onset_relative:.2f}s, 数据段结束于{data_duration:.2f}s")
        if self.eeg_raw.annotations is not None:
            # 将两个注释对象的onset, duration, description合并
            combined_onset = np.concatenate([self.eeg_raw.annotations.onset, valid_onsets])
            combined_duration = np.concatenate([self.eeg_raw.annotations.duration, valid_durations])
            combined_description = np.concatenate([self.eeg_raw.annotations.description, valid_descriptions])
            # 创建新的Annotations对象
            from mne import Annotations
            all_annotations = Annotations(
                onset=combined_onset,
                duration=combined_duration,
                description=combined_description,
                orig_time=self.eeg_raw.annotations.orig_time  # 保持原始时间参考
            )
            self.eeg_raw.set_annotations(all_annotations)
           
    def process_main(self, useICA=False, useFASTER=True,remove_bad_channels=True,stim_analysis=False):
        
        self.eeg_raw = self.raw.copy().pick_types(eeg=True) # 仅保留EEG通道
        if self.all_no_eeg_raw:
            logging.info(f"EEG通道{self.eeg_raw.info['ch_names']}")
            logging.info(f"非EEG通道{self.all_no_eeg_raw.info['ch_names']}")                 
        else:
            logging.info(f"EEG通道{self.eeg_raw.info['ch_names']}")
            logging.info("无非EEG通道")
        
        # 初始化
        full_data = np.zeros((len(self.eeg_raw.ch_names), self.eeg_raw._data.shape[1]))
        eeg_raw_good = self.eeg_raw.copy()
        good_channels = self.eeg_raw.info['ch_names'].copy()
        global_bad_channels,bad_channels = [],[]

        if stim_analysis:
            print("获取非刺激部分数据")
            non_stim_raw,non_stim_mask,non_stim_mask_time = detect_stim_intervals(self.eeg_raw)
        else:
            non_stim_raw = self.eeg_raw.copy()
            
        # CleanRawData 检测坏通道
        if remove_bad_channels:
            cleaner = CleanRawData()
            # 参数设置
            cleaner.flat_duration = 5.0     # 平坦信号持续时间阈值（秒）
            cleaner.flat_threshold = 1e-6   # 平坦信号幅度变化阈值
            cleaner.noise_threshold = 4     # 高频噪声标准差阈值
            cleaner.corr_threshold = 0.8    # 最小相关阈值
            cleaner.n_neighbors = 3         # 邻近通道数量
            bad_channels = cleaner.detect_bad_channels(non_stim_raw)
            
            if bad_channels:
                logging.info(f"CleanRawData检测到坏通道: {bad_channels}, 先移除进行后续处理")
                bad_channel_data = self.eeg_raw.copy().pick(bad_channels).get_data()# 保存原始坏通道数据 (用于后续恢复)
                bad_idx = [self.eeg_raw.ch_names.index(ch) for ch in bad_channels]# 填充坏通道数据 
                full_data[bad_idx] = bad_channel_data[:,:eeg_raw_good._data.shape[1]]

                good_channels = [ch for ch in self.eeg_raw.ch_names if ch not in bad_channels]
                eeg_raw_good = self.eeg_raw.copy().pick(good_channels)

        if stim_analysis:
            logging.info("进行含有刺激marker数据分析处理")
            if useICA:
                logging.info("应用ICA")
                ica = ICA_EEG(eeg_raw_good,stim_analysis=True,non_stim_mask=non_stim_mask)
                eeg_raw_good = ica.clean_raw

            if useFASTER:
                logging.info("应用FASTER")
                faster = FASTER_EEG(eeg_raw_good, stim_analysis=True,epoch_duration=1.0, max_bad_frac=0.3)
                eeg_raw_good, global_bad_channels = faster.clean_raw, faster.global_bad_channels
                # self.set_annotations(processing_info,epoch_duration=1.0)
        
        else:
            logging.info("进行不含刺激marker实验数据分析处理")      
            if useICA:
                logging.info("应用ICA")
                ica = ICA_EEG(eeg_raw_good,stim_analysis=False)
                eeg_raw_good = ica.clean_raw
            
            if useFASTER:
                logging.info("应用FASTER")
                faster = FASTER_EEG(eeg_raw_good, stim_analysis=False,epoch_duration=1.0, max_bad_frac=0.5)
                eeg_raw_good, global_bad_channels,processing_info = faster.clean_raw, faster.global_bad_channels, faster.processing_info
                self.set_annotations(processing_info,epoch_duration=1.0)

        #合并bad通道数据和处理后数据
        good_idx = [self.eeg_raw.ch_names.index(ch) for ch in good_channels]
        full_data[good_idx] = eeg_raw_good.get_data()
        self.eeg_raw._data = full_data # 更新原始raw对象
         #合并所有需要插值的坏通道
        repaired_bad_channels = list(set(global_bad_channels + bad_channels))
        if len(repaired_bad_channels)!=0:
            self.eeg_raw.info['bads'] = repaired_bad_channels
            self.eeg_raw.interpolate_bads(method='spline',reset_bads=True) #'spline', 'MNE', and 'nan'
            logging.info(f"完成坏通道的插值重建: {repaired_bad_channels}")

        #合并eeg和non_eeg通道
        self.raw = self.eeg_raw.copy()
        self.raw.add_channels([self.all_no_eeg_raw], force_update_info=True)

        return  self.raw


def find_eeg_files(folder_path):
    file_path_list = [os.path.join(root, file) 
                    for root, dirs, files in os.walk(folder_path) 
                    for file in natsorted(files) if file.endswith('.bdf') or file.endswith('.edf')]
    return file_path_list

def process_fun(file_path):
    eeg = preprocess_EEG(file_path)
    clean_raw = eeg.process_main(useICA=False, useFASTER=True,remove_bad_channels=False,stim_analysis=False)
    return clean_raw

def main():
    # file_paths = natsorted(glob.glob(r"\\172.16.6.5\project\DATA\Sleep_Datasets\SHHS\shhs\polysomnography\edfs\shhs1\*.edf"))
    file_paths = find_eeg_files(r"\\172.16.6.5\project\DATA\Sleep_Datasets\SHHS\shhs\polysomnography\edfs")
    bids_save_dir = r"\\172.16.6.5\sleep\Neuromodulation\human\test"
    # 1. 初始化BIDS数据处理器
    processor = BIDSProcessor(
        bids_root=bids_save_dir,
        dataset_name="shhs11",
        task="sleep",
        session="ses-1",
        start_subject_id=1  # 从0000001开始
    )

    # 3. 批量处理
    results = processor.batch_process_files(
        file_paths=file_paths,
        task="sleep",
        process_func=process_fun,
        pipeline_name="FASTER_pipeline",
        description="clean"
    )

    # 4. 查看汇总信息
    processor.print_summary()

if __name__ == '__main__':
    mian()

