
"""
预处理EEG信号：
1.读取bdf/edf文件 (如有需要则合并所有bdf为一个文件)
2.滤波、平均参考、基线校正
3.重采样
4.ICA去伪迹
5.分段去除坏epoch(2s)

"""
import mne,os,re
from scipy.signal import detrend
import concurrent.futures
import logging,time
from mne.preprocessing import ICA
from mne_icalabel import label_components 
import numpy as np
from scipy.signal import welch
from scipy import signal
from mne.channels import make_standard_montage
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs
from scipy import stats
from scipy.signal import welch, butter, filtfilt
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from mne_faster import find_bad_channels_in_epochs,find_bad_epochs
from collections import defaultdict
import datetime
from scipy.signal import spectrogram
from ecgdetectors import Detectors
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import neurokit2 as nk
import re
from fuzzywuzzy import fuzz, process
# from analysis_stim_fun.remove_ECG import MultiChannelECGArtifactRemover
import re
from fuzzywuzzy import fuzz, process
import random

import numpy as np
import scipy.stats as stats
from scipy.signal import iirfilter, filtfilt
import mne
import numpy as np
from mne.channels import make_standard_montage, make_dig_montage
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
# from .DeepEEGDenoiser import DeepEEGDenoiser #DeepEEGDenoiser
random.seed(42)
np.random.seed(42)


def setup_logging(base_dir):
    """设置日志记录器"""
    # 创建日志目录
    log_dir = os.path.join(base_dir, "analysis_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"processing_{timestamp}.log")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_dir
base_dir = r'D:\analysis\long_stim_bdf\test_preprocess'
log_dir = setup_logging(base_dir)
print(f"日志文件保存在: {log_dir}")

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
    处理双极导联通道的改进方案
    
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
        # 找出双极导联通道（通常标记为 'eeg' 但名称包含 '-'）
    bipolar_chs = []
    for i, ch_name in enumerate(raw.ch_names):
        if ch_types[i] == 'eeg' and '-' in ch_name:
            bipolar_chs.append(ch_name)
       # 如果没有双极导联，直接返回
    if bipolar_chs:   
        print(f"检测到双极导联通道: {bipolar_chs}")
        for ch in bipolar_chs:
            # 分解双极导联名称
            parts = ch.split('-')
            if len(parts) >= 2:
                first_ch = parts[0]
                
                # 尝试使用第一个通道的位置
                if first_ch in std_ch_pos:
                    custom_ch_pos[ch] = std_ch_pos[first_ch]
                    print(f"为双极导联 {ch} 使用 {first_ch} 的位置")
                else:
                    # 使用默认中心位置
                    center_pos = np.array([0, 0, 0.1])
                    custom_ch_pos[ch] = center_pos
                    print(f"为双极导联 {ch} 使用默认中心位置")
            else:
                # 使用默认中心位置
                center_pos = np.array([0, 0, 0.1])
                custom_ch_pos[ch] = center_pos
                print(f"为双极导联 {ch} 使用默认中心位置")
    
    # 使用 make_dig_montage 创建蒙太奇
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
        
        # if start and end and start < end:
            # if end - start < 2 *60:
            #     stim_intervals.append((start-60, end+10*60)) #sham
            # else:
            #     stim_intervals.append((start-60, end+60))  ### ###########长时程刺激去掉60秒，短时程刺激去掉2秒
        if start and end and start < end:
            if end - start < 28:  #短时程刺激
                stim_intervals.append((start-2, end+30)) #sham
            else:
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
    src_data = raw.get_data() #(22, 15844000)
    non_stim_data = raw.get_data()[:, non_stim_mask] #(22, 10968959)
    non_stim_raw = mne.io.RawArray(non_stim_data, raw.info.copy())

    return non_stim_raw, non_stim_mask,non_stim_mask_time


def filter_detrend(raw, method="fir"):
    """滤波、平均参考、基线校正（支持 FIR 或 IIR Butterworth 滤波器）"""
    raw.load_data()
    
    # 识别通道类型
    eeg_picks = mne.pick_types(raw.info, eeg=True, emg=False, ecg=False, eog=False, misc=False)
    emg_picks = mne.pick_types(raw.info, emg=True, ecg=False, eog=False, misc=False)
    ecg_picks = mne.pick_types(raw.info, emg=False, ecg=True, eog=False, misc=False)
    eog_picks = mne.pick_types(raw.info, emg=False, ecg=False, eog=True, misc=False)
    
    # 根据 method 设置滤波参数
    if method == "iir":
        # Butterworth IIR 滤波器参数（4阶，）
        iir_params = dict(ftype='butter', order=2)
        filter_kwargs = dict(method='iir', iir_params=iir_params)
        notch_kwargs = dict(method='iir', iir_params=iir_params)
    else:  # 默认 FIR
        filter_kwargs = dict(method='fir', fir_design='firwin')
        notch_kwargs = dict(method='fir', fir_design='firwin')
    
    # ========== EEG 通道 ==========
    if len(eeg_picks) > 0:
        raw.filter(l_freq=0.5, h_freq=40, picks=eeg_picks, **filter_kwargs)
        if raw.info['sfreq'] > 100:
            raw.notch_filter(freqs=50, picks=eeg_picks, **notch_kwargs)
    
    # ========== EOG 通道 ==========
    if len(eog_picks) > 0:
        raw.filter(l_freq=1, h_freq=20, picks=eog_picks, **filter_kwargs)
        if raw.info['sfreq'] > 100:
            raw.notch_filter(freqs=50, picks=eog_picks, **notch_kwargs)
    
    # ========== EMG 通道 ==========
    if len(emg_picks) > 0:
        nyquist = raw.info['sfreq'] / 2
        h_freq_emg = min(100, nyquist - 1)
        raw.filter(l_freq=15, h_freq=h_freq_emg, picks=emg_picks, **filter_kwargs)
        
        if 200 > raw.info['sfreq'] > 100:
            raw.notch_filter(freqs=50, picks=emg_picks, **notch_kwargs)
        if raw.info['sfreq'] > 200:
            # IIR 不支持同时陷波多个频率，拆分为两次调用
            raw.notch_filter(freqs=50, picks=emg_picks, **notch_kwargs)
            raw.notch_filter(freqs=100, picks=emg_picks, **notch_kwargs)
    
    # ========== ECG 通道 ==========
    if len(ecg_picks) > 0:
        raw.filter(l_freq=0.5, h_freq=35, picks=ecg_picks, **filter_kwargs)
    
    # 去趋势
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
        print("**************滤波,去趋势完成**************")
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
        
        # 检查每个部分是否在10-20系统电极列表中
        for part in parts:
            if part in self.standard_1020:
                return True
            
            # 模糊匹配
            # best_match, score = process.extractOne(part, self.standard_1020, scorer=fuzz.ratio)
            # if score >= self.similarity_threshold:
            #     return True
        
        # 检查是否包含EEG关键词
        if any('EEG' in part for part in parts):
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
        self.pre_samples = int(0.4 * fs)  # QRS波前200ms
        self.post_samples = int(0.4 * fs)  # QRS波后600ms
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

    def detect_bad_channels(self, raw,stim_analysis):
        """
        检测所有类型的坏通道
        
        参数:
            raw: mne.io.Raw 对象 (必须为预加载数据)
            
        返回:
            坏通道名称列表
        """
        if not raw.preload:
            raw.load_data()
        # if stim_analysis:
        #     print("获取非刺激部分数据检测bad_channels")
        #     raw,non_stim_mask,non_stim_mask_time = detect_stim_intervals(raw)
            
        # 检测平坦通道
        flat_bad = self.detect_flat_channels(raw)
        print(f"检测到平坦通道: {flat_bad}")
        
        # 检测高频噪声通道
        hf_noise_bad = self.detect_hf_noise_channels(raw)
        print(f"检测到高频噪声通道: {hf_noise_bad}")
        
        # 检测低相关性通道(含有刺激时相关性非常低)
        low_corr_bad = self.detect_low_correlation_channels(raw)
        print(f"检测到低相关性通道: {low_corr_bad}")
        
        # 合并所有坏通道
        all_bad = list(set(flat_bad + hf_noise_bad + low_corr_bad))
        print(f"总共检测到坏通道: {len(all_bad)}个")
        
        return all_bad

    def detect_flat_channels(self, raw, win_size=1.0):
        """
        检测平坦通道 - 检测持续平坦信号超过5秒的通道
        
        参数:
            raw: mne.io.Raw 对象
            win_size: 分析窗口大小（秒）
            
        返回:
            平坦通道名称列表
        """
        # print("获取非刺激部分数据检测bad_channels")
        # raw,non_stim_mask,non_stim_mask_time = detect_stim_intervals(raw)

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


def extract_stim_segments_from_raw(raw):
    """
    从raw对象中提取六个时间段的数据
    返回: [eeg_baseline1, eeg_tdcs_off1, eeg_baseline2, eeg_tdcs_off2, eeg_baseline3, eeg_tdcs_off3]
    """
    # 获取采样频率
    sfreq = raw.info['sfreq']
    
    # 复制原始代码中的get_duration_eeg逻辑，但修改为处理所有通道
    data = raw.get_data()  # 获取所有通道的数据
    n_channels, n_times = data.shape
    total_seconds = int(n_times / sfreq)  # 总秒数
    
    offset = 60
    
    # 找到第一个 "5000" 标记的时间点
    first_5000_time = None
    for ann in raw.annotations:
        desc = str(ann['description'])
        if '5000' in desc:
            first_5000_time = ann['onset']
            break

    if first_5000_time is None:
        raise ValueError("未找到 '5000' 标记")

    # 初始化列表
    stim_starts = []
    stim_ends = []
    start_marker_no_stim = []
    end_maker_no_stim = []

    # 只处理第一个 "5000" 标记之后的注释
    for ann in raw.annotations:
        if ann['onset'] < first_5000_time:
            continue
        
        desc = str(ann['description'])
        
        if 'Start of stimulation' in desc:
            stim_starts.append(ann['onset'])
        elif 'End of stimulation' in desc:
            stim_ends.append(ann['onset'])
        elif '5000' in desc:
            start_marker_no_stim.append(ann['onset'])
        elif '2001' in desc:
            end_maker_no_stim.append(ann['onset'])
    
    # 确保有足够的标记点
    if not stim_starts or not start_marker_no_stim or not end_maker_no_stim:
        raise ValueError("缺少必要的标记点")
    assert len(stim_starts) == len(stim_ends), "标记点数量不匹配"

    # 初始化区间列表
    baseline = []
    tdcs_off = []
    
    # 定义基线期
    baseline_start = (stim_starts[0] - offset - 60 * 4)
    baseline_end = (stim_starts[0] - offset)
    baseline.append([int(baseline_start), int(baseline_end)])
    
    # 判断刺激类型（sham 或 active）
    is_sham = stim_ends and (stim_ends[0] - stim_starts[0] < 2 * 60)
    
    if is_sham:  # sham刺激
        # 刺激间期
        for i in range(len(end_maker_no_stim) - 1):
            start_off = int((end_maker_no_stim[i] - 4 * 60))
            end_off = int(end_maker_no_stim[i])
            tdcs_off.append([start_off, end_off])
            baseline.append([end_maker_no_stim[i], int((stim_starts[i + 1]) - offset)])
        
        # 最后一个刺激间期
        start_off = int((end_maker_no_stim[-1] - 4 * 60))
        end_off = int(end_maker_no_stim[-1])
        tdcs_off.append([start_off, end_off])
    
    else:  # 真实刺激
        # 刺激间期
        for i in range(len(stim_ends) - 1):
            start_off = int((stim_ends[i] + offset))
            end_off = int(end_maker_no_stim[i])
            tdcs_off.append([start_off, end_off])
            baseline.append([end_maker_no_stim[i], int((stim_starts[i + 1]) - offset)])
        
        # 最后一个刺激间期
        start_off = int((stim_ends[-1] + offset))
        end_off = int(end_maker_no_stim[-1])
        tdcs_off.append([start_off, end_off])
    
    baseline = np.trunc(np.array(baseline)).astype(int) * int(sfreq)
    tdcs_off = np.trunc(np.array(tdcs_off)).astype(int) * int(sfreq)
    
    # 提取六个时间段的数据
    segments_data = []
    
    # baseline1
    segments_data.append(data[:, int(baseline[0][0]):int(baseline[0][1])])
    # tdcs_off1
    segments_data.append(data[:, int(tdcs_off[0][0]):int(tdcs_off[0][1])])
    # baseline2
    segments_data.append(data[:, int(baseline[1][0]):int(baseline[1][1])])
    # tdcs_off2
    segments_data.append(data[:, int(tdcs_off[1][0]):int(tdcs_off[1][1])])
    # baseline3
    segments_data.append(data[:, int(baseline[2][0]):int(baseline[2][1])])
    # tdcs_off3
    segments_data.append(data[:, int(tdcs_off[2][0]):int(tdcs_off[2][1])])
    
    return segments_data, sfreq



class preprocess_EEG:
    """
    class preprocess_EEG:预处理EEG数据
    """
    def __init__(self, file_paths, only_include_EEG_channels=True):

        if isinstance(file_paths, mne.io.BaseRaw):
            self.raw = file_paths
            first_raw = self.raw
            # 注意：此时需要确保 raw 已预加载，否则后续操作可能出错
            if not self.raw.preload:
                self.raw.load_data()
        else:
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
            elif type(file_paths) is str:
                self.file_path = file_paths
                if os.path.basename(self.file_path).endswith('.bdf'):
                    self.raw = mne.io.read_raw_bdf(self.file_path, preload=True)
                elif os.path.basename(self.file_path).endswith('.edf'):
                    self.raw = mne.io.read_raw_edf(self.file_path, preload=True)
                else:
                    raise ValueError('文件格式错误,请使用bdf或edf格式')
            else:
                #合并多个bdf文件
                self.raw = self.merge_bdf_files(file_paths)
        
        
        self.original_sfreq = self.raw.info['sfreq']
        self.orgin_channel_list = self.raw.info['ch_names']  #原始通道
        self.only_include_EEG_channels = only_include_EEG_channels
        print(f"总时长：{self.raw.times[-1]}s")
        # self.raw = self.raw.copy().crop(tmin=0, tmax=min(0+60*60*1, self.raw.times[-1]))  # 截取前1小时的数据
        # self.raw = self.raw.copy().crop(tmin=313, tmax=min(313+60*60*1, self.raw.times[-1]))  # 截取前1小时的数据
        # print(f"裁剪后时长：{self.raw.times[-1]}s")
        self.raw = channel_type_detection(self.raw)
         # EEG通道和其他通道分别滤波
        self.raw = filter_detrend(self.raw)
        #标准化EEG数据的通道信息,包括通道类型和参考电极
        self.raw = handle_bipolar_channels(self.raw)
        self.raw.load_data()
        self.get_non_eeg_channels()  # 获取非EEG通道,同时去除心电伪迹
        # self.resample(new_fs = 512)  #重采样
       
        # save_path = os.path.join(os.path.dirname(file_paths[0]),"resample_512.edf")
        # mne.export.export_raw(save_path, self.raw, fmt='edf', overwrite=True)

        self.raw = self.raw.copy().pick_types(eeg=True)
        self.eeg_channels = self.raw.info['ch_names']  #去除EOG和EMG后的EEG通道
         # #绘制EMG的时频图
        # self.plot_emg()

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
    
    def merge_channels(self, eeg_raw,non_eeg_raw):
        """合并多个通道的数据"""
        # 合并通道
        merge_raw=None
        if non_eeg_raw is not None:
            raw_info = mne.create_info(
                ch_names=eeg_raw.ch_names + non_eeg_raw.ch_names,
                sfreq= eeg_raw.info['sfreq'],
                ch_types=['eeg']*len(eeg_raw.ch_names) + non_eeg_raw.get_channel_types(),
            )
            min_len = min(eeg_raw.get_data().shape[1], non_eeg_raw.get_data().shape[1])

            merge_raw = mne.io.RawArray(
                np.vstack([eeg_raw.get_data()[:, :min_len], non_eeg_raw.get_data()[:,:min_len]]),
                raw_info
            )
            # 复制原始信息
                        
            if eeg_raw.info['meas_date'] is not None:
                merge_raw.set_meas_date(eeg_raw.info['meas_date'])
            else:
                # 获取当前 UTC 时间
                from datetime import datetime, timezone
                utc_now = datetime.now(timezone.utc)
                merge_raw.set_meas_date(utc_now)
                        
            merge_raw.info['bads'] = eeg_raw.info['bads'].copy()
            merge_raw.set_annotations(eeg_raw.annotations)
            del eeg_raw
            del non_eeg_raw
            return merge_raw
        else:
            del non_eeg_raw
            return eeg_raw
    
    def get_non_eeg_channels(self):
        """使用通道类型匹配方法获取非EEG通道数据"""
        # 获取所有通道类型
        channel_types = self.raw.get_channel_types()
        
        # 找出所有非EEG通道
        non_eeg_channels = [ch for i, ch in enumerate(self.raw.ch_names) 
                        if channel_types[i] != 'eeg']
        
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
                print(f"使用EMG通道: {emg_channels[0]}")
            else:
                self.emg = None
                print("未找到EMG通道")
                
            # 获取EOG通道数据
            eog_channels = [ch for i, ch in enumerate(self.no_eeg_raw.ch_names) 
                        if self.no_eeg_raw.get_channel_types()[i] == 'eog']
            
            if eog_channels:
                self.eog = self.no_eeg_raw.get_data(picks=eog_channels[0])[0, :]
                print(f"使用EOG通道: {eog_channels[0]}")
            else:
                self.eog = None
                
        else:
            self.no_eeg_raw = None
            self.ecg_reference = None
            self.emg = None
            self.eog = None
            print("未找到非EEG通道")

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
        self.no_eeg_raw.resample(new_fs, npad="auto")
        self.fs = new_fs
        print(f"******************重采样完成，新采样率：{self.fs}Hz**************")
  

    def apply_ica_to_non_stim(self, eeg_raw, ica_output_dir=None):
        """使用ICA去除伪迹"""
        logging.info("ICA去除伪迹...")
        #合并通道
        start_ica_time = time.time()
        if self.no_eeg_raw is not None:
            raw = mne.io.RawArray(
                np.vstack([eeg_raw.get_data(), self.no_eeg_raw.get_data()]),
                mne.create_info(
                    ch_names=eeg_raw.ch_names + self.no_eeg_raw.ch_names,
                    sfreq=self.raw.info['sfreq'],
                    ch_types=['eeg']*len(eeg_raw.ch_names) + self.no_eeg_raw.get_channel_types()
                )
            )
            # 复制原始信息
            if eeg_raw.info['meas_date'] is not None:
                raw.set_meas_date(eeg_raw.info['meas_date'])
            else:
                raw.set_meas_date(datetime.now())
            raw.info['bads'] = eeg_raw.info['bads'].copy()
            raw.set_annotations(eeg_raw.annotations)
              ###修复通道名称并加载标准蒙太奇
            montage = make_standard_montage("standard_1020")
            montage.ch_names = [name for name in montage.ch_names]
            # 应用标准蒙太奇前处理双极导联
            raw = handle_bipolar_channels(raw)
            raw.set_montage(montage, on_missing="warn")
        else:
            raw = eeg_raw

        # 1. ICA拟合
        if ica_output_dir is not None:
            after_ICA_path = os.path.join(ica_output_dir, "after_ica.edf")
            if os.path.exists(after_ICA_path):
                logging.info("ICA文件已存在，跳过ICA步骤")
                raw = mne.io.read_raw_edf(after_ICA_path, preload=True)
        else:
            # 创建ICA对象
            ica = ICA(
                n_components=None,          # 解释95%方差的成分
                method='infomax',           # infomax
                random_state=42,            # 随机种子保证结果可复现
                # max_iter=500                # 最大迭代次数
            )
            
            # ICA拟合
            ica.fit(raw)
            logging.info(f"ICA拟合完成，找到{ica.n_components_}个独立成分")
                
            # 3. 使用ICLabel自动标记成分
            print("使用ICLabel标记成分...")
            ic_labels = label_components(raw, ica, method="iclabel")
            labels = ic_labels['labels']
            probabilities = ic_labels['y_pred_proba']

            # 4. 标记伪迹成分
            artefact_threshold = 0.6  # 概率阈值
            artefact_components = []

            # 定义伪迹类型
            artefact_types = {
                'eye blink': [],
                'muscle': [],
                'heart': [],
                'channel noise': [],
                'line noise': []
            }

            for idx, (label, prob) in enumerate(zip(labels, probabilities)):
                # 只考虑高概率的伪迹
                if prob > artefact_threshold and label != 'brain':
                    artefact_components.append(idx)
                    
                    # 分类伪迹类型
                    if label == 'eye blink':
                        artefact_types['eye blink'].append(idx)
                    elif label == 'muscle artifact':
                        artefact_types['muscle'].append(idx)
                    elif label == 'heart beat':
                        artefact_types['heart'].append(idx)
                    elif label == 'channel noise':
                        artefact_types['channel noise'].append(idx)
                    elif label == 'line noise':
                        artefact_types['line noise'].append(idx)
           

            print(f"标记的伪迹成分: {artefact_components}")
            print("伪迹类型分布:")
            
            for artefact, comps in artefact_types.items():
                if comps:
                    logging.info(f"{artefact}: {comps}")
            # 5. 应用ICA去除伪迹
            raw_ica = raw.copy()
            ica.apply(raw_ica,exclude=artefact_components)
        
            logging.info("ICA去伪迹完成")
            #导出
            if ica_output_dir is not None:
                mne.export.export_raw(after_ICA_path, raw_ica, fmt='edf',overwrite=True)
            raw_ica_eeg = raw_ica.copy().pick_types(eeg=True)
            ica_time = time.time()-start_ica_time
            print(f"ica去噪用时: {time.time()-start_ica_time:.2f}秒")
            # raw_ica_non_eeg = raw_ica.copy().pick_types(emg=True)          
            return raw_ica_eeg,ica_time
        
    def apply_ica_to_non_stim_segment(self, eeg_raw, ica_output_dir=None, segment_length=600):
        """
        使用分段ICA去除伪迹
        
        参数:
            raw: 原始Raw对象
            ica_output_dir: ICA结果保存目录
            segment_length: 分段长度 (秒), 默认10分钟 (600秒)
        """
        logging.info("开始分段ICA去除伪迹...")
        #合并通道
        raw = mne.io.RawArray(
            np.vstack([eeg_raw.get_data(), self.no_eeg_raw.get_data()]),
            mne.create_info(
                ch_names=eeg_raw.ch_names + self.no_eeg_raw.ch_names,
                sfreq=self.raw.info['sfreq'],
                ch_types=['eeg']*len(eeg_raw.ch_names) + self.no_eeg_raw.get_channel_types()
            )
        )
        # 复制原始信息
        raw.info['bads'] = eeg_raw.info['bads'].copy()
        # raw.set_annotations(eeg_raw.annotations)

        # 修复通道名称并加载标准蒙太奇
        montage = make_standard_montage("standard_1020")
        montage.ch_names = [name for name in montage.ch_names]
        raw.set_montage(montage, on_missing="warn")

        # 1. 检查ICA文件是否已存在
        if ica_output_dir is not None:
            after_ICA_path = os.path.join(ica_output_dir, "after_ica.edf")
            if os.path.exists(after_ICA_path):
                logging.info("ICA文件已存在，跳过ICA步骤")
                return mne.io.read_raw_edf(after_ICA_path, preload=True)
        
        # 2. 准备分段处理
        sfreq = raw.info['sfreq']
        total_duration = raw.times[-1]
        n_segments = int(np.ceil(total_duration / segment_length))
        cleaned_segments = []
        
        logging.info(f"总时长: {total_duration:.1f}秒, 分成 {n_segments} 段 ({segment_length}秒/段)")
        
        # 3. 创建全局ICA模型 (使用部分数据训练)
        logging.info("训练全局ICA模型...")
        ica = ICA(
            n_components=None,          # 解释95%方差的成分
            method='fastica',           # infomax
            random_state=42,            # 随机种子保证结果可复现
            max_iter=1000                # 最大迭代次数
        )
        
        # 使用前10分钟数据训练ICA模型
        train_duration = min(600, total_duration)
        train_raw = raw.copy().crop(tmin=0, tmax=total_duration)
        ica.fit(train_raw)
        logging.info(f"ICA模型训练完成，找到{ica.n_components_}个独立成分")
        
        # 4. 分段处理
        for seg_idx in range(n_segments):
            start_time = seg_idx * segment_length
            end_time = min((seg_idx + 1) * segment_length, total_duration)
            seg_duration = end_time - start_time
            
            logging.info(f"处理分段 {seg_idx+1}/{n_segments}: {start_time:.1f}-{end_time:.1f}秒 ({seg_duration:.1f}秒)")
            
            # 提取当前分段
            if seg_idx == n_segments - 1:
                seg_raw = raw.copy().crop(tmin=start_time, tmax=end_time,include_tmax=True)
            else:
                seg_raw = raw.copy().crop(tmin=start_time, tmax=end_time,include_tmax=False)
            
            # ica.fit(seg_raw)
            # logging.info(f"ICA模型训练完成，找到{ica.n_components_}个独立成分")
            # 应用全局ICA模型
            seg_ica = ica.copy()
            
            # 使用ICLabel自动标记成分
            ic_labels = label_components(seg_raw, seg_ica, method="iclabel")
            labels = ic_labels['labels']
            probabilities = ic_labels['y_pred_proba']

            # 标记伪迹成分
            artefact_threshold = 0.7  # 概率阈值
            artefact_components = []

            # 定义伪迹类型
            artefact_types = {
                'eye blink': [],
                'muscle': [],
                'heart': [],
                'channel noise': [],
                'line noise': []
            }

            for idx, (label, prob) in enumerate(zip(labels, probabilities)):
                # 只考虑高概率的伪迹
                if prob > artefact_threshold and label != 'brain':
                    artefact_components.append(idx)
                    
                    # 分类伪迹类型
                    if label == 'eye blink':
                        artefact_types['eye blink'].append(idx)
                    elif label == 'muscle artifact':
                        artefact_types['muscle'].append(idx)
                    elif label == 'heart beat':
                        artefact_types['heart'].append(idx)
                    elif label == 'channel noise':
                        artefact_types['channel noise'].append(idx)
                    elif label == 'line noise':
                        artefact_types['line noise'].append(idx)
            
            logging.info(f"分段 {seg_idx+1} 标记的伪迹成分: {artefact_components}")
            logging.info(f"分段 {seg_idx+1} 伪迹类型分布:")
            for artefact, comps in artefact_types.items():
                if comps:
                    logging.info(f"{artefact}: {comps}")
            
            # 应用ICA去除伪迹
            seg_clean = seg_raw.copy()
            seg_ica.apply(seg_clean, exclude=artefact_components)
            
            # 保存处理后的分段
            cleaned_segments.append(seg_clean)
        
        # 5. 合并所有分段
        if cleaned_segments:
            raw_ica = mne.concatenate_raws(cleaned_segments)
            logging.info(f"所有分段合并完成，总时长: {raw_ica.times[-1]:.1f}秒")
        else:
            logging.warning("没有处理任何分段，返回原始数据")
            raw_ica = raw
        
        if raw_ica._data.shape != raw._data.shape:
            logging.warning("处理后的数据形状与原始数据不同，请检查处理过程")
        else:
            logging.info(f"处理后的数据形状与原始数据相同:{raw_ica._data.shape}")

    
        # 6. 保存结果
        if ica_output_dir is not None:
            os.makedirs(ica_output_dir, exist_ok=True)
            after_ICA_path = os.path.join(ica_output_dir, "after_ica.edf")
            # mne.export.export_raw(after_ICA_path, raw_ica, fmt='edf', overwrite=True)
            # logging.info(f"ICA处理后的数据已保存到: {after_ICA_path}")
        raw_ica_eeg = raw_ica.copy().pick_types(eeg=True)
        return raw_ica_eeg

    def apply_ica_to_stim(self,eeg_raw, ):
        """
        仅对非刺激部分应用 ICA，刺激部分保持不变mask
        
        参数:
            raw: mne.io.Raw 对象
            start_prefix: 刺激开始标记前缀
            end_prefix: 刺激结束标记前缀
            
        返回:
            处理后的 Raw 对象（非刺激部分应用ICA，刺激部分保持不变）
        """
        #合并通道
        if self.no_eeg_raw is not None:
            raw_info = mne.create_info(
                ch_names=eeg_raw.ch_names + self.no_eeg_raw.ch_names,
                sfreq= eeg_raw.info['sfreq'],
                ch_types=['eeg']*len(eeg_raw.ch_names) + self.no_eeg_raw.get_channel_types(),
            )
            raw = mne.io.RawArray(
                np.vstack([eeg_raw.get_data(), self.no_eeg_raw.get_data()]),
                raw_info
            )
            if eeg_raw.info['meas_date'] is not None:
                raw.set_meas_date(eeg_raw.info['meas_date'])
            else:
                raw.set_meas_date(datetime.now())
                        
            raw.info['bads'] = eeg_raw.info['bads'].copy()
            raw.set_annotations(eeg_raw.annotations)
            
            # 修复通道名称并加载标准蒙太奇
            montage = make_standard_montage("standard_1020")
            montage.ch_names = [name for name in montage.ch_names]
            raw.set_montage(montage, on_missing="warn")
        else:
            raw = eeg_raw
        
        raw = raw.copy()
        original_annots = raw.annotations.copy()
        if not raw.preload:
            raw.load_data()
        
        #获取非刺激部分数据
        non_stim_raw,non_stim_mask,non_stim_mask_time = detect_stim_intervals(raw)
        
        # 4. 拟合 ICA（仅使用非刺激数据）
        ica = ICA(n_components=None, random_state=42,method="fastica") #infomax
        ica.fit(non_stim_raw)
        
        # 5. 打印非刺激部分混合矩阵
        # print("非刺激部分混合矩阵:")
        # self.print_mixing_matrix(ica, raw.ch_names)
        
        # 6. 自动检测伪迹（仅使用非刺激数据）
        print("使用ICLabel标记成分...")
        ic_labels = label_components(non_stim_raw, ica,method="iclabel")
        labels = ic_labels['labels']
        probabilities = ic_labels['y_pred_proba']

        # 标记伪迹成分
        print("标记伪迹成分...")
        artefact_threshold = 0.7  # 概率阈值
        artefact_components = []
        
        for idx, (label, prob) in enumerate(zip(labels, probabilities)):
            # 只考虑高概率的伪迹
            if prob > artefact_threshold and label != 'brain':
                artefact_components.append(idx)
     
        print(f"标记的伪迹成分: {artefact_components}")
        
        # 7. 仅对非刺激部分应用ICA
        denoised_non_stim = ica.apply(non_stim_raw.copy(), exclude=artefact_components)
        
        # 8. 重建完整数据
        denoised_data = raw.get_data().copy()  # 创建原始数据的副本
        
        # 8.1 非刺激部分应用ICA结果
        denoised_data[:, non_stim_mask] = denoised_non_stim.get_data() #(18, 3948000)
        
        # 8.2 刺激部分保持不变（直接使用原始数据）
        # 注意：non_stim_mask为False的部分（即刺激部分）已经是原始数据
        
        # 9. 创建新的Raw对象
        denoised_raw = mne.io.RawArray(denoised_data, raw.info)
        # denoised_raw.annotations = raw.annotations.copy()
            
        # 5. 恢复原始注释
        denoised_raw.set_annotations(original_annots)
        raw_ica_eeg = denoised_raw.copy().pick_types(eeg=True)
        return raw_ica_eeg

    def merge_bdf_files(self, file_paths):
        """
        按顺序拼接多个BDF文件，调整Annotations为绝对时间
        
        参数:
            file_paths (list): 按顺序排列的BDF文件路径列表
            output_path (str): 输出拼接后BDF文件的路径
            
        返回:
            combined_raw: 拼接后的Raw对象
        """
        if len(file_paths) < 1:
            raise ValueError("至少需要一个文件进行拼接")
        
        # 初始化变量
        all_raws = []        # 存储所有Raw对象
        all_annotations = []  # 存储所有调整后的Annotations
        cumulative_onset = 0   # 累积时间偏移量（秒）
        
        for i, file_path in enumerate(file_paths):
            # 1. 读取当前BDF文件
            raw = mne.io.read_raw_bdf(file_path, preload=True)
            print(f"处理文件 {i+1}/{len(file_paths)}: {file_path}")
            print(f"  原始持续时间: {raw.times[-1]:.3f}秒, 采样频率: {raw.info['sfreq']:.1f}Hz")
            # print(raw.get_data().shape)
            # 2. 获取原始Annotations
            orig_annot = raw.annotations
            
            # 3. 如果是第一个文件，保留所有原始Annotations
            if i == 0:
                # 存储原始Raw和Annotations
                all_raws.append(raw)
                all_annotations.append(orig_annot)
                
                # 记录第一个文件的总时间作为偏移基准
                cumulative_onset = raw.times[-1]
                logging.info(f"首个文件结束时间: {cumulative_onset:.3f}秒")
                continue
            
            # 4. 后续文件处理
            # 4.1 计算这个文件的持续时间（秒）
            current_duration = raw.times[-1]
            
            # 4.2 如果文件有Annotations，调整onset时间为绝对时间
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
                print(f"调整 {len(modified_annot)} 个Annotations, 第一个事件时间: {new_onset[0]:.3f}s → {modified_annot.onset[0]:.3f}s")
            else:
                print("没有Annotations需要调整")
                modified_annot = orig_annot
            
            # 4.3 存储调整后的Annotations
            all_annotations.append(modified_annot)
            
            # 4.4 更新当前Raw对象的时间向量
            # 注意: BDF文件的时间信息存储在数据中，不需要直接修改
            # 只需要更新后续的时间偏移量
            # 5. 存储当前Raw对象
            all_raws.append(raw)
            
            # 6. 更新累积时间偏移量（加上当前文件持续时间）
            cumulative_onset += current_duration
            print(f"累积时间偏移量: {cumulative_onset:.3f}秒")
        
        # 7. 将所有Raw文件拼接成一个
        if len(all_raws) > 1:
            combined_raw = mne.concatenate_raws(all_raws, preload=False)
            logging.info(f"拼接完成，总持续时间: {combined_raw.times[-1]:.3f}秒")
        else:
            combined_raw = all_raws[0]
        
        return combined_raw

    def faster_remove_bad_epoch(self, raw, epoch_duration=1.0, max_bad_frac=0.2):
        """
        优化版的FASTER处理
        """
        # 1. 保存原始注释
        original_annotations = raw.annotations.copy() if raw.annotations is not None else None
        sfreq = raw.info['sfreq']
        
        # 2. 创建固定长度epoch
        n_times_per_epoch = int(epoch_duration * sfreq)
        events = mne.make_fixed_length_events(raw, duration=epoch_duration, overlap=0.0)
        
        # 3. 创建Epochs对象
        epochs = mne.Epochs(
            raw, 
            events, 
            tmin=0, 
            tmax=epoch_duration - 1/sfreq,
            baseline=None, 
            preload=True,
            reject=None, 
            flat=None
        )
        
        # 4. 获取epoch数据
        epoch_data = epochs.get_data()
        n_epochs, n_channels, n_times = epoch_data.shape
        ch_names = epochs.ch_names  # 获取通道名称列表
        
        # 5. 使用MNE-FASTER检测
        bad_channels_per_epoch = find_bad_channels_in_epochs(epochs, thres=2)
        bad_epochs = find_bad_epochs(epochs, thres=2)
        
        # 6. 计算全局坏通道 
        # 创建通道名称到索引的映射
        ch_name_to_idx = {name: idx for idx, name in enumerate(ch_names)}
        
        # 初始化通道计数数组
        channel_counts = np.zeros(n_channels)
        
        # 统计每个通道被标记为坏的次数
        for epoch_bads in bad_channels_per_epoch:
            for chan_name in epoch_bads:
                # 将通道名称转换为索引
                if chan_name in ch_name_to_idx:
                    chan_idx = ch_name_to_idx[chan_name]
                    channel_counts[chan_idx] += 1
        
        # 计算全局坏通道（超过30%的epoch中被标记为坏）
        global_bad_channels = [
            ch_names[i] for i in range(n_channels) 
            if channel_counts[i] / n_epochs > 0.3
        ]
        
        print(f"FASTER全局坏通道: {global_bad_channels}")
        
        # 7. 预分配内存并批量处理
        cleaned_data = np.zeros_like(epoch_data)
        
        # 创建掩码标识需要处理的epoch
        process_mask = np.zeros(n_epochs, dtype=bool)
        for epoch_idx in range(n_epochs):
            if epoch_idx in bad_epochs:
                # 计算坏通道比例
                bad_channels_count = len(bad_channels_per_epoch[epoch_idx])
                bad_channels_ratio = bad_channels_count / n_channels
                process_mask[epoch_idx] = bad_channels_ratio > max_bad_frac
        
        # 批量处理需要插值的epoch
        for epoch_idx in np.where(process_mask)[0]:
            epoch = epochs[epoch_idx]
            
            # 合并全局坏通道和当前epoch的坏通道
            all_bad_channels = list(set(global_bad_channels + bad_channels_per_epoch[epoch_idx]))
            epoch.info['bads'] = all_bad_channels
            
            # 检查坏通道比例是否过高
            if len(all_bad_channels) > 0.3 * n_channels:
                # print(f"epoch{epoch_idx}: 坏channel比例大于阈值重建{all_bad_channels}_n_channels{n_channels}")
                epoch.interpolate_bads()
            
            cleaned_data[epoch_idx] = epoch.get_data()[0]
        
        # 复制不需要处理的epoch
        not_process_mask = ~process_mask
        cleaned_data[not_process_mask] = epoch_data[not_process_mask]
        
        # 8. 重建连续数据
        clean_data = cleaned_data.transpose(1, 0, 2).reshape(n_channels, -1)
        
        # 9. 确保数据长度一致
        if clean_data.shape[1] != raw.n_times:
            # 使用更精确的长度调整方法
            clean_data_adjusted = np.zeros((n_channels, raw.n_times))
            min_length = min(clean_data.shape[1], raw.n_times)
            clean_data_adjusted[:, :min_length] = clean_data[:, :min_length]
            clean_data = clean_data_adjusted
        
        # 10. 创建Raw对象
        clean_raw = mne.io.RawArray(clean_data, raw.info)
        
        # 11. 恢复原始注释
        if original_annotations is not None:
            clean_raw.set_annotations(original_annotations)
        
        return clean_raw, global_bad_channels
    
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
    
    def process_main(self, useICA=False, useFASTER=True,remove_bad_channels=True,stim_analysis=False):
        #处理bdf文件入口
        logging.info(f"EEG通道{self.raw.info['ch_names']}")
        if self.no_eeg_raw:
            logging.info(f"非EEG通道{self.no_eeg_raw.info['ch_names']}")  #25831500               
        else:
            logging.info("无非EEG通道")
        # self.raw = self.raw.crop(tmin=0, tmax=191)
        bad_channels,global_bad_channels = [],[]
        ica_time =0
        if remove_bad_channels:
            # 创建CleanRawData实例
            cleaner = CleanRawData()
            # 参数设置
            cleaner.flat_duration = 5.0     # 平坦信号持续时间阈值（秒）
            cleaner.flat_threshold = 1e-6   # 平坦信号幅度变化阈值
            cleaner.noise_threshold = 4     # 高频噪声标准差阈值
            cleaner.corr_threshold = 0.5 # 最小相关阈值 （）
            cleaner.n_neighbors = 3         # 邻近通道数量
            # 检测坏通道
            bad_channels = cleaner.detect_bad_channels(self.raw,stim_analysis)
           
            # 2. 如果有坏通道，先移除并处理
        if bad_channels and remove_bad_channels:
            logging.info(f"CleanRawData检测到坏通道: {bad_channels}, 先移除进行后续处理")
            
            # 保存原始坏通道数据 (用于后续恢复)
            bad_channel_data = self.raw.copy().pick(bad_channels).get_data()# (3, 8146000)
           
            # 创建只包含好通道的副本
            good_channels = [ch for ch in self.raw.ch_names if ch not in bad_channels]
            raw_good = self.raw.copy().pick(good_channels)

            if useICA:
                logging.info("非刺激数据上应用ICA")
                raw_good,ica_time = self.apply_ica_to_non_stim(raw_good)

            if useFASTER:
                raw_good, global_bad_channels = self.faster_remove_bad_epoch(raw_good,epoch_duration=1)
                logging.info("非刺激make数据上应用FASTER")
            
            # 3. 准备恢复坏通道
            # 创建完整数据数组
            
            full_data = np.zeros((len(self.raw.ch_names), raw_good._data.shape[1]))
            
            # 填充好通道数据
            good_idx = [self.raw.ch_names.index(ch) for ch in good_channels]
            full_data[good_idx] = raw_good.get_data()
            
            # 填充坏通道数据 (使用原始数据)
            bad_idx = [self.raw.ch_names.index(ch) for ch in bad_channels]
           
            full_data[bad_idx] = bad_channel_data[:,:raw_good._data.shape[1]]
            
            # 更新原始raw对象
            self.raw._data = full_data
        
        else:
            # 没有坏通道或不需要移除，直接处理
            if useICA:
                self.raw,ica_time= self.apply_ica_to_non_stim(self.raw)
            if useFASTER:
                self.raw, global_bad_channels = self.faster_remove_bad_epoch(self.raw,epoch_duration=1) #

        # 4. 合并所有需要插值的坏通道
        repaired_bad_channels = list(set(global_bad_channels + bad_channels))
        if len(repaired_bad_channels)!=0:
            # 标记坏通道并插值
            self.raw.info['bads'] = repaired_bad_channels
            self.raw.interpolate_bads(method='spline',reset_bads=True) #'spline', 'MNE', and 'nan'
            logging.info(f"完成坏通道的插值重建: {repaired_bad_channels}")
        
        # #再次检测验证
        # check_bad_channels = self.happe_bad_channel_detection(self.raw)
        # logging.info(f"happe验证是否有剩余坏通道: {check_bad_channels}")
       
        return self.raw,self.no_eeg_raw

    def DeepDenoiser(self):
        model_path = r"D:\DENOISE\EEGDiR-master\results\2025_09_29_18_DiR_layer_4_head_8_mini_seq32_hidden_dim512_win_EOG\weight\best.pth"
         #2 D:\DENOISE\EEGDiR-master\results\2025_09_29_15_DiR_layer_4_head_8_mini_seq32_hidden_dim512_EOG\weight\best.pth
        #0916 "D:\DENOISE\EEGDiR-master\results\2025_09_29_14_DiR_layer_4_head_8_mini_seq32_hidden_dim512_EOG\weight\best.pth"
        output_base_dir = r"D:\DENOISE\EEGDiR-master\results\predict"
        # 创建去噪器实例
        denoiser = DeepEEGDenoiser(model_path,output_base_dir)
        # 快速去噪流程
        denoiser.quick_denoise(self.raw,self.no_eeg_raw,self.original_sfreq)
        print(f"完成去噪")

if __name__ == '__main__':
    # 读取文件路径
    # file_paths = [r"D:\DENOISE\EEGDnet\monkey_bdf\test\20250319173202_2_split_14400.bdf"]
    file_paths = [r"D:\DENOISE\EEGDnet\monkey_bdf\test\BMAL1\20250818174650_1.bdf",
                  r"D:\DENOISE\EEGDnet\monkey_bdf\test\BMAL1\20250818174650_2.bdf",
                  r"D:\DENOISE\EEGDnet\monkey_bdf\test\BMAL1\20250818174650_3.bdf"]
    
    # output_path = r"D:\analysis\long_stim_bdf\test_preprocess\out_info"
    # 合并BDF文件
    eeg = preprocess_EEG(file_paths)   
    eeg.DeepDenoiser()
    # mne.export.export_raw(r"D:\analysis\long_stim_bdf\test_preprocess\20250617222048_1_edited_processed.edf",clean_raw, overwrite=True, fmt='edf')
    # print(clean_raw.info)
