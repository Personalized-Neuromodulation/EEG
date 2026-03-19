import os
import numpy as np
import mne
import datetime
from scipy.io import loadmat
import h5py
from multiprocessing import Pool
from tqdm import tqdm

def parse_hea_file(hea_path):
    """解析.hea头文件"""
    with open(hea_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 解析第一行: tr03-0005 13 200 5147000
    first_line = lines[0].split()
    record_name = first_line[0]
    n_channels = int(first_line[1])
    fs = float(first_line[2])  # 200 Hz
    n_samples = int(first_line[3])  # 5147000
    
    # 解析通道信息
    ch_names = []
    ch_gains = []
    
    for i in range(1, min(len(lines), n_channels + 1)):
        parts = lines[i].split()
        # 格式: tr03-0005.mat 16+24 1/uV 16 0 -9 139 0 F3-M2
        if len(parts) >= 9:
            # 通道名称在最后一个字段
            ch_name = parts[8]
            ch_names.append(ch_name)
            
            # 增益在第三个字段 (1/uV)，提取数字部分
            gain_str = parts[2].split('/')[0]
            try:
                ch_gains.append(float(gain_str))
            except:
                ch_gains.append(1.0)
        else:
            # 如果格式不完整，使用默认值
            ch_names.append(f'EEG{i}')
            ch_gains.append(1.0)
    
    # 确保通道数量正确
    while len(ch_names) < n_channels:
        ch_names.append(f'EEG{len(ch_names)+1}')
        ch_gains.append(1.0)
    
    return {
        'fs': fs,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'ch_names': ch_names,
        'ch_gains': ch_gains
    }

def load_mat_signals(mat_path, n_channels):
    """加载.mat信号数据"""
    mat_data = loadmat(mat_path)
    
    # 查找信号数据
    for key in mat_data:
        if not key.startswith('__'):
            value = mat_data[key]
            if isinstance(value, np.ndarray) and value.size > 1:
                signals = value
                break
    
    # 确保形状为 (n_channels, n_samples)
    if signals.shape[0] == n_channels:
        return signals
    elif signals.shape[1] == n_channels:
        return signals.T
    else:
        # 自动调整形状
        if signals.shape[0] < signals.shape[1] and signals.shape[0] <= n_channels:
            return signals
        else:
            return signals.T

def load_sleep_stages_hdf5(arousal_mat_path, fs, n_samples):
    """从HDF5格式的-arousal.mat文件加载睡眠阶段注释"""
    if not os.path.exists(arousal_mat_path):
        return []
    
    try:
        with h5py.File(arousal_mat_path, 'r') as f:
            print(f"  文件结构: {list(f.keys())}")
            
            # 查找data组
            if 'data' in f:
                data_group = f['data']
                print(f"  data组内项目: {list(data_group.keys())}")
                
                # 处理sleep_stages组
                if 'sleep_stages' in data_group:
                    sleep_stages_group = data_group['sleep_stages']
                    stage_names = list(sleep_stages_group.keys())
                    print(f"  sleep_stages组内成员: {stage_names}")
                    
                    # 收集所有注释
                    all_annotations = []
                    
                    # 处理每个睡眠阶段
                    for stage_name in stage_names:
                        stage_dataset = sleep_stages_group[stage_name]
                        
                        if isinstance(stage_dataset, h5py.Dataset):
                            # 读取数据
                            stage_data = stage_dataset[()].flatten()
                            print(f"    读取 {stage_name}，形状: {stage_data.shape}")
                            
                            # 将二值标记转换为注释
                            stage_annotations = binary_to_annotations(
                                stage_data, stage_name, fs, n_samples
                            )
                            all_annotations.extend(stage_annotations)
                    
                    print(f"  总共生成 {len(all_annotations)} 个睡眠阶段注释")
                    
                    # 按起始时间排序
                    all_annotations.sort(key=lambda x: x['onset'])
                    
                    return all_annotations
    
    except Exception as e:
        print(f"  读取HDF5文件失败: {e}")
    
    return []

def binary_to_annotations(binary_vector, stage_name, fs, n_samples):
    """将二值向量转换为注释列表"""
    annotations = []
    
    # 确保向量长度正确
    binary_vector = binary_vector[:n_samples]
    
    # 找到所有为1的连续段
    current_state = 0
    start_idx = 0
    
    for i in range(len(binary_vector)):
        value = int(binary_vector[i])
        
        if value != current_state:
            if current_state == 1:  # 结束一个阶段
                end_idx = i
                duration = (end_idx - start_idx) / fs
                onset = start_idx / fs
                
                annotations.append({
                    'onset': onset,
                    'duration': duration,
                    'description': f"Sleep_{stage_name}"
                })
            
            current_state = value
            start_idx = i
    
    # 处理最后一个阶段
    if current_state == 1:
        end_idx = len(binary_vector)
        duration = (end_idx - start_idx) / fs
        onset = start_idx / fs
        
        annotations.append({
            'onset': onset,
            'duration': duration,
            'description': f"Sleep_{stage_name}"
        })
    
    return annotations

def convert_to_mne_raw(signals, header_info, annotations):
    """转换为MNE Raw对象"""
    n_channels = signals.shape[0]
    
    # 应用增益
    for i in range(min(n_channels, len(header_info['ch_gains']))):
        if header_info['ch_gains'][i] != 1.0:
            signals[i] = signals[i] / header_info['ch_gains'][i]
    
    # 确定通道类型
    ch_types = ['eeg'] * n_channels  # 默认为EEG
    
    # 创建MNE Info对象
    info = mne.create_info(
        ch_names=header_info['ch_names'][:n_channels],
        sfreq=header_info['fs'],
        ch_types=ch_types
    )
    
    # 创建Raw对象 - 不设置测量日期，避免UTC问题
    raw = mne.io.RawArray(signals, info)
    
    # 添加注释
    if annotations:
        mne_annotations = mne.Annotations(
            onset=[a['onset'] for a in annotations],
            duration=[a['duration'] for a in annotations],
            description=[a['description'] for a in annotations]
        )
        raw.set_annotations(mne_annotations)
        
        # 统计各阶段时长
        stage_stats = {}
        for ann in annotations:
            stage = ann['description']
            stage_stats[stage] = stage_stats.get(stage, 0) + ann['duration']
        
        print(f"  睡眠阶段统计:")
        for stage, duration in stage_stats.items():
            print(f"    {stage}: {duration:.1f}秒 ({duration/60:.1f}分钟)")
    
    return raw

import numpy as np
import pyedflib
from mne.io import Raw

def raw_to_edf(raw: Raw, edf_fname: str):
    """
    将 MNE Raw 对象保存为 EDF 文件
    """
    # 获取数据（通道数 × 时间点数）
    data, times = raw[:, :]
    n_channels, n_samples = data.shape
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    ch_types = raw.get_channel_types()

    # 创建 EDF 文件
    with pyedflib.EdfWriter(edf_fname, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        # 设置通道信息
        channel_info = []
        for i in range(n_channels):
            ch_dict = {
                'label': ch_names[i],
                'dimension': 'uV',  # 单位，可根据数据调整
                'sample_frequency': sfreq,
                'physical_min': np.min(data[i]),
                'physical_max': np.max(data[i]),
                'digital_min': -32768,
                'digital_max': 32767,
                'transducer': '',
                'prefilter': ''
            }
            channel_info.append(ch_dict)
        f.setSignalHeaders(channel_info)

        # 写入数据（注意 EDF 要求数据按通道组织，且为 int16 类型）
        # 将数据缩放到 digital_min/digital_max 范围内
        data_int = []
        for i in range(n_channels):
            phys_min = channel_info[i]['physical_min']
            phys_max = channel_info[i]['physical_max']
            dig_min = channel_info[i]['digital_min']
            dig_max = channel_info[i]['digital_max']
            # 线性缩放
            scaled = (data[i] - phys_min) / (phys_max - phys_min) * (dig_max - dig_min) + dig_min
            scaled = np.clip(scaled, dig_min, dig_max).astype(np.int16)
            data_int.append(scaled)
        data_int = np.array(data_int)

        f.writeSamples(data_int)

def process_sleep_record(folder_path, output_dir):
    """处理单个睡眠记录"""
    # 获取文件夹名作为记录名
    record_name = os.path.basename(folder_path)
    file = os.path.join(output_dir,f"{record_name}.fif")
    if os.path.exists(file):
        print(f"跳过 {record_name}: 文件已存在")
        return None
    
    # 定义文件路径
    base_path = os.path.join(folder_path, record_name)
    hea_path = base_path +".hea"
    mat_path = base_path + '.mat'

    
    print(f"处理: {record_name}")
    

    # 1. 解析头文件
    header_info = parse_hea_file(hea_path)
    print(f"  采样频率: {header_info['fs']} Hz")
    print(f"  通道数: {header_info['n_channels']}")
    print(f"  样本数: {header_info['n_samples']}")
    print(f"  通道名称: {header_info['ch_names']}...")  # 只显示前5个
    
    # 2. 加载信号数据
    signals = load_mat_signals(mat_path, header_info['n_channels'])
    print(f"  信号形状: {signals.shape}")
    
    # 3. 调整信号维度以匹配头文件信息
    if signals.shape[0] != header_info['n_channels']:
        print(f"  调整信号维度...")
        if signals.shape[1] == header_info['n_channels']:
            signals = signals.T
    
    # 4. 加载睡眠阶段注释
    annotations = None #load_sleep_stages_hdf5(arousal_mat_path, header_info['fs'], header_info['n_samples'])
    
    # 5. 转换为MNE Raw对象
    raw = convert_to_mne_raw(signals, header_info, annotations)
    
    # 6. 保存为EDF
    output_path = os.path.join(output_dir, f"{record_name}.fif")
    raw.save(output_path, overwrite=True)
  
    # raw_to_edf(raw,output_path)

    print(f"  已保存: {output_path}")
    return output_path

def process_sleep_record_wrapper(args):
    """包装函数，用于进程池"""
    folder_path, output_dir = args
    return process_sleep_record(folder_path, output_dir)

def process_all_records(input_dir, output_dir):
    """使用进程池处理所有记录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 收集所有要处理的文件夹路径
    folders = []
    for item in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, item)
        if os.path.isdir(folder_path):
            folders.append(folder_path)
    
    total = len(folders)
    success = 0
    
    # 使用进程池，可根据CPU核心数调整进程数量
    with Pool(processes=8) as pool:
        # 准备参数列表
        args_list = [(folder, output_dir) for folder in folders]
        # 使用imap_unordered并行处理，并用tqdm显示进度
        for result in tqdm(pool.imap_unordered(process_sleep_record_wrapper, args_list), total=total):
            if result is not None:
                success += 1
    
    print(f"\n处理完成: {success}/{total} 个记录成功")


if __name__ == "__main__":
    input_dir = r"/mnt/nas2/DATA/Sleep_Datasets/phys/training"
    output_dir = r"/mnt/nas2/DATA/Sleep_Datasets/phys/fif"
    
    process_all_records(input_dir, output_dir)
    
    print("处理完成!")