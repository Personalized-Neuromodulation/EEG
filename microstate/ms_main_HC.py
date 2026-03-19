import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from ms_base import Microstate
from natsort import natsorted
from scipy.signal import spectrogram, butter, filtfilt
from matplotlib.ticker import ScalarFormatter
import datetime

# 导入预处理
import Microstate_analysis.monkey_preprocess_ms as monkey_preprocess_ms

# ------------------ 配置参数 ------------------
CONFIG = {
    'input_dir': r"\\172.16.6.5\project\Monkey\HC_Monkey\source_data\01_Hanhan_171321\freemoving\2026-3-10",
    'output_dir': None,  # 如果为None，则在每个输入文件夹内创建结果子文件夹
    'fs': None,          # 采样率，None表示从数据自动获取
    'do_microstate': False,      # 是否进行微状态分析（默认False）
    'plot_long_duration': True,  # 是否绘制长时图（10小时）
    'plot_short_duration': False, # 是否绘制短时图（如30秒）
    'short_duration_sec': 30,    # 短时图时长（秒）
    'short_start_offset': 4*3600 + 11*60 + 15,  # 短时图起始偏移（秒）
    'beijing_start_hour': 20,    # 长时图的时间轴起始小时
    'filter_eeg_low': 1,          # EEG带通低截止
    'filter_eeg_high': 40,        # EEG带通高截止
    'filter_emg_low': 10,
    'filter_emg_high': 200,
    'filter_eog_low': 0.1,
    'filter_eog_high': 15,
    'filter_order': 2,
}
# ---------------------------------------------

def preprocess_bdf(file_path):
    """预处理BDF/EDF文件，返回去噪后的raw对象和通道名"""
    save_path = os.path.dirname(file_path[0])
    eeg = monkey_preprocess_ms.preprocess_EEG(file_path)
    src_raw = eeg.raw
    src_raw.load_data()
    # 保存原始预处理数据（可选）
    # mne.export.export_raw(os.path.join(save_path, 'preprocessed_data_RAW.edf'), src_raw, fmt='edf', overwrite=True)
    eeg_raw, non_eeg_raw, ica_time = eeg.process_main(useICA=False, useFASTER=True,
                                                       remove_bad_channels=False, stim_analysis=False)
    denoised_raw = eeg.merge_channels(eeg_raw, non_eeg_raw)
    denoised_raw.load_data()
    # mne.export.export_raw(os.path.join(save_path, 'preprocessed_data.edf'), denoised_raw, fmt='edf', overwrite=True)
    return denoised_raw, eeg_raw.info['ch_names']

def check_non_eeg_raw(raw):
    """设置非EEG通道类型"""
    if "EMG" in raw.ch_names:
        raw.set_channel_types({"EMG": "emg"})
    if "ECG" in raw.ch_names:
        raw.set_channel_types({"ECG": "ecg"})
    import re
    pattern = re.compile(r'^EOG[-_]?[A-Za-z0-9]*$', re.IGNORECASE)
    for ch_name in raw.ch_names:
        if pattern.match(ch_name):
            raw.set_channel_types({ch_name: "eog"})
    return raw

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """巴特沃斯带通滤波器"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def analyze_microstates(data, n_maps=4, res_path=None):
    """微状态分析（需要ms_base模块）"""
    ms = Microstate(data.T)
    maps, cvs = ms.microstate(n_maps, n_repetition=8)
    # 保存和加载等操作可根据需要添加
    min_cv_index = np.argmin(cvs)
    maps = np.asarray(maps[min_cv_index])
    seq = ms.fit_back_peaks(maps)
    lzma = ms.create_lempel_ziv_markov_chain(seq, epoch_time=24)
    return lzma

def plot_long_results(folder_path, eeg_data, channel_names, emg_data, eog_data,
                      fs, total_hours=10, beijing_start=20):
    """绘制长时图（包含EEG、EMG、EOG和时频图）"""
    folder_name = os.path.basename(folder_path)
    hours_samples = int(fs * 3600 * total_hours)
    eeg_data = eeg_data[:, :hours_samples]
    if emg_data is not None:
        emg_data = emg_data[:hours_samples]
    if eog_data is not None:
        eog_data = eog_data[:hours_samples]

    plt.rcParams.update({'font.size': 24, 'axes.unicode_minus': False})

    for ch_idx, ch_name in enumerate(channel_names):
        fig = plt.figure(figsize=(24, 16))
        gs = plt.GridSpec(4, 1, height_ratios=[2, 2, 2, 6])
        ax1 = plt.subplot(gs[0])  # EEG
        ax2 = plt.subplot(gs[1])  # EMG
        ax3 = plt.subplot(gs[2])  # EOG
        ax4 = plt.subplot(gs[3])  # spectrogram

        time_axis = np.linspace(beijing_start, beijing_start + total_hours,
                                 eeg_data.shape[1] if eeg_data.shape[1] < 100000 else 50000)
        if eeg_data.shape[1] > 100000:
            step = len(eeg_data[ch_idx]) // 50000
            eeg_plot = eeg_data[ch_idx][::step]
        else:
            eeg_plot = eeg_data[ch_idx]
        ax1.plot(time_axis, eeg_plot, color='blue', linewidth=0.5)
        ax1.set_title(f'EEG - {ch_name}', fontsize=24)
        ax1.set_ylabel('Amplitude (uV)', fontsize=24)
        ax1.set_ylim(-100, 100)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.tick_params(axis='y', labelsize=24)
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # EMG
        if emg_data is not None:
            if len(emg_data) > 100000:
                step = len(emg_data) // 50000
                emg_plot = emg_data[::step]
                emg_time = np.linspace(beijing_start, beijing_start + total_hours, len(emg_plot))
            else:
                emg_plot = emg_data
                emg_time = time_axis
            ax2.plot(emg_time, emg_plot, color='green', linewidth=0.5)
            ax2.set_title('EMG', fontsize=24)
        else:
            ax2.plot([beijing_start, beijing_start+total_hours], [0,0], 'k-')
            ax2.set_title('EMG (No Data)', fontsize=24)
        ax2.set_ylabel('Amplitude (uV)', fontsize=24)
        ax2.set_ylim(-200, 200)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='y', labelsize=24)
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # EOG
        if eog_data is not None:
            if len(eog_data) > 100000:
                step = len(eog_data) // 50000
                eog_plot = eog_data[::step]
                eog_time = np.linspace(beijing_start, beijing_start + total_hours, len(eog_plot))
            else:
                eog_plot = eog_data
                eog_time = time_axis
            ax3.plot(eog_time, eog_plot, color='purple', linewidth=0.5)
            ax3.set_title('EOG', fontsize=24)
        else:
            ax3.plot([beijing_start, beijing_start+total_hours], [0,0], 'k-')
            ax3.set_title('EOG (No Data)', fontsize=24)
        ax3.set_ylabel('Amplitude (uV)', fontsize=24)
        ax3.set_ylim(-800, 800)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', labelbottom=False)
        ax3.tick_params(axis='y', labelsize=24)
        ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # Spectrogram
        f, t, Sxx = spectrogram(eeg_data[ch_idx], fs=fs, nperseg=int(fs*10), noverlap=int(fs*5))
        t_hours = t / 3600 + beijing_start
        mask = f < 40
        im = ax4.pcolormesh(t_hours, f[mask], 10*np.log10(Sxx[mask]), shading='auto',
                            cmap='jet', vmin=-30, vmax=30)
        ax4.set_xlabel('Time (hours)', fontsize=24)
        ax4.set_ylabel('Frequency (Hz)', fontsize=24)
        ax4.set_ylim(0.5, 40)
        ax4.set_xlim(beijing_start, beijing_start + total_hours)
        ax4.set_xticks(np.arange(beijing_start, beijing_start+total_hours+1, 1))
        ax4.set_xticklabels([f'{int(h%24):02d}:00' for h in np.arange(beijing_start, beijing_start+total_hours+1, 1)])
        ax4.tick_params(labelsize=24)

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Power (dB)', fontsize=24)
        cbar.ax.tick_params(labelsize=24)

        out_dir = os.path.join(folder_path, 'plot_long')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{folder_name}_{ch_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved long plot: {ch_name}")

def plot_short_results(folder_path, eeg_data, channel_names, emg_data, eog_data,
                       fs, start_offset, duration_sec, config):
    """绘制短时波形图（EEG/EMG/EOG滤波后）"""
    folder_name = os.path.basename(folder_path)
    start_sample = int(start_offset * fs)
    end_sample = start_sample + int(duration_sec * fs)

    eeg_seg = eeg_data[:, start_sample:end_sample]
    emg_seg = emg_data[start_sample:end_sample] if emg_data is not None else None
    eog_seg = eog_data[start_sample:end_sample] if eog_data is not None else None

    plt.rcParams.update({'font.size': 24, 'axes.unicode_minus': False})

    for ch_idx, ch_name in enumerate(channel_names):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 12))
        time_axis = np.linspace(0, duration_sec, eeg_seg.shape[1])

        # EEG
        eeg_filtered = butter_bandpass_filter(eeg_seg[ch_idx], config['filter_eeg_low'],
                                               config['filter_eeg_high'], fs, config['filter_order'])
        ax1.plot(time_axis, eeg_filtered, color='blue', linewidth=0.8)
        ax1.set_title(f'EEG ({ch_name}) - {duration_sec}s', fontsize=24)
        ax1.set_ylabel('uV', fontsize=24)
        ax1.set_ylim(-50, 50)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.tick_params(axis='y', labelsize=24)
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # EMG
        if emg_seg is not None:
            emg_filtered = butter_bandpass_filter(emg_seg, config['filter_emg_low'],
                                                   config['filter_emg_high'], fs, config['filter_order'])
            ax2.plot(time_axis, emg_filtered, color='green', linewidth=0.8)
            ax2.set_title(f'EMG - {duration_sec}s', fontsize=24)
        else:
            ax2.plot([0, duration_sec], [0,0], 'k-')
            ax2.set_title('EMG (No Data)', fontsize=24)
        ax2.set_ylabel('uV', fontsize=24)
        ax2.set_ylim(-50, 50)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='y', labelsize=24)
        ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # EOG
        if eog_seg is not None:
            eog_filtered = butter_bandpass_filter(eog_seg, config['filter_eog_low'],
                                                   config['filter_eog_high'], fs, config['filter_order'])
            ax3.plot(time_axis, eog_filtered, color='purple', linewidth=0.8)
            ax3.set_title(f'EOG - {duration_sec}s', fontsize=24)
        else:
            ax3.plot([0, duration_sec], [0,0], 'k-')
            ax3.set_title('EOG (No Data)', fontsize=24)
        ax3.set_xlabel('Time (s)', fontsize=24)
        ax3.set_ylabel('uV', fontsize=24)
        ax3.set_ylim(-150, 150)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=24)
        ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # 动态刻度间隔
        if duration_sec <= 24:
            tick_interval = 5
        elif duration_sec <= 60:
            tick_interval = 10
        elif duration_sec <= 300:
            tick_interval = 30
        else:
            tick_interval = 60
        ax3.set_xticks(np.arange(0, duration_sec+1, tick_interval))

        plt.tight_layout()
        out_dir = os.path.join(folder_path, f'plot_short_{duration_sec}s')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{folder_name}_{ch_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved short plot: {ch_name}")

def process_single_folder(folder_path, config):
    """处理单个文件夹：预处理、微状态分析、绘图"""
    print(f"\nProcessing: {folder_path}")

    # 查找BDF/EDF文件
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for f in natsorted(files):
            if f.endswith(('.bdf', '.edf')):
                file_list.append(os.path.join(root, f))
    if not file_list:
        print("No BDF/EDF files found.")
        return None

    # 预处理（使用已有的预处理结果文件？可添加检查）
    raw, eeg_ch_names = preprocess_bdf(file_list)
    raw = check_non_eeg_raw(raw)

    # 提取数据
    eeg_data = raw.get_data(picks='eeg', units='uV')
    if 'EMG' in raw.ch_names:
        emg_data = raw.get_data(picks='emg', units='uV').flatten()
    else:
        emg_data = None
    eog_ch = [ch for ch in raw.ch_names if ch.startswith('EOG')]
    if eog_ch:
        eog_data = raw.get_data(picks=eog_ch[0], units='uV').flatten()
    else:
        eog_data = None

    fs = raw.info['sfreq']

    # 可选微状态分析
    if config['do_microstate']:
        lzma = analyze_microstates(eeg_data, n_maps=4)
    else:
        lzma = None

    # 绘图
    if config['plot_long_duration']:
        plot_long_results(folder_path, eeg_data, eeg_ch_names, emg_data, eog_data,
                          fs, total_hours=10, beijing_start=config['beijing_start_hour'])

    if config['plot_short_duration']:
        plot_short_results(folder_path, eeg_data, eeg_ch_names, emg_data, eog_data,
                           fs, config['short_start_offset'], config['short_duration_sec'], config)

    return lzma

def main():
    # 使用配置字典（可从命令行参数更新）
    config = CONFIG.copy()
    input_dir = config['input_dir']
    output_root = config['output_dir'] or input_dir

    # 遍历子文件夹
    sub_folders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    results = {}
    for folder in natsorted(sub_folders):
        lzma = process_single_folder(folder, config)
        if lzma is not None:
            results[folder] = lzma

    # 如果需要组间比较，可以在此处添加代码
    print("\nAll done.")

if __name__ == "__main__":
    main()
