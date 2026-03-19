import os
import h5py
import numpy as np
from scipy.signal import butter, filtfilt
import multiprocessing
from loguru import logger
import argparse
import mne
from natsort import natsorted

class EDFToHDF5Converter:
    """将EDF/BDF文件转换为HDF5格式，并进行重采样、标准化和坏道过滤"""

    def __init__(self, root_dir, target_dir, resample_rate=512, num_threads=1, num_files=-1,
                 flat_amp_threshold=1e-6, min_flat_duration=60*20):
        """
        Args:
            root_dir: 原始EDF/BDF文件根目录（递归扫描）
            target_dir: 输出HDF5文件目录
            resample_rate: 目标采样率 (Hz)
            num_threads: 并行进程数
            num_files: 处理的文件数，-1表示全部
            flat_amp_threshold: 平直通道检测的幅度阈值（窗口内最大值与最小值之差小于该值则视为平直）
            min_flat_duration: 最小平直持续时间（秒），若通道中存在连续平直段超过该值则剔除
        """
        self.resample_rate = resample_rate
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.num_threads = num_threads
        self.num_files = num_files
        self.flat_amp_threshold = flat_amp_threshold
        self.min_flat_duration = min_flat_duration

    def find_eeg_files(self, root_dir, file_extension):
        """递归查找指定扩展名的文件"""
        all_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(file_extension):
                    all_files.append(os.path.join(root, file))
        return natsorted(all_files)

    def get_files(self):
        """获取所有EDF文件路径"""
        return self.find_eeg_files(self.root_dir, file_extension="fif")

    def read_edf(self, file_path):
        """使用MNE读取EDF/BDF文件，返回信号列表、采样率和通道名"""
        logger.info(f'Reading {file_path}')
        if file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.bdf'):
            raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
        elif file_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        else:
            raise ValueError("Unsupported file format")
        signals = [raw.get_data(picks=[ch_name])[0] for ch_name in raw.ch_names]
        sample_rates = np.array([raw.info['sfreq'] for _ in raw.ch_names])
        channel_names = np.array(raw.ch_names)
        return signals, sample_rates, channel_names

    def filter_signal(self, signal, sample_rate):
        """抗混叠低通滤波，防止重采样时混叠"""
        nyquist = sample_rate / 2
        cutoff = min(self.resample_rate / 2, nyquist)
        if cutoff >= nyquist:
            return signal
        normalized_cutoff = cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype='low')
        filtered = filtfilt(b, a, signal)
        return filtered

    def safe_standardize(self, signal):
        """安全标准化，处理标准差为零的情况"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal - mean
        else:
            return (signal - mean) / std

    def resample_signals(self, signals, sample_rates):
        """对信号列表进行重采样、滤波和标准化"""
        logger.info('Resampling signals')
        resampled = []
        for sig, rate in zip(signals, sample_rates):
            duration = len(sig) / rate
            orig_time = np.linspace(0, duration, len(sig), endpoint=False)
            new_len = int(duration * self.resample_rate)
            new_time = np.linspace(0, duration, new_len, endpoint=False)

            # 如果原始采样率高于目标，先滤波
            if rate > self.resample_rate:
                sig = self.filter_signal(sig, rate)

            # 线性插值
            resig = np.interp(new_time, orig_time, sig)
            resig = self.safe_standardize(resig)

            if np.isnan(resig).any():
                logger.warning('NaN detected after resampling, skipping channel')
                continue
            resampled.append(resig)
        return np.stack(resampled) if resampled else np.array([])

    def save_to_hdf5(self, signals, channel_names, file_path):
        """保存信号到HDF5文件，使用分块和压缩"""
        logger.info(f'Saving to {file_path}')
        chunk_size = 5 * 60 * self.resample_rate  # 5分钟一个chunk
        with h5py.File(file_path, 'w') as hdf:
            for sig, name in zip(signals, channel_names):
                # 确保数据集名唯一
                base = name
                i = 1
                while base in hdf:
                    base = f"{name}_{i}"
                    i += 1
                hdf.create_dataset(base, data=sig,
                                   dtype='float16',
                                   chunks=(chunk_size,),
                                   compression="gzip")

    def _has_flat_segment(self, signal, fs):
        """
        检测信号中是否存在连续平直段超过最小持续时间。
        返回 True 表示该通道应被剔除（存在长平直段）。
        """
        win_len = int(fs)  # 1秒窗口
        n_windows = len(signal) // win_len
        if n_windows == 0:
            return False

        # 将信号分割为连续窗口，计算每个窗口的极差
        flat_windows = []
        for i in range(n_windows):
            seg = signal[i*win_len:(i+1)*win_len]
            peak_to_peak = np.max(seg) - np.min(seg)
            flat_windows.append(peak_to_peak < self.flat_amp_threshold)

        # 寻找最长连续平直窗口序列
        max_consecutive = 0
        current = 0
        for flag in flat_windows:
            if flag:
                current += 1
            else:
                if current > max_consecutive:
                    max_consecutive = current
                current = 0
        if current > max_consecutive:
            max_consecutive = current

        # 最长连续平直秒数
        flat_seconds = max_consecutive
        return flat_seconds >= self.min_flat_duration

    def convert(self, edf_path, hdf5_path):
        """转换单个EDF文件：先剔除平直通道，再重采样、标准化、保存"""
        signals, rates, ch_names = self.read_edf(edf_path)

        # 假设所有通道采样率相同，取第一个
        fs = rates[0] if len(rates) > 0 else None
        if fs is None:
            logger.warning(f"No channels in {edf_path}, skipping.")
            return

        valid_indices = []
        for i, sig in enumerate(signals):
            # 转换为float64以提高数值稳定性
            sig_float = np.asarray(sig, dtype=np.float64)
            if not self._has_flat_segment(sig_float, fs):
                valid_indices.append(i)
            else:
                logger.info(f"Removed channel due to long flat segment: {ch_names[i]}")

        if not valid_indices:
            logger.warning(f"No valid channels in {edf_path}, skipping.")
            return

        signals = [signals[i] for i in valid_indices]
        rates = rates[valid_indices]
        ch_names = ch_names[valid_indices]

        # 重采样
        resampled = self.resample_signals(signals, rates)
        if resampled.size == 0:
            logger.warning(f"No valid signals after resampling in {edf_path}")
            return

        self.save_to_hdf5(resampled, ch_names, hdf5_path)

    def convert_multiprocessing(self, edf_list):
        """多进程的工作函数：处理一批文件"""
        for edf in edf_list:
            parent_dir = os.path.basename(edf).split(".")[0]
            hdf5 = os.path.join(self.target_dir, f"{parent_dir}.hdf5")
            if os.path.exists(hdf5):
                logger.info(f"Already exists: {hdf5}")
                continue
            self.convert(edf, hdf5)
        return 1

    def convert_all_multiprocessing(self):
        """多进程并行转换所有文件"""
        edf_files = self.get_files()
        if self.num_files != -1:
            edf_files = edf_files[:self.num_files]

        chunks = np.array_split(edf_files, self.num_threads)
        tasks = [(chunk,) for chunk in chunks]

        with multiprocessing.Pool(self.num_threads) as pool:
            pool.starmap(self.convert_multiprocessing, tasks)
        logger.info(f"Conversion completed, processed {len(edf_files)} files.")


def main():
    parser = argparse.ArgumentParser(description="Convert EDF/BDF to HDF5 with resampling and flat channel removal.")
    parser.add_argument('--root_dir', type=str, default="/mnt/nas2/DATA/Sleep_Datasets/phys/fif",
                        help='Root directory containing EDF/BDF files (recursive scan).')
    parser.add_argument('--target_dir', type=str, default="/mnt/DATA/poolab/jepa_dataset/phsy/hdf5",
                        help='Directory to save HDF5 files.')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of parallel threads.')
    parser.add_argument('--num_files', type=int, default=-1,
                        help='Number of files to process. -1 means all.')
    parser.add_argument('--resample_rate', type=int, default=128,
                        help='Target sampling rate (Hz).')
    parser.add_argument('--flat_amp_threshold', type=float, default=1e-6,
                        help='Amplitude threshold for flat segment detection (peak-to-peak).')
    parser.add_argument('--min_flat_duration', type=int, default=60*5,
                        help='Minimum duration (seconds) of flat segment to discard a channel.')
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    converter = EDFToHDF5Converter(
        root_dir=args.root_dir,
        target_dir=args.target_dir,
        num_threads=args.num_threads,
        num_files=args.num_files,
        resample_rate=args.resample_rate,
        flat_amp_threshold=args.flat_amp_threshold,
        min_flat_duration=args.min_flat_duration
    )
    converter.convert_all_multiprocessing()


if __name__ == "__main__":
    main()