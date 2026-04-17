"""
draw_sem_t_no_draw_dc.py
功能：从预处理好的 .npy 文件中提取 PSD，绘制各通道的平均 PSD ± SEM，并标注统计显著性。
去除所有 try-except 捕获。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import shutil
from natsort import natsorted
from scipy.stats import ttest_rel
import matplotlib.patches as mpatches
import json
import pickle
from statsmodels.stats.multitest import multipletests

# ==================== 全局配置参数 ====================
sampling_rate = 1
nperseg = 0
condition_names = ["Before Stimulation", "After Stimulation"]
condition_colors = ['#1f77b4', '#d62728']
comparison_colors = {'after_vs_before': '#006400'}
comparison_names = {"after_vs_before": "After vs before (FDR)"}
n_channels = 18
channel_names = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F7', 'F4', 'F8',
    'C3', 'T7', 'C4', 'T8', 'Pz', 'P3', 'P7', 'P4', 'P8',
    'O1', 'O2'
]
crop_params = {
    'before': {'pre': 0, 'post': 0},
    'after': {'pre': 0, 'post': 0}
}

# ==================== 文件组织函数 ====================
def organize_files_into_conditions_shuffle(original_folders, files_per_condition,
                                           condition_output_dirs, condition_index, num_groups):
    for folder_idx, folder_path in enumerate(original_folders):
        npy_files = natsorted(glob.glob(os.path.join(folder_path, "*.npy")))
        total_files = len(npy_files)
        print(f"文件夹 {os.path.basename(folder_path)} 有 {total_files} 个文件")

        condition_folders = [os.path.join(condition_output_dirs[i], os.path.basename(folder_path))
                             for i in range(num_groups)]
        for cf in condition_folders:
            os.makedirs(cf, exist_ok=True)

        for group_idx in range(num_groups):
            indices = condition_index[group_idx]
            file_indices = []
            for i, idx in enumerate(indices):
                raw_idx = (idx - 1) + i * num_groups
                if raw_idx < total_files:
                    file_indices.append(raw_idx)
            if len(file_indices) < files_per_condition:
                print(f"警告: condition {group_idx+1} 只有 {len(file_indices)} 个文件，预期 {files_per_condition}")

            for file_idx in file_indices:
                src = npy_files[file_idx]
                dst = os.path.join(condition_folders[group_idx], os.path.basename(src))
                shutil.copy2(src, dst)
            print(f"  复制 {len(file_indices)} 个文件到 {os.path.basename(condition_folders[group_idx])}")

    print("文件已组织到各个 condition 文件夹中")

# ==================== PSD 提取函数 ====================
def extract_all_file_psds(folder_path, is_normalization=False):
    npy_files = natsorted(glob.glob(os.path.join(folder_path, "*.npy")))
    if not npy_files:
        print(f"警告: {folder_path} 中没有找到任何文件")
        return np.array([]), np.array([])

    all_psds = []
    common_freqs = None
    file_shapes = []
    for file_path in npy_files:
        data = np.load(file_path)
        if data.shape[0] > n_channels:
            data = data[:n_channels, :]
        elif data.shape[0] < n_channels:
            print(f"警告: {file_path} 通道数 {data.shape[0]} 少于配置的 {n_channels}，将补零")
            pad = np.zeros((n_channels - data.shape[0], data.shape[1]))
            data = np.vstack([data, pad])

        if data.shape[0] != n_channels:
            data = data.T

        base = os.path.basename(file_path)
        if "before" in base:
            pre = crop_params['before']['pre'] * sampling_rate
            post = crop_params['before']['post'] * sampling_rate
        elif "after" in base:
            pre = crop_params['after']['pre'] * sampling_rate
            post = crop_params['after']['post'] * sampling_rate
        else:
            pre = 0
            post = 0

        if pre + post > 0:
            data = data[:, pre:data.shape[1]-post]

        if not is_normalization and not base.startswith("normalization"):
            data = data * 1e6

        file_psds = []
        for ch_idx in range(n_channels):
            ch_data = data[ch_idx, :]
            ch_data = ch_data[np.isfinite(ch_data)]
            if len(ch_data) == 0:
                psd_db = np.array([])
            else:
                freqs, psd = signal.welch(
                    ch_data, fs=sampling_rate, nperseg=nperseg,
                    scaling='density', average='mean', detrend='constant'
                )
                psd_db = 10 * np.log10(psd)
                if common_freqs is None:
                    mask = freqs <= 40
                    common_freqs = freqs[mask]
                    n_freqs_actual = len(common_freqs)
                if len(psd_db) >= n_freqs_actual:
                    psd_db = psd_db[:n_freqs_actual]
                else:
                    pad_len = n_freqs_actual - len(psd_db)
                    psd_db = np.pad(psd_db, (0, pad_len), constant_values=np.nan)
            file_psds.append(psd_db)

        file_psds_array = np.array(file_psds)
        all_psds.append(file_psds_array)
        file_shapes.append((os.path.basename(file_path), file_psds_array.shape))

    if not all_psds:
        print(f"错误: {folder_path} 中没有有效文件")
        return np.array([]), np.array([])

    all_psds = np.array(all_psds)
    if len(all_psds) > 1:
        first_shape = all_psds[0].shape
        for name, shape in file_shapes:
            if shape != first_shape:
                print(f"错误：PSD结果形状不一致！文件 {name} 形状 {shape}，期望 {first_shape}")
                raise ValueError("PSD 形状不一致，请检查数据长度或 nperseg 设置")

    return all_psds, common_freqs

# ==================== 统计检验函数 ====================
def calculate_pairwise_t_tests_with_fdr(all_psds_conditions, freqs, save_dir=None, condition_idx=None, force_recompute=False):
    n_freqs = len(freqs)
    if len(all_psds_conditions) < 2:
        print("错误: 需要至少两种条件进行配对 t 检验")
        return {}

    n_files = min(psds.shape[0] for psds in all_psds_conditions if psds.size > 0)
    if n_files == 0:
        print("警告: 无有效数据")
        return {}

    comparisons = [('after_vs_before', 1, 0)]
    results = {}

    for comp_name, idx_test, idx_ref in comparisons:
        data_test = all_psds_conditions[idx_test][:n_files, :, :n_freqs]
        data_ref  = all_psds_conditions[idx_ref][:n_files, :, :n_freqs]

        channel_p_values = np.ones((n_channels, n_freqs))
        for ch in range(n_channels):
            for f in range(n_freqs):
                test_vals = data_test[:, ch, f]
                ref_vals  = data_ref[:, ch, f]
                mask = np.isfinite(test_vals) & np.isfinite(ref_vals)
                if np.sum(mask) < 2:
                    p_val = 1.0
                else:
                    _, p_val = ttest_rel(test_vals[mask], ref_vals[mask])
                channel_p_values[ch, f] = p_val

        corrected_p = np.ones_like(channel_p_values)
        significance = np.zeros_like(channel_p_values, dtype=bool)
        for ch in range(n_channels):
            p_vals = channel_p_values[ch, :]
            finite_mask = np.isfinite(p_vals)
            if np.any(finite_mask):
                reject, p_corr, _, _ = multipletests(p_vals[finite_mask], alpha=0.05, method='fdr_bh')
                corrected_p[ch, finite_mask] = p_corr
                significance[ch, finite_mask] = reject

        results[comp_name] = {
            'channel_p_values': channel_p_values,
            'corrected_p_values': corrected_p,
            'channel_significance': significance,
            'n_files': n_files
        }

    if save_dir and condition_idx is not None:
        os.makedirs(save_dir, exist_ok=True)
        cache_file = os.path.join(save_dir, f"condition_{condition_idx}_ttest_results.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"t 检验结果已保存到: {cache_file}")

    return results

# ==================== 绘图函数 ====================
def plot_channel_mean_with_sem(all_psds_conditions, freqs, channel_idx, channel_name,
                               output_dir, condition_name, is_log_psd=True):
    n_freqs = len(freqs)
    plt.figure(figsize=(12, 8))
    plt.title(f"{channel_name} - {condition_name}", fontsize=14, weight='bold')

    condition_mean = []
    condition_sem = []
    valid_conditions = []

    for cond_idx, psds in enumerate(all_psds_conditions):
        if psds.size == 0:
            condition_mean.append(None)
            condition_sem.append(None)
            continue
        ch_data = psds[:, channel_idx, :]
        ch_data = ch_data[~np.isnan(ch_data).any(axis=1)]
        if ch_data.shape[0] == 0:
            condition_mean.append(None)
            condition_sem.append(None)
            continue
        mean_psd = np.mean(ch_data, axis=0)
        sem_psd = np.std(ch_data, axis=0) / np.sqrt(ch_data.shape[0])
        condition_mean.append(mean_psd)
        condition_sem.append(sem_psd)
        valid_conditions.append(cond_idx)

    if not valid_conditions:
        print(f"警告: {channel_name} 没有有效数据")
        plt.close()
        return

    for cond_idx in valid_conditions:
        color = condition_colors[cond_idx]
        label = f"{condition_names[cond_idx]} ({all_psds_conditions[cond_idx].shape[0]})"
        mean_psd = condition_mean[cond_idx]
        sem_psd = condition_sem[cond_idx]
        lower = mean_psd - sem_psd
        upper = mean_psd + sem_psd

        plt.fill_between(freqs, lower, upper, color=color, alpha=0.3, linewidth=0)
        plt.plot(freqs, mean_psd, color=color, linewidth=2.5, label=label)

    if is_log_psd:
        y_label = "Power (dB)"
    else:
        y_label = "Power (μV²/Hz)"
    plt.ylabel(y_label, fontsize=10)
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.xlim(0.5, 40)

    all_means = [mean for mean in condition_mean if mean is not None]
    if all_means:
        all_vals = np.concatenate([m for m in all_means if m is not None])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) > 0:
            y_min = np.percentile(all_vals, 1) - 5
            y_max = np.percentile(all_vals, 99) + 5
            plt.ylim(y_min, y_max)

    freq_bands = {
        r'$\delta$': (0.5, 4), r'$\theta$': (4, 8),
        r'$\alpha$': (8, 12), r'$\beta$': (12, 30),
        r'$\gamma$': (30, 40)
    }
    band_y = y_min + 0.02 * (y_max - y_min) if 'y_min' in locals() else -30
    for band, (f_min, f_max) in freq_bands.items():
        plt.axvspan(f_min, f_max, color='gray', alpha=0.05)
        plt.text((f_min+f_max)/2, band_y, band, fontsize=10, ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, pad=2, lw=0))

    ttest_results = calculate_pairwise_t_tests_with_fdr(
        all_psds_conditions, freqs, save_dir=output_dir, condition_idx=condition_name, force_recompute=False
    )
    if ttest_results:
        y_top = y_max - 2 if 'y_max' in locals() else plt.ylim()[1] - 2
        for comp_name, res in ttest_results.items():
            if 'channel_significance' not in res:
                continue
            sig_mask = res['channel_significance'][channel_idx]
            freq_mask = (freqs >= 0.5) & (freqs <= 40)
            sig_indices = np.where(sig_mask & freq_mask)[0]
            color = comparison_colors.get(comp_name, 'black')
            for f_idx in sig_indices:
                freq = freqs[f_idx]
                plt.text(freq, y_top, '*', fontsize=14, color=color,
                         ha='center', va='center', weight='bold', zorder=35)

    plt.legend(loc='lower left', fontsize=8, frameon=True, framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{channel_name}_optimized.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

# ==================== 汇总目录创建 ====================
def create_channel_summary_directories(base_output_dir, files_per_condition):
    summary_dir = os.path.join(base_output_dir, "channel_summary")
    os.makedirs(summary_dir, exist_ok=True)
    channel_dirs = {}
    for ch_name in channel_names:
        ch_dir = os.path.join(summary_dir, ch_name)
        os.makedirs(ch_dir, exist_ok=True)
        channel_dirs[ch_name] = ch_dir
    return summary_dir, channel_dirs

if __name__ == "__main__":
    print("draw_sem_t_no_draw_dc 模块已加载")
    print(f"当前配置: 采样率={sampling_rate} Hz, 通道数={n_channels}")