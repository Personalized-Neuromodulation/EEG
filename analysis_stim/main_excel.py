"""
实验使用进程池并行，每个实验内部使用线程池处理片段。
全局配置从 config.yaml 读取，实验参数从 Excel 读取（包括时间偏移量）。
新增 PSD 比较策略：1. before vs after (shift n)  2. before vs before (shift n)  3. after vs after (shift n)
"""

import os
import sys
import time
import logging
import numpy as np
import mne
import shutil
import re
import glob
import pandas as pd
import yaml
from natsort import natsorted
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from datetime import datetime, timezone

sys.path.append(r'D:\code\time_frequency_analyse\time_frequency_analysis\analysis_stim_fun')
sys.path.append(r'D:\code\time_frequency_analyse\time_frequency_analysis\analysis_stim_fun')

from stim_analysis.merge_bdf import concatenate_bdf_files
from stim_analysis.monkey_preprocess import preprocess_EEG
import stim_analysis.draw_sem_t_no_draw_dc as draw_module

# ==================== 日志设置 ====================
def setup_logging(base_dir):
    log_dir = os.path.join(base_dir, "analysis_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"processing_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    return log_dir

# ==================== 标记提取函数 ====================
def extract_marker_info(desc, flag=True):
    desc_str = str(desc).strip()
    if re.search(r'2001\b\D*(\d+)', desc_str):
        return desc_str
    if desc_str == "2001":
        return "2001"
    if re.search(r'5000\b\D*(\d+)', desc_str) and flag:
        flag = False
        return desc_str
    if re.search(r'2001\s+\d+', desc_str):
        return desc_str
    return None

# ==================== 提取原始文件名前缀 ====================
def extract_file_prefix(file_paths):
    if not file_paths:
        return "unknown"
    basename = os.path.basename(file_paths[0])
    name_without_ext = os.path.splitext(basename)[0]
    match = re.match(r'^(.*?)(?:_\d+)?$', name_without_ext)
    if match:
        return match.group(1)
    return name_without_ext

# ==================== 收集子片段任务 ====================
def collect_subsegment_tasks(annotations, segment_info, output_dir, index, file_prefix,
                             before_tmin_offset, before_tmax_offset,
                             after_tmin_offset, after_tmax_offset):
    seg_start = segment_info['start_time']
    seg_end = segment_info['end_time']
    before_dir = os.path.join(output_dir, "before_30_preprocessed")
    after_dir = os.path.join(output_dir, "after_30_preprocessed")
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)

    seg_annotations = []
    for i, onset in enumerate(annotations.onset):
        if seg_start <= onset <= seg_end:
            seg_annotations.append({
                'time': onset,
                'duration': annotations.duration[i],
                'description': str(annotations.description[i])
            })
    seg_annotations.sort(key=lambda x: x['time'])

    before_30_start = seg_start
    before_30_end = None
    after_30_start = None

    for anno in seg_annotations:
        desc = str(anno['description'])
        if "Start of stimulation" in desc and before_30_end is None:
            before_30_end = anno['time']
        if "End of stimulation" in desc or "2000" in desc:
            after_30_start = anno['time']

    tasks = []
    if before_30_end:
        output_path = os.path.join(before_dir, f"{file_prefix}_{index}_before_30.npy")
        if not os.path.exists(output_path):
            tmin = before_30_start + before_tmin_offset
            tmax = before_30_start + before_tmax_offset
            tasks.append({
                'tmin': tmin,
                'tmax': tmax,
                'output_path': output_path,
                'start_desc': segment_info['start_desc'],
                'info_file': os.path.join(before_dir, f"{file_prefix}_info.txt")
            })
    if after_30_start:
        output_path = os.path.join(after_dir, f"{file_prefix}_{index}_after_30.npy")
        if not os.path.exists(output_path):
            tmin = after_30_start + after_tmin_offset
            tmax = after_30_start + after_tmax_offset
            tasks.append({
                'tmin': tmin,
                'tmax': tmax,
                'output_path': output_path,
                'start_desc': segment_info['start_desc'],
                'info_file': os.path.join(after_dir, f"{file_prefix}_info.txt")
            })
    return tasks

# ==================== 预处理单个片段（线程池任务） ====================
def preprocess_segment_thread(params, raw_data, raw_info, useICA, useFASTER):
    from monkey_preprocess import preprocess_EEG
    import mne
    import numpy as np
    from datetime import datetime, timezone

    tmin = params['tmin']
    tmax = params['tmax']
    output_path = params['output_path']
    start_desc = params['start_desc']
    info_file = params['info_file']

    sfreq = raw_info['sfreq']
    start_sample = int(tmin * sfreq)
    end_sample = int(tmax * sfreq)
    segment_data = raw_data[:, start_sample:end_sample].copy()

    info = mne.create_info(raw_info["ch_names"], sfreq)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, match_case=False, on_missing='warn')
    temp_raw = mne.io.RawArray(segment_data, info)

    eeg = preprocess_EEG(temp_raw, only_include_EEG_channels=True)
    processed_raw, no_eeg_raw = eeg.process_main(
        useICA=useICA,
        useFASTER=useFASTER,
        remove_bad_channels=False,
        stim_analysis=False
    )
    if no_eeg_raw is not None:
        eeg_raw = processed_raw.copy()
        raw_info_merged = mne.create_info(
            ch_names=eeg_raw.ch_names + no_eeg_raw.ch_names,
            sfreq=eeg_raw.info['sfreq'],
            ch_types=['eeg']*len(eeg_raw.ch_names) + no_eeg_raw.get_channel_types(),
        )
        merged_raw = mne.io.RawArray(
            np.vstack([eeg_raw.get_data(), no_eeg_raw.get_data()[:, :eeg_raw.get_data().shape[1]]]),
            raw_info_merged
        )
        if eeg_raw.info['meas_date'] is not None:
            merged_raw.set_meas_date(eeg_raw.info['meas_date'])
        else:
            merged_raw.set_meas_date(datetime.now(timezone.utc))
        merged_raw.info['bads'] = eeg_raw.info['bads'].copy()
        merged_raw.set_annotations(eeg_raw.annotations)
        processed_raw = merged_raw

    processed_data = processed_raw.get_data()
    np.save(output_path, processed_data)
    with open(info_file, 'a') as f:
        f.write(f"start_desc: {start_desc}\n")
    return True, output_path

# ==================== 切割并预处理（使用线程池） ====================
def cut_and_preprocess_threads(file_path, output_dir, file_prefix, useICA, useFASTER,
                               before_tmin_offset, before_tmax_offset,
                               after_tmin_offset, after_tmax_offset,
                               n_threads):
    if n_threads is None:
        n_threads = max(1, multiprocessing.cpu_count() * 2)
    logging.info(f"使用 {n_threads} 个线程并行预处理片段")

    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"加载完整原始数据: {file_path}")
    if file_path.endswith('.bdf'):
        raw_full = mne.io.read_raw_bdf(file_path, preload=True)
    else:
        raw_full = mne.io.read_raw_edf(file_path, preload=True)
    raw_data = raw_full.get_data()
    raw_info = raw_full.info
    logging.info(f"数据加载完成，形状: {raw_data.shape}, 采样率: {raw_info['sfreq']} Hz")

    annotations = raw_full.annotations
    total_time = raw_full.times[-1] if raw_full.times.size > 0 else raw_full.n_times / raw_full.info['sfreq']

    markers = []
    seq_time = []
    flag = True
    for idx, desc in enumerate(annotations.description):
        sequence = extract_marker_info(desc, flag)
        if sequence is not None:
            if len(seq_time) == 0:
                seq_time.append(annotations.onset[idx])
                markers.append({'time': annotations.onset[idx], 'sequence': sequence, 'description': desc})
            elif annotations.onset[idx] - seq_time[-1] > 3 and total_time - annotations.onset[idx] > 10:
                seq_time.append(annotations.onset[idx])
                markers.append({'time': annotations.onset[idx], 'sequence': sequence, 'description': desc})

    with open(os.path.join(output_dir, "markers.txt"), 'w') as f:
        for marker in markers:
            f.write(f"{marker['time']},{marker['sequence']},{marker['description']}\n")

    if not markers:
        logging.error("未找到任何有效的标记")
        return 0, raw_info['sfreq']

    segments = []
    for i in range(len(markers) - 1):
        current = markers[i]
        next_marker = markers[i + 1]
        segments.append({
            'start_time': current['time'],
            'end_time': next_marker['time'],
            'start_seq': current['sequence'],
            'end_seq': next_marker['sequence'],
            'start_desc': current['description'],
            'end_desc': next_marker['description']
        })

    all_tasks = []
    for i, segment in enumerate(segments):
        tasks = collect_subsegment_tasks(
            annotations, segment, output_dir, i, file_prefix,
            before_tmin_offset, before_tmax_offset,
            after_tmin_offset, after_tmax_offset
        )
        all_tasks.extend(tasks)

    logging.info(f"收集到 {len(all_tasks)} 个预处理任务，开始线程池处理...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        task_func = partial(preprocess_segment_thread, raw_data=raw_data, raw_info=raw_info,
                            useICA=useICA, useFASTER=useFASTER)
        futures = {executor.submit(task_func, task): task for task in all_tasks}
        for future in as_completed(futures):
            success, out_path = future.result()
            if success:
                success_count += 1
                logging.info(f"预处理完成: {out_path}")
            else:
                logging.error(f"预处理失败: {out_path}")

    return len(segments), raw_info['sfreq']

# ==================== 绘制 PSD 比较图（支持多种策略） ====================
def draw_psd_single_wrapper(before_folder, after_folder, output_dir, strategy='before_vs_after', shift_n=0):
    """
    绘制 PSD 比较图，支持三种策略：
    - 'before_vs_after': 原始策略，before[i] vs after[i]（shift_n 无效）
    - 'before_vs_after_shift': before[i] vs after[i+shift_n]
    - 'before_vs_before_shift': before[i] vs before[i+shift_n]
    - 'after_vs_after_shift': after[i] vs after[i+shift_n]
    """
    os.makedirs(output_dir, exist_ok=True)
    before_psds, before_freqs = draw_module.extract_all_file_psds(before_folder, is_normalization=False)
    after_psds, after_freqs = draw_module.extract_all_file_psds(after_folder, is_normalization=False)

    if before_freqs is None or after_freqs is None:
        logging.error("无法获取频率轴")
        return

    # 统一频率轴长度
    min_len = min(len(before_freqs), len(after_freqs))
    freqs = before_freqs[:min_len]
    before_psds = before_psds[:, :, :min_len] if before_psds.size > 0 else before_psds
    after_psds = after_psds[:, :, :min_len] if after_psds.size > 0 else after_psds

    n_before = before_psds.shape[0]
    n_after = after_psds.shape[0]

    # 根据策略生成配对数据
    if strategy == 'before_vs_after':
        # 原始配对：取相同索引，取 min(n_before, n_after)
        n_pairs = min(n_before, n_after)
        data1 = before_psds[:n_pairs, :, :]
        data2 = after_psds[:n_pairs, :, :]
        title_suffix = "Before vs After (paired)"
        comp_name = "before_vs_after"
    elif strategy == 'before_vs_after_shift':
        shift_n = int(shift_n)
        if shift_n < 0:
            logging.error("shift_n 必须非负")
            return
        n_pairs = min(n_before, n_after - shift_n)
        if n_pairs <= 0:
            logging.error(f"shift_n={shift_n} 导致无有效配对，跳过")
            return
        data1 = before_psds[:n_pairs, :, :]
        data2 = after_psds[shift_n:shift_n+n_pairs, :, :]
        title_suffix = f"Before vs After (shift +{shift_n})"
        comp_name = f"before_vs_after_shift_{shift_n}"
    elif strategy == 'before_vs_before_shift':
        shift_n = int(shift_n)
        if shift_n <= 0:
            logging.error("before_vs_before_shift 需要 shift_n > 0")
            return
        n_pairs = n_before - shift_n
        if n_pairs <= 0:
            logging.error(f"shift_n={shift_n} 超出 before 片段数，跳过")
            return
        data1 = before_psds[:n_pairs, :, :]
        data2 = before_psds[shift_n:shift_n+n_pairs, :, :]
        title_suffix = f"Before vs Before (shift +{shift_n})"
        comp_name = f"before_vs_before_shift_{shift_n}"
    elif strategy == 'after_vs_after_shift':
        shift_n = int(shift_n)
        if shift_n <= 0:
            logging.error("after_vs_after_shift 需要 shift_n > 0")
            return
        n_pairs = n_after - shift_n
        if n_pairs <= 0:
            logging.error(f"shift_n={shift_n} 超出 after 片段数，跳过")
            return
        data1 = after_psds[:n_pairs, :, :]
        data2 = after_psds[shift_n:shift_n+n_pairs, :, :]
        title_suffix = f"After vs After (shift +{shift_n})"
        comp_name = f"after_vs_after_shift_{shift_n}"
    else:
        logging.error(f"未知策略: {strategy}")
        return

    all_psds_conditions = [data1, data2]
    # 临时修改 condition_names 和 condition_colors 用于绘图（可选）
    # 这里直接使用原有绘图函数，但标题会显示 "Comparison"
    summary_dir, channel_dirs = draw_module.create_channel_summary_directories(output_dir, 1)
    plot_dir = os.path.join(output_dir, "psd_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for channel_idx, channel_name in enumerate(draw_module.channel_names):
        logging.info(f"绘制通道 {channel_name} 的PSD图 ({title_suffix})")
        # 调用原有绘图函数，但 condition_name 中包含策略信息
        draw_module.plot_channel_mean_with_sem(
            all_psds_conditions, freqs, channel_idx,
            channel_name, plot_dir, title_suffix
        )
        src_file = os.path.join(plot_dir, f"{channel_name}_optimized.png")
        dest_file = os.path.join(channel_dirs[channel_name], f"psd_{comp_name}.png")
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)

# ==================== 处理单个实验（拼接、切割、预处理、绘图） ====================
def process_concatenated_data_single_group(
    bdf_files,
    output_base_dir,
    subject_name,
    draw_psd,
    only_draw_psd,
    force_reconcat,
    useICA,
    useFASTER,
    before_tmin_offset,
    before_tmax_offset,
    after_tmin_offset,
    after_tmax_offset,
    n_threads,
    psd_strategy,
    psd_shift_n
):
    bdf_files = natsorted(bdf_files)
    file_prefix = extract_file_prefix(bdf_files)
    logging.info(f"使用文件前缀: {file_prefix}")

    concat_dir = os.path.join(output_base_dir, f"{subject_name}_concatenated")
    cut_dir = os.path.join(output_base_dir, f"{subject_name}_segments")
    os.makedirs(concat_dir, exist_ok=True)
    os.makedirs(cut_dir, exist_ok=True)
    concat_file_path = os.path.join(concat_dir, f"{subject_name}_concatenated.bdf")

    if not only_draw_psd:
        logging.info("=" * 60)
        logging.info(f"开始处理被试: {subject_name}")
        logging.info(f"待拼接文件数量: {len(bdf_files)}")
        for i, f in enumerate(bdf_files):
            logging.info(f"  文件 {i+1}: {os.path.basename(f)}")

        if os.path.exists(concat_file_path) and not force_reconcat:
            logging.info(f"使用已存在的拼接文件: {concat_file_path}")
        else:
            logging.info(f"开始拼接BDF文件...")
            concatenate_bdf_files(bdf_files, concat_file_path)
            logging.info(f"拼接完成: {concat_file_path}")

        logging.info(f"开始切割并使用线程池预处理数据...")
        processed_segments, sfreq = cut_and_preprocess_threads(
            concat_file_path, cut_dir, file_prefix, useICA, useFASTER,
            before_tmin_offset, before_tmax_offset,
            after_tmin_offset, after_tmax_offset,
            n_threads
        )
        if processed_segments == 0:
            logging.error(f"被试 {subject_name} 没有找到有效片段")
            return False
        logging.info(f"切割并预处理完成: {processed_segments} 个片段")
        logging.info(f"采样频率: {sfreq} Hz")

        before_folder = os.path.join(cut_dir, "before_30_preprocessed")
        after_folder = os.path.join(cut_dir, "after_30_preprocessed")
        if not os.path.exists(before_folder) or not os.path.exists(after_folder):
            logging.error(f"预处理文件夹不存在")
            return False
    else:
        before_folder = os.path.join(cut_dir, "before_30_preprocessed")
        after_folder = os.path.join(cut_dir, "after_30_preprocessed")

    if draw_psd:
        psd_output_dir = os.path.join(output_base_dir, f"{subject_name}_psd")
        os.makedirs(psd_output_dir, exist_ok=True)
        logging.info(f"开始绘制PSD图，策略: {psd_strategy}, shift_n={psd_shift_n}")
        draw_psd_single_wrapper(before_folder, after_folder, psd_output_dir,
                                strategy=psd_strategy, shift_n=psd_shift_n)
        logging.info(f"PSD图保存至: {psd_output_dir}")

    logging.info(f"被试 {subject_name} 处理完成!")
    return True

# ==================== 单个实验的入口（进程池调用） ====================
def process_single_experiment(row, global_cfg):
    # 必须字段
    bdf_folder = row['bdf_folder']
    output_base_dir = row['output_base_dir']
    subject_name = row.get('subject_name', f"sub_{os.path.basename(bdf_folder)[:8]}")
    sampling_rate = row.get('sampling_rate')
    if sampling_rate is None or pd.isna(sampling_rate):
        logging.error(f"实验 {subject_name} 缺少 sampling_rate，跳过")
        return False

    # 时间偏移量
    before_tmin_offset = row.get('before_tmin_offset')
    before_tmax_offset = row.get('before_tmax_offset')
    after_tmin_offset = row.get('after_tmin_offset')
    after_tmax_offset = row.get('after_tmax_offset')
    if any(v is None or pd.isna(v) for v in [before_tmin_offset, before_tmax_offset, after_tmin_offset, after_tmax_offset]):
        logging.error(f"实验 {subject_name} 缺少时间偏移量，跳过")
        return False

    # 检查是否已完成
    psd_output_dir = os.path.join(output_base_dir, f"{subject_name}_psd")
    if os.path.exists(psd_output_dir) and os.path.exists(os.path.join(psd_output_dir, "channel_summary")):
        logging.info(f"已完成（{psd_output_dir} 下存在 channel_summary），跳过")
        return True

    # 设置采样率
    sr = int(sampling_rate)
    draw_module.sampling_rate = sr
    logging.info(f"设置采样率为 {sr} Hz")

    # 片段时长（秒）= before_tmax_offset - before_tmin_offset
    segment_duration = before_tmax_offset - before_tmin_offset
    if segment_duration <= 0:
        logging.error(f"片段时长无效: {segment_duration}，跳过")
        return False
    total_points = int(sr * segment_duration)

    # 处理 nperseg
    nperseg_cfg = global_cfg.get('nperseg')
    nperseg = min(total_points, sr * nperseg_cfg)
    draw_module.nperseg = nperseg
    logging.info(f"设置 nperseg 为 {draw_module.nperseg}")

    # 读取 PSD 比较策略参数（从 Excel 或 config）
    psd_strategy = row.get('psd_strategy', global_cfg.get('psd_strategy', 'before_vs_after'))
    psd_shift_n = row.get('psd_shift_n', global_cfg.get('psd_shift_n', 0))
    # 验证策略有效性
    valid_strategies = ['before_vs_after', 'before_vs_after_shift', 'before_vs_before_shift', 'after_vs_after_shift']
    if psd_strategy not in valid_strategies:
        logging.warning(f"无效策略 {psd_strategy}，使用默认 before_vs_after")
        psd_strategy = 'before_vs_after'
    if psd_strategy != 'before_vs_after' and psd_shift_n <= 0:
        logging.warning(f"策略 {psd_strategy} 需要 shift_n > 0，当前为 {psd_shift_n}，强制设为 1")
        psd_shift_n = 1

    # 其他全局参数
    draw_psd = global_cfg.get('draw_psd', True)
    only_draw_psd = global_cfg.get('only_draw_psd', False)
    force_reconcat = global_cfg.get('force_reconcat', False)
    useICA = global_cfg.get('useICA', False)
    useFASTER = global_cfg.get('useFASTER', True)
    n_threads = global_cfg.get('n_threads_per_exp', None)

    # 获取所有 BDF 文件
    bdf_files = natsorted(glob.glob(os.path.join(bdf_folder, "*.bdf")))
    if not bdf_files:
        logging.error(f"在 {bdf_folder} 中没有找到BDF文件")
        return False

    logging.info(f"开始处理实验: {subject_name}, 共 {len(bdf_files)} 个BDF文件")
    logging.info(f"裁剪时间偏移: before({before_tmin_offset},{before_tmax_offset}) after({after_tmin_offset},{after_tmax_offset})")
    logging.info(f"片段时长: {segment_duration} 秒")
    logging.info(f"PSD 策略: {psd_strategy}, shift_n={psd_shift_n}")
    logging.info(f"draw_psd={draw_psd}, only_draw_psd={only_draw_psd}")

    success = process_concatenated_data_single_group(
        bdf_files=bdf_files,
        output_base_dir=output_base_dir,
        subject_name=subject_name,
        draw_psd=draw_psd,
        only_draw_psd=only_draw_psd,
        force_reconcat=force_reconcat,
        useICA=useICA,
        useFASTER=useFASTER,
        before_tmin_offset=before_tmin_offset,
        before_tmax_offset=before_tmax_offset,
        after_tmin_offset=after_tmin_offset,
        after_tmax_offset=after_tmax_offset,
        n_threads=n_threads,
        psd_strategy=psd_strategy,
        psd_shift_n=psd_shift_n
    )
    return success

# ==================== 主函数 ====================
def main():
    # 读取配置文件
    config_path = r"D:\code\time_frequency_analyse\time_frequency_analysis\analysis_stim_fun\stim_analysis\config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    global_cfg = config.get('global', {})
    excel_path = global_cfg.get('excel_path')

    # 读取实验参数
    df = pd.read_excel(excel_path)
    n_experiments = len(df)

    # 确定并行进程数
    n_processes = global_cfg.get('n_processes')
    if n_processes is None or n_processes <= 0:
        n_processes = max(1, multiprocessing.cpu_count() // 2)
    n_processes = min(n_experiments, n_processes)

    # 确定每个实验内部的线程数
    n_threads_per_exp = global_cfg.get('n_threads_per_exp')
    if n_threads_per_exp is None or n_threads_per_exp <= 0:
        n_threads_per_exp = max(1, multiprocessing.cpu_count() * 2 // n_processes)

    # 将线程数放入全局配置
    global_cfg['n_threads_per_exp'] = n_threads_per_exp

    logging.info(f"共 {n_experiments} 个实验，将使用 {n_processes} 个进程并行，每个实验内部使用 {n_threads_per_exp} 个线程")

    # 准备参数列表
    experiments = [(row, global_cfg) for _, row in df.iterrows()]

    # 启动进程池
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.starmap(process_single_experiment, experiments)

    success_count = sum(results)
    logging.info(f"所有实验处理完成: 成功 {success_count}/{n_experiments}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()