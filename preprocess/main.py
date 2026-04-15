#!/usr/bin/env python
"""
主程序入口：批量处理多天 EEG 数据
每个子文件夹代表一天（一个 session），内含多个 BDF/EDF 文件，预处理后保存为 BIDS 格式。
"""
import os
import json
import glob
import multiprocessing
from functools import partial
from tqdm import tqdm
from preprocessing_utils import find_eeg_files
from natsort import natsorted
from mne_bids import write_raw_bids, BIDSPath
from preprocess_fun import preprocess_EEG
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
def process_session(session_info, config):
    """处理单个 session 的函数，用于并行执行"""
    idx, session_name, session_path = session_info
    # 收集该 session 下的所有 BDF/EDF 文件
    file_paths = find_eeg_files(session_path)
    if not file_paths:
        return f"警告: 子文件夹 {session_name} 中没有 EEG 文件，跳过", False

    print(f"处理 session: {session_name}，包含 {len(file_paths)} 个文件")

    # 预处理（传入文件列表）
    eeg = preprocess_EEG(file_paths)
    clean_raw = eeg.process_main(
        useICA=config.get('useICA', False),
        useFASTER=config.get('useFASTER', True),
        remove_bad_channels=config.get('remove_bad_channels', False),
        stim_analysis=config.get('stim_analysis', False)
    )

    # 生成 BIDS 标识
    subject_id = f"{config['start_subject_id'] + idx:03d}"
    import re
    session_label = re.sub(r'[^a-zA-Z0-9]', '', session_name)
    bids_path = BIDSPath(
        subject=subject_id,
        session=session_label,
        task=config.get('task', 'sleep'),
        root=config['output_dir'],
        datatype='eeg'
    )   
    # 保存为 BIDS 格式
    write_raw_bids(
        clean_raw, 
        bids_path, 
        overwrite=True, 
        verbose=False, 
        allow_preload=True,   # 允许传入已预加载的 Raw 对象
        format='EDF'          # 指定输出格式，可选 'EDF', 'BDF' 等
    )
    return f"已保存 {subject_id} / {session_label}", True
    
def main():
    """主函数，根据配置文件执行"""
    config_path = r'D:\analysis\Microstate_analysis\preprocess\config.yaml'
    config = load_config(config_path)

    input_dir = config['input_dir']
    output_dir = config['output_dir']
    
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有子文件夹（每个代表一个 session）
    subdirs = [d for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d))]
    subdirs = natsorted(subdirs)
    if not subdirs:
        print(f"错误: 在 {input_dir} 中未找到任何子文件夹")
        return

    # 准备 session 信息列表
    session_infos = []
    for idx, session_name in enumerate(subdirs):
        session_path = os.path.join(input_dir, session_name)
        session_infos.append((idx, session_name, session_path))

    # 并行处理
    n_jobs = config.get('n_jobs', multiprocessing.cpu_count())
    print(f"使用 {n_jobs} 个进程并行处理 {len(session_infos)} 个 session")

    func = partial(process_session, config=config)

    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap_unordered(func, session_infos), total=len(session_infos)))

    # 输出结果统计
    success_count = 0
    fail_count = 0
    for msg, success in results:
        print(msg)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n处理完成！成功: {success_count}, 失败: {fail_count}")

if __name__ == '__main__':
    main()