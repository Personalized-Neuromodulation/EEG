#!/usr/bin/env python
"""
主程序入口：处理 EEG 数据，可选保存到 BIDS 格式
"""
import os
import argparse
import glob
from natsort import natsorted
from preprocessing_utils import find_eeg_files
from preprocess import preprocess_EEG
from bids.BIDS import BIDSProcessor 

def process_fun(file_path):
    """单个文件的处理函数（供 BIDSProcessor 调用）"""
    eeg = preprocess_EEG(file_path)
    clean_raw = eeg.process_main(useICA=False, useFASTER=True, remove_bad_channels=False, stim_analysis=False)
    return clean_raw

def main():
    parser = argparse.ArgumentParser(description="EEG 数据预处理管道")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录，包含 BDF/EDF 文件（递归搜索）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录，用于保存 BIDS 格式数据')
    parser.add_argument('--dataset_name', type=str, default='eeg_dataset',
                        help='数据集名称')
    parser.add_argument('--task', type=str, default='sleep',
                        help='任务名称')
    parser.add_argument('--session', type=str, default='ses-1',
                        help='会话名称')
    parser.add_argument('--start_subject_id', type=int, default=1,
                        help='起始受试者编号')
    parser.add_argument('--useICA', action='store_true',
                        help='是否使用 ICA')
    parser.add_argument('--useFASTER', action='store_true', default=True,
                        help='是否使用 FASTER')
    parser.add_argument('--remove_bad_channels', action='store_true',
                        help='是否移除坏通道')
    parser.add_argument('--stim_analysis', action='store_true',
                        help='是否启用刺激分析模式')
    args = parser.parse_args()

    # 获取所有待处理文件
    file_paths = find_eeg_files(args.input_dir)
    if not file_paths:
        print(f"在目录 {args.input_dir} 中未找到任何 BDF/EDF 文件")
        return

    # 初始化 BIDS 处理器
    processor = BIDSProcessor(
        bids_root=args.output_dir,
        dataset_name=args.dataset_name,
        task=args.task,
        session=args.session,
        start_subject_id=args.start_subject_id
    )

    # 定义处理函数，将命令行参数传递给 process_fun（通过闭包或 functools.partial）
    def process_with_args(file_path):
        eeg = preprocess_EEG(file_path)
        clean_raw = eeg.process_main(
            useICA=args.useICA,
            useFASTER=args.useFASTER,
            remove_bad_channels=args.remove_bad_channels,
            stim_analysis=args.stim_analysis
        )
        return clean_raw

    # 批量处理
    results = processor.batch_process_files(
        file_paths=file_paths,
        task=args.task,
        process_func=process_with_args,
        pipeline_name="FASTER_pipeline",
        description="clean"
    )

    # 输出汇总
    processor.print_summary()

if __name__ == '__main__':
    main()
