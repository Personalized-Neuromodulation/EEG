import os
import json
import random
from pathlib import Path
import argparse

def split_hdf5_files_to_train_val(
    folder_path: str, 
    split_ratio: float = 0.8,
    seed: int = 42,
    output_json: str = None
) -> dict:
    """
    扫描HDF5文件夹并按比例拆分训练集和验证集
    
    Args:
        folder_path: 文件夹路径
        split_ratio: 训练集比例 (0-1)
        seed: 随机种子
        output_json: 输出JSON文件路径 (可选)
        
    Returns:
        包含训练和验证文件列表的字典
    """
    # 设置随机种子
    random.seed(seed)
    
    # 支持的HDF5文件扩展名
    hdf5_extensions = ['.h5', '.hdf5', '.hdf']
    
    # 扫描文件夹中的所有HDF5文件
    hdf5_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in hdf5_extensions):
            hdf5_files.append(file)
    
    print(f"找到 {len(hdf5_files)} 个HDF5文件")
    
    # 随机打乱文件列表
    # random.shuffle(hdf5_files)
    from natsort import natsorted
    hdf5_files = natsorted(hdf5_files)
    # 计算分割点
    split_idx = int(len(hdf5_files) * split_ratio)
    
    # 分割文件列表
    pretrain_files = hdf5_files[:split_idx]
    validation_files = hdf5_files[split_idx:]
    
    # 创建结果字典
    result = {
        "pretrain": pretrain_files,
        "validation": validation_files
    }
    
    # 输出统计信息
    print(f"训练集: {len(pretrain_files)} 个文件")
    print(f"验证集: {len(validation_files)} 个文件")
    print(f"比例: {len(pretrain_files)}:{len(validation_files)}")
    
    # 如果需要，保存到JSON文件
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_json}")
    
    return result

def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(description='拆分HDF5文件为训练集和验证集')
    parser.add_argument('folder_path', help='包含HDF5文件的文件夹路径')
    parser.add_argument('--split-ratio', type=float, default=0.8, 
                       help='训练集比例 (默认: 0.8)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子 (默认: 42)')
    parser.add_argument('--output', type=str, 
                       help='输出JSON文件路径 (可选)')
    
    args = parser.parse_args()
    
    # 执行拆分
    result = split_hdf5_files_to_train_val(
        folder_path=args.folder_path,
        split_ratio=args.split_ratio,
        seed=args.seed,
        output_json=args.output
    )
    
    # 如果没有指定输出文件，则打印结果
    if not args.output:
        print("\n拆分结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # 直接运行示例
    folder_path = r"/mnt/DATA/poolab/jepa_dataset/phsy/hdf5"
    
    # 执行拆分
    result = split_hdf5_files_to_train_val(
        folder_path=folder_path,
        split_ratio=0.8,
        seed=42,
        output_json="/mnt/DATA/poolab/jepa_dataset/EEG_JEPA/configs/dataset_split_cls_phsy.json"  # 可选：保存到文件
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
