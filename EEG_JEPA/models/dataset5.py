import os
import json
import pickle
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
from loguru import logger
import glob
import random
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import click
import yaml
from datetime import datetime
from sklearn.metrics import f1_score
import deepspeed
from torch.utils.tensorboard import SummaryWriter

import os
import json
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
from loguru import logger
import glob
import random
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset

class SleepStagingDataset(Dataset):
    """
    睡眠分期数据集，每个样本对应一个30秒epoch
    返回：
        multi_modal_data: list of [实际通道数, T]  每个模态一个张量
        multi_modal_mask: list of [实际通道数]     每个模态的掩码（True表示填充，当前返回全False）
        label: int (0-4)
    """
    
    STAGE_MAP = {
        'Wake': 0, 'W': 0,
        'N1': 1,
        'N2': 2,
        'N3': 3,
        'REM': 4, 'R': 4
    }
    
    def __init__(
        self,
        config: Dict,
        channel_groups: Dict,
        split: str = "train",          # "train", "validation", 或 "all"
    ):
        super().__init__()
        
        self.config = config
        self.channel_groups = channel_groups
        self.split = split
        
        self.sampling_freq = config.get("sampling_freq", 128)
        self.epoch_duration = 30
        self.samples_per_epoch = self.epoch_duration * self.sampling_freq  # 3840
        
        # 加载标签文件路径映射
        labels_path = config.get("labels_path", "")
        label_files = glob.glob(os.path.join(labels_path, "**", "*.csv"), recursive=True)
        self.labels_dict = {os.path.basename(f).replace(".csv", ""): f for f in label_files}
        
        # 获取hdf5文件列表
        # 根据 split 从 split_path 中读取
        data_path = config["data_path"]
        split_data = self._load_split(config["split_path"])
        if split in split_data:
            hdf5_paths = [os.path.join(data_path, path) for path in split_data[split]]
        else:
            raise ValueError(f"Split '{split}' not found")
        self.hdf5_paths = [f for f in hdf5_paths if os.path.exists(f)]
        
        if config.get("max_files"):
            self.hdf5_paths = self.hdf5_paths[:config["max_files"]]
        
        # 预计算每个文件的模态到通道映射以及文件长度
        logger.info(f"预计算文件通道映射和长度 for {split}...")
        self.file_channel_map, self.file_lengths = self._precompute_channel_maps_and_lengths()
        
        # 构建索引和标签列表
        self.index_map = []      # 每个元素为 (hdf5_path, start_sample, label)
        self.labels = []          # 存储所有样本的标签，与 index_map 顺序一致
        pbar = tqdm(self.hdf5_paths, desc=f"Indexing {split} data")
        
        for hdf5_path in pbar:
            study_id = os.path.basename(hdf5_path).replace(".hdf5", "")
            if study_id not in self.labels_dict:
                continue
            label_path = self.labels_dict[study_id]
            labels_df = pd.read_csv(label_path, usecols=['sleep_stage'])
            total_samples = self.file_lengths[hdf5_path]
            n_epochs = min(total_samples // self.samples_per_epoch, len(labels_df))
            for epoch_idx in range(n_epochs):
                start_sample = epoch_idx * self.samples_per_epoch
                label_val = labels_df.iloc[epoch_idx]['sleep_stage']
                self.index_map.append((hdf5_path, start_sample, label_val))
                self.labels.append(label_val)   # 同步记录标签
        
        # 训练集打乱时，同时打乱 index_map 和 labels
        if split == "full":
            combined = list(zip(self.index_map, self.labels))
            random.shuffle(combined)
            self.index_map, self.labels = zip(*combined)
            self.index_map = list(self.index_map)
            self.labels = list(self.labels)
        
        logger.info(f"{split} set: {len(self.index_map)} epochs")
        if self.index_map:
            unique, counts = np.unique(self.labels, return_counts=True)
            logger.info(f"Class distribution: {dict(zip(unique, counts))}")
        
        # 文件句柄缓存
        self._open_files = {}
    
    def _load_split(self, split_path: str) -> Dict:
        with open(split_path, 'r') as f:
            return json.load(f)
    
    def _precompute_channel_maps_and_lengths(self) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, int]]:
        """预计算每个HDF5文件的模态通道映射和总样本长度"""
        file_channel_map = {}
        file_lengths = {}
        for hdf5_path in tqdm(self.hdf5_paths, desc="Precomputing channel maps and lengths"):
            try:
                with h5py.File(hdf5_path, 'r', rdcc_nbytes=1024*1024*1024) as hf:
                    dset_names = list(hf.keys())
                    modality_to_channels = {mod_type: [] for mod_type in self.config["modality_types"]}
                    for dset_name in dset_names:
                        for mod_type in self.config["modality_types"]:
                            if dset_name in self.channel_groups.get(mod_type, []):
                                modality_to_channels[mod_type].append(dset_name)
                                break
                    # 找到第一个非空模态的第一个通道以获取总长度
                    first_ch = None
                    for mod, chs in modality_to_channels.items():
                        if chs:
                            first_ch = chs[0]
                            break
                    if first_ch is None:
                        logger.warning(f"文件 {hdf5_path} 无任何可用通道，跳过")
                        continue
                    total_samples = hf[first_ch].shape[0]
                    file_channel_map[hdf5_path] = modality_to_channels
                    file_lengths[hdf5_path] = total_samples
            except Exception as e:
                logger.error(f"Error reading {hdf5_path}: {e}")
                continue
        return file_channel_map, file_lengths
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        hdf5_path, epoch_start, label = self.index_map[idx]
        epoch_end = epoch_start + self.samples_per_epoch
        
        # 获取或打开文件句柄（缓存）
        if hdf5_path not in self._open_files:
            self._open_files[hdf5_path] = h5py.File(
                hdf5_path, 'r', rdcc_nbytes=1024*1024*1024
            )
        hf = self._open_files[hdf5_path]
        
        epoch_data = []
        epoch_masks = []
        modality_to_channels = self.file_channel_map.get(hdf5_path, {})
        
        for mod_type in self.config["modality_types"]:
            channel_names = modality_to_channels.get(mod_type, [])
            if not channel_names:
                data = torch.empty((0, self.samples_per_epoch))
                mask = torch.empty(0, dtype=torch.bool)
            else:
                channel_data = []
                for ch_name in channel_names:
                    signal = hf[ch_name][epoch_start:epoch_end]
                    channel_data.append(torch.from_numpy(signal).float())
                data = torch.stack(channel_data, dim=0)
                mask = torch.zeros(len(channel_names), dtype=torch.bool)
                for ch in range(data.shape[0]):
                    if not mask[ch]:  # 真实通道
                        mean = data[ch].mean()
                        std = data[ch].std() + 1e-8
                        data[ch] = (data[ch] - mean) / std
                        
            epoch_data.append(data)
            epoch_masks.append(mask)
        
        return epoch_data, epoch_masks, label
    
    def __del__(self):
        """析构时关闭所有打开的文件句柄"""
        for f in self._open_files.values():
            f.close()
        self._open_files.clear()

def collate_fn_cls(batch):
    num_modalities = len(batch[0][0])
    batch_size = len(batch)
    collated_data = []
    collated_masks = []
    for m_idx in range(num_modalities):
        modality_data = [item[0][m_idx] for item in batch]
        modality_masks = [item[1][m_idx] for item in batch]
        max_channels = max([d.shape[0] for d in modality_data])
        T = modality_data[0].shape[1]
        padded_data = []
        padded_masks = []
        for data, mask in zip(modality_data, modality_masks):
            C = data.shape[0]
            if C < max_channels:
                pad_c = max_channels - C
                data = torch.cat([data, torch.zeros(pad_c, T, dtype=data.dtype)], dim=0)
                mask = torch.cat([mask, torch.ones(pad_c, dtype=torch.bool)], dim=0)
            padded_data.append(data)
            padded_masks.append(mask)
        collated_data.append(torch.stack(padded_data, dim=0))
        collated_masks.append(torch.stack(padded_masks, dim=0))
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return collated_data, collated_masks, labels

