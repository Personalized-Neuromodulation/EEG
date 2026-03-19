from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import h5py
import random
import torch
from tqdm import tqdm
import multiprocessing
import time
import sys
sys.path.append("../")
from utils import load_data, save_data
import pandas as pd
import json
from loguru import logger
from einops import rearrange


def index_file_helper(args):
    file_path, channel_like, chunk_size, channel_groups, modality_types = args
    file_index_map = []
    modality_to_channels = {modality_type: [] for modality_type in modality_types}
    # try:
    with h5py.File(file_path, 'r', rdcc_nbytes = 1024*1024*1024) as hf:
        dset_names = []
        # print(hf.keys())
        for dset_name in hf.keys():
            if not channel_like or dset_name in channel_like:
                if isinstance(hf[dset_name], h5py.Dataset):
                    dset_names.append(dset_name)
                    if dset_name in channel_groups["BAS"]:
                        modality_to_channels["BAS"].append(dset_name)
                    if dset_name in channel_groups["RESP"]:
                        modality_to_channels["RESP"].append(dset_name)
                    if dset_name in channel_groups["EKG"]:
                        modality_to_channels["EKG"].append(dset_name)
                    if dset_name in channel_groups["EMG"]:
                        modality_to_channels["EMG"].append(dset_name)
        # print(modality_to_channels)
        flag = True
        for modality, channels in modality_to_channels.items():
            if len(channels) == 0:
                flag = False
                break
        if flag:
            num_samples = hf[dset_name].shape[0]
            num_chunks = num_samples // chunk_size
            for chunk_start in range(0, num_chunks * chunk_size, chunk_size):
                file_index_map.append((file_path, dset_names, chunk_start))
    # except (OSError, AttributeError) as e:
    #     with open("problem_hdf5.txt", "a") as f:
    #         f.write(f"Error processing file {file_path}: {str(e)}\n")
    return file_index_map

def index_files(hdf5_paths, channel_like, samples_per_chunk, num_workers, channel_groups=None, modality_types=None):
    # results = index_file_helper()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(index_file_helper, [(path, channel_like, samples_per_chunk, channel_groups, modality_types) for path in hdf5_paths]), total=len(hdf5_paths), desc="Indexing files", position=0, leave=True))
    return [item for sublist in results for item in sublist]


class SetTransformerDataset(Dataset):
    def __init__(self, 
                 config,
                 channel_groups,
                 hdf5_paths=[],
                 split="pretrain"):

        self.config = config
        self.channel_groups = channel_groups
        channel_like = []
        for modality_type in config["modality_types"]:
            channel_like += channel_groups[modality_type]
        channel_like = set(channel_like)

        if len(hdf5_paths) == 0:
            data_path = config["data_path"]
            hdf5_paths = load_data(config["split_path"])[split]
            hdf5_paths = [os.path.join(data_path, path) for path in hdf5_paths]

        if split in ["pretrain"]:
            random.shuffle(hdf5_paths)

        if config["max_files"]:
            self.hdf5_paths = hdf5_paths[:config["max_files"]]
        else:
            self.hdf5_paths = hdf5_paths

        if split == "validation":
            self.hdf5_paths = self.hdf5_paths #config["val_size"]
        
        self.samples_per_chunk = config["sampling_duration"] * config["sampling_freq"]  #30秒 
        
        # Use multiprocessing to index files in parallel
        self.index_map = index_files(self.hdf5_paths, channel_like, self.samples_per_chunk, config["num_workers"], channel_groups=self.channel_groups, modality_types=config["modality_types"])

        # random.shuffle(self.hdf5_paths)
        # self.index_map = sorted(self.index_map, key=lambda x: (x[1], x[2]))
        self.total_len = len(self.index_map)
        # self.modalities_length = []
        # for modality_type in self.config["modality_types"]:
        #     self.modalities_length.append(self.config[f'{modality_type}_CHANNELS'])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        file_path, dset_names, chunk_start = self.index_map[idx]

        modality_to_channels = {modality_type: [] for modality_type in self.config["modality_types"]}
        for dset_name in dset_names:
            if dset_name in self.channel_groups["BAS"]:
                modality_to_channels["BAS"].append(dset_name)
            # if dset_name in self.channel_groups["RESP"]:
            #     modality_to_channels["RESP"].append(dset_name)
            if dset_name in self.channel_groups["EKG"]:
                modality_to_channels["EKG"].append(dset_name)
            if dset_name in self.channel_groups["EMG"]:
                modality_to_channels["EMG"].append(dset_name)

        target = []
        with h5py.File(file_path, 'r', rdcc_nbytes=1024*1024*500) as hf:
            for modality_type in self.config["modality_types"]:
                # num_channels = self.config[f"{modality_type}_CHANNELS"]
                data = np.zeros((len(modality_to_channels[modality_type]), self.samples_per_chunk))
                ds_names = modality_to_channels[modality_type]
                for idx, ds_name in enumerate(ds_names):
  
                    signal = hf[ds_name][chunk_start:chunk_start+self.samples_per_chunk]
                    data[idx] = signal
                target.append(torch.from_numpy(data).float())
        return target, file_path, dset_names, chunk_start#, self.modalities_length


def collate_fn(batch):
    # Determine the number of modalities

    file_paths = [batch[i][1] for i in range(len(batch))]
    dset_names_list = [batch[i][2] for i in range(len(batch))]
    chunk_starts = [batch[i][3] for i in range(len(batch))]
    batch = [batch[i][0] for i in range(len(batch))]

    num_modalities = len(batch[0])
    
    # Initialize lists to hold padded data and masks for each modality
    padded_batch_list = [[] for _ in range(num_modalities)]
    mask_list = [[] for _ in range(num_modalities)]
    
    # Iterate over each modality
    for modality_index in range(num_modalities):
        test = batch[0][modality_index]
        max_channels = max(data[modality_index].shape[0] for data in batch)
        
        for data in batch:
            modality_data = data[modality_index]
            channels, length = modality_data.shape
            pad_channels = max_channels - channels
            
            # Create mask: 0 for real values, 1 for padded values
            mask = torch.cat((torch.zeros(channels), torch.ones(pad_channels)), dim=0)
            mask_list[modality_index].append(mask)
            
            # Pad the channel dimension
            pad_channel_tensor = torch.zeros((pad_channels, length))
            modality_data = torch.cat((modality_data, pad_channel_tensor), dim=0)
            
            padded_batch_list[modality_index].append(modality_data)
        
        # Stack the padded data and masks for the current modality
        padded_batch_list[modality_index] = torch.stack(padded_batch_list[modality_index])
        mask_list[modality_index] = torch.stack(mask_list[modality_index])

    return padded_batch_list, mask_list, file_paths, dset_names_list, chunk_starts

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import pandas as pd
import os
import random
from tqdm import tqdm
from loguru import logger
import glob
import json
from typing import List, Dict, Tuple

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
        hdf5_paths: List[str] = [],
        split: str = "train"
    ):
        super().__init__()
        
        self.config = config
        self.channel_groups = channel_groups
        self.split = split
        
        self.sampling_freq = config.get("sampling_freq", 128)
        self.epoch_duration = 30
        self.samples_per_epoch = self.epoch_duration * self.sampling_freq  # 3840
        
        # 获取hdf5文件列表
        if not hdf5_paths:
            data_path = config["data_path"]
            split_data = self._load_split(config["split_path"])
            if split in split_data:
                hdf5_paths = [os.path.join(data_path, path) for path in split_data[split]]
            else:
                raise ValueError(f"Split '{split}' not found")
        
        self.hdf5_paths = [f for f in hdf5_paths if os.path.exists(f)]
        if config.get("max_files"):
            self.hdf5_paths = self.hdf5_paths[:config["max_files"]]
        
        # 加载标签文件路径映射
        labels_path = config.get("labels_path", "")
        label_files= glob.glob(os.path.join(labels_path,"**", "*.csv"), recursive=True)
        self.labels_dict = {os.path.basename(f).replace(".csv", ""): f for f in label_files}
        
        # 预计算每个文件的模态到通道映射以及文件长度（合并为一次文件打开）
        logger.info(f"预计算文件通道映射和长度 for {split}...")
        self.file_channel_map, self.file_lengths = self._precompute_channel_maps_and_lengths()
        
        # 构建索引
        self.index_map = []
        pbar = tqdm(self.hdf5_paths, desc=f"Indexing {split} data")
        
        for hdf5_path in pbar:
            study_id = os.path.basename(hdf5_path).replace(".hdf5", "")
            if study_id not in self.labels_dict:
                logger.warning(f"Missing label file for {study_id}")
                continue
            
            label_path = self.labels_dict[study_id]
            labels_df = pd.read_csv(label_path, usecols=['sleep_stage'])
            
            total_samples = self.file_lengths[hdf5_path]
            n_epochs = min(total_samples // self.samples_per_epoch, len(labels_df))
            
            for epoch_idx in range(n_epochs):
                start_sample = epoch_idx * self.samples_per_epoch
                label_val = labels_df.iloc[epoch_idx]['sleep_stage']
                self.index_map.append((hdf5_path, start_sample, label_val))
        
        # 训练集打乱
        if split == "train":
            random.shuffle(self.index_map)
        
        logger.info(f"{split} set: {len(self.index_map)} epochs")
        if self.index_map:
            labels = [item[2] for item in self.index_map]
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"Class distribution: {dict(zip(unique, counts))}")
        
        # ---------- 新增：文件句柄缓存 ----------
        self._open_files = {}  # 按路径缓存已打开的 h5py.File 对象
    
    def _load_split(self, split_path: str) -> Dict:
        with open(split_path, 'r') as f:
            return json.load(f)
    
    def _precompute_channel_maps_and_lengths(self) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, int]]:
        """
        预计算每个HDF5文件的模态通道映射和总样本长度
        返回:
            file_channel_map: {hdf5_path: {modality_type: [channel_names]}}
            file_lengths: {hdf5_path: total_samples}
        """
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
        
        # ---------- 获取或打开文件句柄（缓存） ----------
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
               
            epoch_data.append(data)
            epoch_masks.append(mask)
        
        return epoch_data, epoch_masks, label
    
    def __del__(self):
        """析构时关闭所有打开的文件句柄"""
        for f in self._open_files.values():
            f.close()
        self._open_files.clear()


def collate_fn_cls(batch):
    # 保持不变
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

