import torch
import torch.nn as nn
import sys
import yaml
import json
import pickle
from typing import Any
import numpy as np
from einops import rearrange


def count_parameters(model):
    def count_recursive(module):
        total_params = 0
        num_layers = 0
        
        for child in module.children():
            child_layers, child_params = count_recursive(child)
            num_layers += child_layers
            total_params += child_params
        
        if list(module.children()) == []:  # if module has no children, it's a layer
            num_layers = 1
            for param in module.parameters():
                total_params += param.numel()
        
        return num_layers, total_params
    
    return count_recursive(model)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def instantiate_model(model_name, in_channel):
    model_class = getattr(sys.modules[__name__], model_name)
    return model_class(in_channel=in_channel)


def save_data(data: Any, filename: str) -> None:
    """
    Save data to a file in either pickle, JSON, YAML, or NPY format based on the file extension.

    Parameters:
    - data: The data to save.
    - filename: The name of the file to save the data to. Should have .pickle, .pkl, .p, .json, .yaml, or .npy extension.
    """
    if filename.endswith(('.pkl', '.pickle', '.p')):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    elif filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    elif filename.endswith('.yaml'):
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    elif filename.endswith('.npy'):
        np.save(filename, data)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, .p, .json, .yaml, or .npy")


def load_data(filename: str) -> Any:
    """
    Load data from a file in either pickle, JSON, YAML, or NPY format based on the file extension.

    Parameters:
    - filename: The name of the file to load the data from. Should have .pickle, .pkl, .p, .json, .yaml, or .npy extension.

    Returns:
    - The loaded data.
    """
    if filename.endswith(('.pkl', '.pickle', '.p')):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    elif filename.endswith('.yaml'):
        with open(filename, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    elif filename.endswith('.npy'):
        return np.load(filename, allow_pickle=True)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, .p, .json, .yaml, or .npy")

def create_causal_mask(seq_len):
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    return causal_mask



class DeepSpeedWrapper:
    """DeepSpeed包装器，统一接口"""
    def __init__(self, engine, is_deepspeed=True, gradient_accumulation_steps=1):
        self.engine = engine
        self.is_deepspeed = is_deepspeed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accumulation_step = 0
    
    def train(self):
        """设置训练模式"""
        if self.is_deepspeed:
            self.engine.train()
        else:
            self.engine.module.train()
    
    def eval(self):
        """设置评估模式"""
        if self.is_deepspeed:
            self.engine.eval()
        else:
            self.engine.module.eval()
    
    def backward(self, loss):
        """反向传播，支持梯度累积"""
        if self.is_deepspeed:
            # DeepSpeed自动处理梯度累积
            self.engine.backward(loss)
        else:
            # 手动处理梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            self.accumulation_step += 1
    
    def step(self):
        """优化器步进，支持梯度累积"""
        if self.is_deepspeed:
            self.engine.step()
        else:
            if self.accumulation_step >= self.gradient_accumulation_steps:
                self.engine.optimizer.step()
                self.engine.optimizer.zero_grad()
                self.accumulation_step = 0
    
    def zero_grad(self):
        """清零梯度"""
        if not self.is_deepspeed:
            self.engine.optimizer.zero_grad()
            self.accumulation_step = 0
    
    @property
    def module(self):
        """获取模型"""
        if self.is_deepspeed:
            return self.engine.module
        else:
            return self.engine.module
    
    @property
    def device(self):
        """获取设备"""
        if self.is_deepspeed:
            return self.engine.device
        else:
            return next(self.engine.module.parameters()).device
    
    @property
    def local_rank(self):
        """获取local_rank"""
        if self.is_deepspeed:
            return self.engine.local_rank
        else:
            return 0
    
    def get_lr(self):
        """获取学习率"""
        if self.is_deepspeed:
            # DeepSpeed中获取学习率的方式
            try:
                return [self.engine.get_lr()]
            except:
                return [group['lr'] for group in self.engine.optimizer.param_groups]
        else:
            return [group['lr'] for group in self.engine.optimizer.param_groups]


