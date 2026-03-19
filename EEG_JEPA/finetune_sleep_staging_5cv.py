import os
import json
import pickle
import numpy as np
import h5py
import pandas as pd
import tqdm
from loguru import logger
import glob
import random
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
import yaml
from datetime import datetime
from sklearn.metrics import f1_score
import deepspeed
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from models.models import SetTransformerForSleepStaging
from models.dataset5 import SleepStagingDataset, collate_fn_cls

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def build_engine(model, ds_config, local_rank, config_params=None):
    if isinstance(ds_config, str):
        with open(ds_config, 'r') as f:
            ds_config = json.load(f)
    if config_params is not None:
        if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
            ds_config["optimizer"]["params"]["lr"] = config_params.get("lr", 1e-4)
            ds_config["optimizer"]["params"]["weight_decay"] = config_params.get("weight_decay", 1e-4)
        if "gradient_accumulation_steps" in ds_config:
            ds_config["gradient_accumulation_steps"] = config_params.get(
                "gradient_accumulation_steps", ds_config["gradient_accumulation_steps"]
            )
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    class Args:
        def __init__(self, local_rank):
            self.local_rank = local_rank
    args = Args(local_rank)
    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config
    )
    return engine

def run_iter(batch, model_engine, criterion, device, return_logits=False):
    data, masks, labels = batch
    data = [d.to(device, dtype=torch.float16) for d in data]
    masks = [m.to(device, dtype=torch.bool) for m in masks]
    labels = labels.to(device)
    logits = model_engine(data, masks)
    logits = logits.float()
    loss = criterion(logits, labels)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    if return_logits:
        return loss, correct, total, logits
    else:
        return loss, correct, total

def train_one_epoch(num_epochs,model_engine, dataloader, criterion, epoch, device, local_rank, writer=None, global_step=0):
    model_engine.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    is_main = (local_rank in [-1, 0])
    if is_main:
        pbar = tqdm.tqdm(total=len(dataloader), desc=f"Train Epoch {epoch+1}/{num_epochs}")
    else:
        pbar = None
    for batch_idx, batch in enumerate(dataloader):
        loss, correct, num = run_iter(batch, model_engine, criterion, device)
        if torch.isnan(loss) or torch.isinf(loss):
            if is_main:
                logger.warning(f"Loss is {loss.item()} at batch {batch_idx}, skipping this batch.")
            continue  # 跳过该批次
        model_engine.backward(loss)
        model_engine.step()
        batch_loss = loss.item()
        total_loss += batch_loss * num
        total_correct += correct
        total_samples += num
        if is_main:
            pbar.update()
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{100.0 * total_correct / total_samples:.2f}%'
            })
            if writer is not None:
                writer.add_scalar('train/loss', batch_loss, global_step)
                if batch_idx == 0:
                    lr = model_engine.get_lr()[0] if model_engine.get_lr() else 0
                    writer.add_scalar('train/lr', lr, global_step)
        global_step += 1
    if is_main:
        pbar.close()
    if torch.distributed.is_initialized():
        loss_tensor = torch.tensor(total_loss).to(device)
        correct_tensor = torch.tensor(total_correct).to(device)
        samples_tensor = torch.tensor(total_samples).to(device)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(samples_tensor, op=torch.distributed.ReduceOp.SUM)
        total_loss = loss_tensor.item()
        total_correct = correct_tensor.item()
        total_samples = samples_tensor.item()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy, global_step

@torch.no_grad()
def validate(num_epochs,model_engine, dataloader, criterion, epoch, device, local_rank, writer=None, val_step=0):
    model_engine.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    is_main = (local_rank in [-1, 0])
    local_preds = []
    local_labels = []
    if is_main:
        pbar = tqdm.tqdm(total=len(dataloader), desc=f"Val Epoch {epoch+1}/{num_epochs}")
    else:
        pbar = None
    for batch_idx, batch in enumerate(dataloader):
        loss, correct, num, logits = run_iter(batch, model_engine, criterion, device, return_logits=True)
        batch_loss = loss.item()
        total_loss += batch_loss * num
        total_correct += correct
        total_samples += num
        preds = torch.argmax(logits, dim=1)
        local_preds.append(preds.cpu())
        local_labels.append(batch[2].cpu())
        if is_main:
            pbar.update()
            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
            if writer is not None:
                writer.add_scalar('val/loss', batch_loss, val_step)
        val_step += 1
    if is_main:
        pbar.close()
    if torch.distributed.is_initialized():
        loss_tensor = torch.tensor(total_loss).to(device)
        correct_tensor = torch.tensor(total_correct).to(device)
        samples_tensor = torch.tensor(total_samples).to(device)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(samples_tensor, op=torch.distributed.ReduceOp.SUM)
        total_loss = loss_tensor.item()
        total_correct = correct_tensor.item()
        total_samples = samples_tensor.item()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    # 收集所有进程的预测和标签
    if torch.distributed.is_initialized():
        local_preds = torch.cat(local_preds, dim=0).to(device)
        local_labels = torch.cat(local_labels, dim=0).to(device)
        world_size = torch.distributed.get_world_size()
        local_size = torch.tensor(local_preds.size(0), device=device, dtype=local_preds.dtype)
        all_sizes = [torch.zeros(1, device=device, dtype=local_preds.dtype) for _ in range(world_size)]
        torch.distributed.all_gather(all_sizes, local_size)
        all_preds_gathered = [torch.zeros(int(s.item()), dtype=local_preds.dtype, device=device) for s in all_sizes]
        all_labels_gathered = [torch.zeros(int(s.item()), dtype=local_labels.dtype, device=device) for s in all_sizes]
        torch.distributed.all_gather(all_preds_gathered, local_preds)
        torch.distributed.all_gather(all_labels_gathered, local_labels)
        if is_main:
            all_preds = torch.cat(all_preds_gathered, dim=0).cpu().numpy()
            all_labels = torch.cat(all_labels_gathered, dim=0).cpu().numpy()
            f1_weighted = f1_score(all_labels, all_preds, average='weighted')
            class_f1 = f1_score(all_labels, all_preds, average=None)
            for i, f1_val in enumerate(class_f1):
                logger.info(f"Class {i} F1: {f1_val:.4f}")
                if writer is not None:
                    writer.add_scalar(f'val/f1_class_{i}', f1_val, epoch+1)
        else:
            f1_weighted = 0.0
    else:
        all_preds = torch.cat(local_preds, dim=0).numpy()
        all_labels = torch.cat(local_labels, dim=0).numpy()
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        class_f1 = f1_score(all_labels, all_preds, average=None)
        for i, f1_val in enumerate(class_f1):
            logger.info(f"Class {i} F1: {f1_val:.4f}")
            if writer is not None:
                writer.add_scalar(f'val/f1_class_{i}', f1_val, epoch+1)
    return avg_loss, accuracy, f1_weighted, val_step


# ------------------------------
# 单折训练函数
# ------------------------------
def train_fold(
    fold,
    config,
    channel_groups,
    pretrained_path,
    freeze_backbone,
    deepspeed_config,
    train_subset,
    val_subset,
    output_root,
    device,
    local_rank,
):
    """训练一折模型"""
    fold_output_dir = os.path.join(output_root, f"fold_{fold+1}")
    if local_rank in [-1, 0]:
        os.makedirs(fold_output_dir, exist_ok=True)
        logger.add(os.path.join(fold_output_dir, "training.log"))
        # logger.info(f"Fold {fold+1} 输出目录: {fold_output_dir}")

    # 创建模型
    model = SetTransformerForSleepStaging(
        in_channels=config['in_channels'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        latent_dim=config['latent_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        jepa_depth=config['jepa_depth'],
        pooling_head=config['pooling_head'],
        dropout=config['dropout'],
        max_seq_length=config.get('max_seq_length', 128),
        num_sleep_stages=config.get('num_sleep_stages', 5),
        freeze_backbone=freeze_backbone,
        num_patches=config["num_patches"]
    )

    # 加载预训练权重
    if local_rank in [-1, 0]:
        logger.info(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint
        # 移除 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        # 过滤分类头和位置编码
        pretrained_dict = {}
        for k, v in state_dict.items():
            if 'pos_embed' in k:
                logger.info(f"跳过位置编码: {k}")
                continue
            if k in model.state_dict():
                if v.shape == model.state_dict()[k].shape:
                    pretrained_dict[k] = v
                else:
                    logger.warning(f"形状不匹配，跳过: {k} (预训练: {v.shape}, 当前: {model.state_dict()[k].shape})")
        logger.info(f"成功加载 {len(pretrained_dict)} 个权重")
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        if missing:
            logger.warning(f"缺失的键（未加载）: {missing}")
        if unexpected:
            logger.warning(f"意外的键（预训练中有但模型没有）: {unexpected}")
    else:
        pass

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if local_rank in [-1, 0]:
        logger.info(f"总参数: {total_params/1e6:.2f}M")
        logger.info(f"可训练参数: {trainable_params/1e6:.2f}M")
    
    #统计数据

    logger.info(f"Fold {fold+1} 训练集: {len(train_subset)}")
    logger.info(f"Fold {fold+1} 验证集: {len(val_subset)}")
    # 计算类别权重（基于当前折的训练集）
    if local_rank in [-1, 0]:
        train_labels = [train_subset.dataset.labels[i] for i in train_subset.indices]
        class_counts = np.bincount(train_labels, minlength=config.get('num_sleep_stages', 5))
        class_weights = 1.0 / np.sqrt(class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)  # 直接移动到 GPU
    else:
        class_weights_tensor = None
  
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    if local_rank in [-1, 0]:
        logger.info(f"Fold {fold+1} 类别权重: {class_weights_tensor}")

    # DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn_cls,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn_cls,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # DeepSpeed 引擎
    config_params = {
        "lr": config.get("lr", 1e-4),
        "weight_decay": config.get("weight_decay", 1e-4),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1)
    }
    model_engine = build_engine(model, deepspeed_config, local_rank, config_params)

    # TensorBoard
    if local_rank in [-1, 0] and config.get("use_tensorboard", True):
        tensorboard_dir = os.path.join(fold_output_dir, "tensorboard")
        train_writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, "train"))
        val_writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, "val"))
        logger.info(f"TensorBoard 日志保存在: {tensorboard_dir}")
    else:
        train_writer = None
        val_writer = None

    # 训练循环
    best_val_accuracy = 0.0
    num_epochs = config.get('epochs', 50)
    global_step = 0
    val_step = 0

    for epoch in range(num_epochs):
        train_loss, train_acc, global_step = train_one_epoch(num_epochs,
            model_engine, train_loader, criterion, epoch, device, local_rank,
            writer=train_writer, global_step=global_step
        )
        if local_rank in [-1, 0]:
            logger.info(f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} 训练: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

        val_loss, val_acc, val_f1, val_step = validate(num_epochs,
            model_engine, val_loader, criterion, epoch, device, local_rank,
            writer=val_writer, val_step=val_step
        )
        if local_rank in [-1, 0]:
            logger.info(f"Fold {fold+1} Epoch {epoch+1}{num_epochs} 验证: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.4f}")

        if model_engine.lr_scheduler is not None:
            model_engine.lr_scheduler.step()

        if local_rank in [-1, 0] and val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_path = os.path.join(fold_output_dir, "best_model.pth")
            torch.save(model_engine.module.state_dict(), best_path)
            logger.info(f"Saved best model with accuracy {val_acc:.2f}% at {best_path}")
            torch.cuda.empty_cache()

    # 最终保存
    if local_rank in [-1, 0]:
        final_path = os.path.join(fold_output_dir, "final_model.pth")
        torch.save(model_engine.module.state_dict(), final_path)
        logger.info(f"Final model saved at {final_path}")
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()

    return best_val_accuracy


# ------------------------------
# 主入口：5折交叉验证
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="5折交叉验证训练脚本")
    parser.add_argument('--config', type=str, default='/mnt/DATA/poolab/jepa_dataset/EEG_JEPA/configs/config_finetune_sleep_events.yaml',
                        help='YAML配置文件路径')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='分布式训练的本地进程编号（由 DeepSpeed 自动传入）')
    args = parser.parse_args()

    # 从 YAML 加载配置
    config = load_config(args.config)
    local_rank = args.local_rank

    # 从配置中读取必要的路径和参数
    channel_groups_path = config.get('channel_groups_path')
    pretrained_path = config['pretrained_path']          
    freeze_backbone = config.get('freeze_backbone')
    deepspeed_config = config['deepspeed_config']       
    num_folds = config.get('num_folds', 5)

    with open(channel_groups_path, 'r') as f:
        channel_groups = json.load(f)

    set_seed()
    full_dataset = SleepStagingDataset(config, channel_groups, split="all")
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    fold_size = num_samples // num_folds
    fold_accuracies = []

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_path = os.path.join(config.get("output_dir", "./results"),f"cv_results_{current_timestamp}")
    os.makedirs(fold_path, exist_ok=True)
    logger.info(f"输出目录: {fold_path}")

    for fold in range(num_folds):
        if local_rank in [-1, 0]:
            logger.info(f"========== Fold {fold+1}/{num_folds} ==========")
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < num_folds - 1 else num_samples
        logger.info(f"Fold {fold+1} 验证集索引: {start}_{end}/{num_samples}")
        val_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        set_seed(42 + fold)
        
        acc = train_fold(
            fold=fold,
            config=config,
            channel_groups=channel_groups,
            pretrained_path=pretrained_path,
            freeze_backbone=freeze_backbone,
            deepspeed_config=deepspeed_config,
            train_subset=train_subset,
            val_subset=val_subset,
            output_root=fold_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            local_rank=local_rank,
        )
        fold_accuracies.append(acc)

    if local_rank in [-1, 0]:
        logger.info(f"Cross-validation results: {fold_accuracies}")
        logger.info(f"Mean accuracy: {np.mean(fold_accuracies):.2f}%, Std: {np.std(fold_accuracies):.2f}%")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()