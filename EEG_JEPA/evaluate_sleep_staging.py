import argparse
import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report
import yaml

# 添加项目路径以便导入自定义模块
sys.path.append("../")
from models.models import SetTransformerForSleepStaging
from models.dataset import SleepStagingDataset, collate_fn_cls


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, channel_groups, weight_path, device, freeze_backbone=False):
    """
    根据配置和权重文件创建并加载模型
    """
    model = SetTransformerForSleepStaging(
        in_channels=config['in_channels'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        latent_dim=config['latent_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        jepa_depth=config.get('jepa_depth', 6),
        pooling_head=config['pooling_head'],
        dropout=config['dropout'],
        max_seq_length=config.get('max_seq_length', 128),
        num_sleep_stages=config.get('num_sleep_stages', 5),
        freeze_backbone=freeze_backbone,   # 推理时通常不冻结，但为了保持结构一致
        num_patches=config["num_patches"]
    )
    
    # 加载权重
    state_dict = torch.load(weight_path, map_location='cpu')
    # 如果保存的是完整模型（含 module.），则去掉前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # 严格加载，确保所有参数匹配
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"警告：缺失的键：{missing}")
    if unexpected:
        print(f"警告：意外的键：{unexpected}")
    
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def inference(model, dataloader, device):
    """
    在给定 DataLoader 上运行推理，返回所有预测和真实标签
    """
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        data, masks, labels = batch
        # 将数据移至设备
        data = [d.to(device, dtype=torch.float16 if device.type=='cuda' else torch.float32) for d in data]
        masks = [m.to(device, dtype=torch.bool) for m in masks]
        labels = labels.to(device)
        
        # 前向传播
        logits = model(data, masks)  # 假设模型返回 logits
        preds = torch.argmax(logits, dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵并计算每个类别的准确率（召回率）
    """
    cm = confusion_matrix(y_true, y_pred)
    # 归一化按行（真实类别）得到召回率
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized by True Class)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存至：{save_path}")
    plt.show()
    
    # 计算每个类别的准确率（对角线值）
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, name in enumerate(class_names):
        print(f"{name:5s} 准确率: {per_class_accuracy[i]:.4f} ({cm.diagonal()[i]}/{cm.sum(axis=1)[i]})")
    
    return per_class_accuracy, cm


def main(args):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    
    # 加载配置和通道组
    config = load_config(args.config)
    with open(args.channel_groups, 'r') as f:
        channel_groups = json.load(f)
    
    # 创建测试数据集
    test_dataset = SleepStagingDataset(
        config,
        channel_groups,
        split="test"   # 假设配置中有 test 划分
    )
    print(f"测试集样本数：{len(test_dataset)}")
    
    # 创建 DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_cls,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # 加载模型
    model = load_model(config, channel_groups, args.weight, device, freeze_backbone=False)
    print("模型加载成功，开始推理...")
    
    # 推理
    preds, labels = inference(model, test_loader, device)
    
    # 计算总体指标
    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    print(f"\n总体准确率: {acc:.4f}")
    print(f"平衡准确率: {balanced_acc:.4f}")
    print("\n分类报告：")
    class_names = ["Wake", "N1", "N2", "N3", "REM"]  # 根据 num_sleep_stages 调整
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(labels, preds, class_names, save_path=args.output_fig)
    
    # 可选：保存预测结果到文件
    if args.save_pred:
        np.savez(args.save_pred, predictions=preds, ground_truth=labels)
        print(f"预测结果已保存至：{args.save_pred}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="睡眠分期模型推理与评估")
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径 (YAML)")
    parser.add_argument("--channel_groups", type=str, required=True,
                        help="通道组 JSON 文件路径")
    parser.add_argument("--weight", type=str, required=True,
                        help="训练好的模型权重文件 (.pth)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="推理批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 工作进程数")
    parser.add_argument("--output_fig", type=str, default=None,
                        help="混淆矩阵保存路径（如 confusion_matrix.png）")
    parser.add_argument("--save_pred", type=str, default=None,
                        help="保存预测结果到 .npz 文件")
    args = parser.parse_args()
    
    main(args)