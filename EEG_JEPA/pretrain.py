import yaml
import torch
from loguru import logger
import os
import sys
sys.path.append("../")
from utils import *
from models.dataset import SetTransformerDataset, collate_fn
from models.models0 import SetTransformer
import click
import time
import math
import datetime
import numpy as np
import tqdm
import wandb
from torch.optim import AdamW
import torch.nn.functional as F


def compute_masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    计算mask区域的Smooth L1损失
    
    Args:
        pred: 预测特征 [B, C, T_z]
        target: 目标特征 [B, C, T_z]
        mask: mask矩阵 [B, T_z]
        
    Returns:
        mask区域的Smooth L1损失
    """
    # 扩展mask到特征维度 [B, T_z] -> [B, C, T_z]
    mask_expanded = mask.unsqueeze(1).expand_as(pred)
    
    # 计算每个位置的Smooth L1损失
    loss_per_element = F.smooth_l1_loss(pred, target, reduction='none')
    
    # 仅计算mask区域的损失
    masked_loss = loss_per_element * mask_expanded
    
    # 归一化：除以mask位置的数量
    num_masked = mask_expanded.sum() + 1e-8
    return masked_loss.sum() / num_masked


def run_iter(batch, model, device, mask_ratio=0.25):
    """
    JEPA模型的训练迭代
    
    Args:
        batch: 包含数据和mask的批次
        model: JEPA模型
        device: 计算设备
        mask_ratio: mask比例
        
    Returns:
        平均损失和损失字典
    """
    # 开始计时
    start_time = time.time()
    
    # 解包批次数据
    batch_data, mask_list, *_ = batch
    (bas, ekg, emg) = batch_data
    (mask_bas, mask_ekg, mask_emg) = mask_list

    # 将数据移动到设备
    bas = bas.to(device)
    ekg = ekg.to(device)
    emg = emg.to(device)
    mask_bas = mask_bas.to(device)
    mask_ekg = mask_ekg.to(device)
    mask_emg = mask_emg.to(device)
    
    # 获取批次信息
    B, C, T = bas.shape  # 批次大小, 通道数, 序列长度
    
    # 打印批次信息
    print(f"\n{'='*60}")
    print(f"批次信息: B={B}, C={C}, T={T}")
    print(f"设备: {device}")
    print(f"{'='*60}")
    
    # 模型前向传播
    print("开始模型推理...")
    model_start_time = time.time()
    
    bas_result = model(bas, mask_bas)
    ekg_result = model(ekg, mask_ekg)
    emg_result = model(emg, mask_emg)
    
    model_time = time.time() - model_start_time
    print(f"模型推理完成，耗时: {model_time:.3f}秒")
    
    # 解包结果
    bas_context, bas_pred, bas_mask, bas_target = bas_result
    ekg_context, ekg_pred, ekg_mask, ekg_target = ekg_result
    emg_context, emg_pred, emg_mask, emg_target = emg_result
    
    # 验证形状
    print(f"\n特征形状验证:")
    print(f"BAS: pred={bas_pred.shape}, target={bas_target.shape}, mask={bas_mask.shape}")
    print(f"EKG: pred={ekg_pred.shape}, target={ekg_target.shape}, mask={ekg_mask.shape}")
    print(f"EMG: pred={emg_pred.shape}, target={emg_target.shape}, mask={emg_mask.shape}")
    
    # 计算各模态的mask区域Smooth L1损失
    print(f"\n{'='*60}")
    print("计算各模态损失...")
    print(f"{'='*60}")
    
    loss_start_time = time.time()
    
    # BAS模态损失
    bas_loss = compute_masked_smooth_l1(bas_pred, bas_target, bas_mask)
    print(f"BAS 损失: {bas_loss.item():.6f}")
    
    # EKG模态损失
    ekg_loss = compute_masked_smooth_l1(ekg_pred, ekg_target, ekg_mask)
    print(f"EKG 损失: {ekg_loss.item():.6f}")
    
    # EMG模态损失
    emg_loss = compute_masked_smooth_l1(emg_pred, emg_target, emg_mask)
    print(f"EMG 损失: {emg_loss.item():.6f}")
    
    loss_time = time.time() - loss_start_time
    print(f"损失计算完成，耗时: {loss_time:.3f}秒")
    
    # 计算平均损失
    total_loss = (bas_loss + ekg_loss + emg_loss) / 3.0
    
    # 计算额外评估指标
    with torch.no_grad():
        # 计算各模态的mask比例
        bas_mask_ratio = bas_mask.float().mean()
        ekg_mask_ratio = ekg_mask.float().mean()
        emg_mask_ratio = emg_mask.float().mean()
        
        # 计算各模态的全序列Smooth L1损失（仅用于监控）
        def compute_full_smooth_l1(pred, target):
            return F.smooth_l1_loss(pred, target)
        
        bas_full_loss = compute_full_smooth_l1(bas_pred, bas_target)
        ekg_full_loss = compute_full_smooth_l1(ekg_pred, ekg_target)
        emg_full_loss = compute_full_smooth_l1(emg_pred, emg_target)
        
        # 计算各模态的信号统计
        def compute_signal_stats(pred, target):
            signal_power = torch.mean(target ** 2)
            noise_power = torch.mean((pred - target) ** 2)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
            return snr
        
        bas_snr = compute_signal_stats(bas_pred, bas_target)
        ekg_snr = compute_signal_stats(ekg_pred, ekg_target)
        emg_snr = compute_signal_stats(emg_pred, emg_target)
    
    # 总耗时
    total_time = time.time() - start_time
    
    # 打印详细统计信息
    print(f"\n{'='*60}")
    print("详细统计信息:")
    print(f"{'='*60}")
    print(f"平均损失: {total_loss.item():.6f}")
    print(f"\n各模态详细统计:")
    print(f"{'-'*40}")
    print(f"BAS: Mask比例={bas_mask_ratio.item():.3f}, 全序列损失={bas_full_loss.item():.6f}, SNR={bas_snr.item():.2f} dB")
    print(f"EKG: Mask比例={ekg_mask_ratio.item():.3f}, 全序列损失={ekg_full_loss.item():.6f}, SNR={ekg_snr.item():.2f} dB")
    print(f"EMG: Mask比例={emg_mask_ratio.item():.3f}, 全序列损失={emg_full_loss.item():.6f}, SNR={emg_snr.item():.2f} dB")
    print(f"{'-'*40}")
    print(f"\n时间统计:")
    print(f"模型推理: {model_time:.3f}秒")
    print(f"损失计算: {loss_time:.3f}秒")
    print(f"总耗时: {total_time:.3f}秒")
    print(f"{'='*60}")
    
    return total_loss



def validate_model(model, val_dataloader, device, num_modalities, config):
    """
    在验证集上评估模型
    """
    model.eval()
    
    total_loss = 0.
    total_reconstruction_loss = 0.
    total_predictive_loss = 0.
    total_snr = 0.
    total_corr = 0.
    total_n = 0
    
    with torch.no_grad():
        with tqdm.tqdm(total=len(val_dataloader), desc="验证") as pbar:
            for batch in val_dataloader:
                results = run_iter(
                    batch, model, device, 
                    mask_ratio=config.get("mask_ratio", 0.25)
                )
                
                loss = results['total_loss']
       
                batch_size_actual = batch[0][0].size(0)
                total_loss += loss.item() * batch_size_actual
                total_n += batch_size_actual
                
                pbar.update()
    
    # 计算平均指标
    avg_loss = total_loss / total_n if total_n > 0 else 0
    avg_rec_loss = total_reconstruction_loss / total_n if total_n > 0 else 0
    avg_pred_loss = total_predictive_loss / total_n if total_n > 0 else 0
    avg_snr = total_snr / total_n if total_n > 0 else 0
    avg_corr = total_corr / total_n if total_n > 0 else 0
    
    return avg_loss, avg_rec_loss, avg_pred_loss, avg_snr, avg_corr


@click.command("pretrain")
@click.option("--config_path", type=str, default=r'/mnt/nas1/Neuromodulation/human/code/EEG_JEPA/configs/config_set_transformer_contrastive.yaml')
@click.option("--channel_groups_path", type=str, default=r'/mnt/nas1/Neuromodulation/human/code/EEG_JEPA/configs/channel_groups.json')
@click.option("--checkpoint_path", type=str, default=None)
@click.option("--use_wandb", type=bool, default=False)
def pretrain(
    config_path, 
    channel_groups_path, 
    checkpoint_path,
    use_wandb
):
    """
    预训练主函数
    """
    
    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if checkpoint_path:
        output = checkpoint_path
        logger.info("加载保存的配置")
        config_path = os.path.join(output, "config.json")
        config = load_config(config_path)
    else:
        output = os.path.join(
            config["save_path"], 
            f"{config['model']}//{config['embed_dim']}_patch_{config['patch_size']}_{current_timestamp}"
        )
        os.makedirs(output, exist_ok=True)

    # 保存配置到输出目录
    save_data(config, os.path.join(output, "config.json"))

    data_path = config["data_path"]
    modality_types = config["modality_types"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    weight_decay = config["weight_decay"]
    in_channels = config["in_channels"]
    sampling_freq = config["sampling_freq"]
    patch_size = config["patch_size"]
    embed_dim = config["embed_dim"]
    latent_dim = config.get("latent_dim")  # 潜在空间维度
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    pooling_head = config["pooling_head"]
    log_interval = config["log_interval"]
    model_name = config["model"]
    
    # 新增JEPA相关参数
    use_jepa = config.get("use_jepa", True)
    jepa_predictive_steps = config.get("jepa_predictive_steps", 1)
    mask_ratio = config.get("mask_ratio", 0.25)

    # 初始化WandB
    if use_wandb:
        wandb.init(
            project="PSG-JEPA",
            name=f"jepa_{current_timestamp}",
            config=config
        )
        os.environ['WANDB_DIR'] = output

    logger.info(f"输出路径: {output}")
    logger.info(f"模态类型: {modality_types}")
    logger.info(f"使用JEPA: {use_jepa}")
    logger.info(f"掩码比例: {mask_ratio}")
    logger.info(f"批大小: {batch_size}; 工作线程数: {num_workers}")
    logger.info(f"权重衰减: {weight_decay}; 学习率: {lr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    num_modalities = len(modality_types)

    start = time.time()
    
    # 加载数据集
    logger.info("加载训练集...")
    train_dataset = SetTransformerDataset(config, channel_groups, split="pretrain")
    
    logger.info("加载验证集...")
    val_dataset = SetTransformerDataset(config, channel_groups, split="validation")
    
    logger.info(f"数据集加载时间: {time.time() - start:.1f}秒")
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True
    )
    
    logger.info(f"模型类: {model_name}")

    # 创建模型
    model = SetTransformer(
        in_channels=in_channels,
        patch_size=patch_size,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        pooling_head=pooling_head,
        dropout=dropout,
        sampling_freq=sampling_freq
    )
    
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    total_layers, total_params = count_parameters(model)
    logger.info(f'可训练参数: {total_params / 1e6:.2f} 百万')
    logger.info(f'层数: {total_layers}')

    # 优化器
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, 
        T_max=epochs,
        eta_min=1e-6
    )

    epoch_resume = 0
    best_val_loss = math.inf

    # 加载检查点
    if checkpoint_path and os.path.isfile(os.path.join(checkpoint_path, "checkpoint.pt")):
        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optim.load_state_dict(checkpoint["optim_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_dict"])
        epoch_resume = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_loss"]
        logger.info(f"从第 {epoch_resume} 轮恢复训练，最佳验证损失: {best_val_loss:.4f}")
    else:
        logger.info("从头开始训练")
    
    # 创建日志文件
    os.makedirs(os.path.join(output, "log"), exist_ok=True)
    log_file = os.path.join(output, "log", f"{current_timestamp}.tsv")
    
    with open(log_file, "w") as f:
        f.write("Epoch\tSplit\tTotal Loss\tReconstruction Loss\tPredictive Loss\tSNR (dB)\tReconstruction Correlation\tLR\n")
        f.flush()
        
        count_iter = 1
        
        for epoch in range(epoch_resume, epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"{'='*50}")
            
            # ========== 训练阶段 ==========
            model.train()      
            total_train_loss = 0.
            with tqdm.tqdm(total=len(train_loader), desc=f"训练 Epoch {epoch+1}") as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    # 训练迭代
                    results = run_iter(
                        batch, model, device, mask_ratio
                    )
                    
                    loss = results['total_loss']
                  
                    # 反向传播和优化
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optim.step()
                    
                    batch_size_actual = batch[0][0].size(0)
                    total_train_loss += loss.item() * batch_size_actual
                    total_train_n += batch_size_actual
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'Loss': f"{total_train_loss/total_train_n:.4f}",
            
                    })
                    
                    # 定期记录到wandb
                    if use_wandb and count_iter % log_interval == 0:
                        wandb.log({
                            "train_loss": loss.item(),
                            "lr": optim.param_groups[0]['lr']
                        }, step=count_iter)
                    
                    # 定期保存检查点
                    if (count_iter % config.get("save_iter", 500) == 0):
                        logger.info(f"迭代 {count_iter}: 保存检查点...")
                        save = {
                            "epoch": epoch,
                            "optim_dict": optim.state_dict(),
                            "scheduler_dict": scheduler.state_dict(),
                            "best_loss": best_val_loss,
                            "loss": loss.item(),
                            "state_dict": model.state_dict()
                        }
                        torch.save(save, os.path.join(output, "checkpoint.pt"))
                    
                    count_iter += 1
                    pbar.update()
            
            # 计算训练平均指标
            avg_train_loss = total_train_loss / total_train_n if total_train_n > 0 else 0
            avg_train_rec_loss = total_train_rec_loss / total_train_n if total_train_n > 0 else 0
            avg_train_pred_loss = total_train_pred_loss / total_train_n if total_train_n > 0 else 0
            avg_train_snr = total_train_snr / total_train_n if total_train_n > 0 else 0
            avg_train_corr = total_train_corr / total_train_n if total_train_n > 0 else 0
            
            # ========== 验证阶段 ==========
            logger.info("开始验证...")
            avg_val_loss, avg_val_rec_loss, avg_val_pred_loss, avg_val_snr, avg_val_corr = validate_model(
                model, val_loader, device, num_modalities, config
            )
            
            # 记录到日志文件
            f.write(f"{epoch+1}\tTrain\t{avg_train_loss:.5f}\t{avg_train_rec_loss:.5f}\t{avg_train_pred_loss:.5f}\t{avg_train_snr:.2f}\t{avg_train_corr:.4f}\t{optim.param_groups[0]['lr']:.6f}\n")
            f.write(f"{epoch+1}\tValidation\t{avg_val_loss:.5f}\t{avg_val_rec_loss:.5f}\t{avg_val_pred_loss:.5f}\t{avg_val_snr:.2f}\t{avg_val_corr:.4f}\t{optim.param_groups[0]['lr']:.6f}\n")
            f.flush()
            
            # 记录到wandb
            if use_wandb:
                wandb.log({
                    "epoch": epoch+1,
                    "avg_train_loss": avg_train_loss,
                    "avg_train_rec_loss": avg_train_rec_loss,
                    "avg_train_pred_loss": avg_train_pred_loss,
                    "avg_train_snr": avg_train_snr,
                    "avg_train_corr": avg_train_corr,
                    "avg_val_loss": avg_val_loss,
                    "avg_val_rec_loss": avg_val_rec_loss,
                    "avg_val_pred_loss": avg_val_pred_loss,
                    "avg_val_snr": avg_val_snr,
                    "avg_val_corr": avg_val_corr,
                    "lr": optim.param_groups[0]['lr']
                })
            
            # 打印训练和验证结果
            logger.info(f"训练结果 - Loss: {avg_train_loss:.4f}, Rec: {avg_train_rec_loss:.4f}, "
                       f"Pred: {avg_train_pred_loss:.4f}, SNR: {avg_train_snr:.2f}dB, "
                       f"Corr: {avg_train_corr:.4f}")
            
            logger.info(f"验证结果 - Loss: {avg_val_loss:.4f}, Rec: {avg_val_rec_loss:.4f}, "
                       f"Pred: {avg_val_pred_loss:.4f}, SNR: {avg_val_snr:.2f}dB, "
                       f"Corr: {avg_val_corr:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save = {
                    "epoch": epoch,
                    "optim_dict": optim.state_dict(),
                    "scheduler_dict": scheduler.state_dict(),
                    "best_loss": best_val_loss,
                    "loss": avg_val_loss,
                    "state_dict": model.state_dict()
                }
                torch.save(save, os.path.join(output, "best.pt"))
                logger.info(f"新的最佳模型保存在轮次 {epoch+1}, 验证损失: {best_val_loss:.4f}")
            
            # 更新学习率
            scheduler.step()
            
            # 定期保存检查点
            if (epoch + 1) % config.get("checkpoint_interval", 5) == 0:
                save = {
                    "epoch": epoch,
                    "optim_dict": optim.state_dict(),
                    "scheduler_dict": scheduler.state_dict(),
                    "best_loss": best_val_loss,
                    "loss": avg_val_loss,
                    "state_dict": model.state_dict()
                }
                torch.save(save, os.path.join(output, f"checkpoint_epoch_{epoch+1}.pt"))
                logger.info(f"保存检查点: checkpoint_epoch_{epoch+1}.pt")
    
    # 保存最终模型
    final_save = {
        "epoch": epochs,
        "optim_dict": optim.state_dict(),
        "scheduler_dict": scheduler.state_dict(),
        "best_loss": best_val_loss,
        "loss": avg_val_loss,
        "state_dict": model.state_dict()
    }
    torch.save(final_save, os.path.join(output, "final.pt"))
    
    if use_wandb:
        wandb.finish()
    
    logger.info("训练完成！")
    logger.info(f"所有结果保存在: {output}")


if __name__ == '__main__':
    pretrain()