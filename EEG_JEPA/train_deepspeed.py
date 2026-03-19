import torch
import torch.nn as nn
from loguru import logger
import os
import sys
import json
sys.path.append("../")
from utils import *
from models.dataset import SetTransformerDataset, collate_fn
from models.models import SetTransformer
import click
import time
import math
import datetime
import numpy as np
import tqdm
import wandb
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
import deepspeed
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compute_masked_loss0000(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # 余弦相似度损失
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    cos_loss = (1 - cos_sim).mean()
    return cos_loss


def compute_masked_loss000(pred: torch.Tensor, target: torch.Tensor,mask: torch.Tensor) -> torch.Tensor:
    """
    计算L2损失（均方误差）
    
    参数:
    - pred: 预测张量
    - target: 目标张量
    
    返回:
    - 损失值
    """
    return F.mse_loss(pred, target)

def compute_masked_loss(predicted_targets, target_features, maks,loss_exp=1):
    loss = torch.mean(torch.abs(predicted_targets - target_features)**loss_exp) / loss_exp
    return loss

def run_iter(batch, model, device, use_fp16=True):
    """
    JEPA模型训练
    """
    batch_data, mask_list, *_ = batch
    (bas, ekg, emg) = batch_data
    (mask_bas, mask_ekg, mask_emg) = mask_list

    # 准备数据列表
    psg_data = []
    mask_data = []

    for data, mask in zip([bas, ekg, emg], [mask_bas, mask_ekg, mask_emg]):
        if torch.isnan(data).any() or torch.isinf(data).any():
            logger.warning(f"检测到NaN或Inf数据，跳过该batch")
            return None
        psg_data.append(data.to(device, dtype=torch.float16))
        mask_data.append(mask.to(device))

    # 前向传播 context_features, predicted_targets, target_features, full_mask
    _, pred, target, pred_mask = model(psg_data, mask_data)

    # 计算损失
    return compute_masked_loss(pred, target, pred_mask)

def build_engine(model: nn.Module, ds_config, args, config_params=None):
    """
    构建DeepSpeed引擎
    """
    if isinstance(ds_config, str):
        with open(ds_config, "r") as f:
            ds_config = json.load(f)

    # 如果提供了训练配置参数，更新DeepSpeed配置
    if config_params is not None:
        if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
            ds_config["optimizer"]["params"]["lr"] = config_params.get("lr", 1.5e-4)
            ds_config["optimizer"]["params"]["weight_decay"] = config_params.get("weight_decay", 0.01)

        if "gradient_accumulation_steps" in ds_config:
            ds_config["gradient_accumulation_steps"] = config_params.get(
                "gradient_accumulation_steps", ds_config["gradient_accumulation_steps"]
            )

    # 设置CUDA设备
    if torch.cuda.is_available() and args.local_rank is not None and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    # 初始化DeepSpeed引擎
    engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config
    )

    grad_accum_steps = ds_config.get("gradient_accumulation_steps", 1)
    return DeepSpeedWrapper(engine, is_deepspeed=True, gradient_accumulation_steps=grad_accum_steps), scheduler

def train_epoch_deepspeed(wrapper, dataloader, epoch, config, split="train",
                         train_writer=None, val_writer=None, global_step=0,
                         val_global_step_start=0):
    """
    使用DeepSpeed训练一个epoch
    """
    is_train = (split == "pretrain")

    if is_train:
        wrapper.train()
    else:
        wrapper.eval()

    total_loss = 0.0
    total_samples = 0
    is_main_process = (wrapper.local_rank == 0)
    current_step = global_step if is_train else val_global_step_start

    with torch.set_grad_enabled(is_train):
        if is_main_process:
            pbar = tqdm.tqdm(total=len(dataloader), desc=f"{split.capitalize()} Epoch {epoch+1}")
        else:
            pbar = None

        for batch_idx, batch in enumerate(dataloader):
            total_loss_iter = run_iter(batch, wrapper.module, wrapper.device)

            if is_train:
                wrapper.backward(total_loss_iter)
                wrapper.step()

                if train_writer and is_main_process and batch_idx == 0:
                    lr = wrapper.get_lr()[0][0]
                    train_writer.add_scalar('learning_rate', lr, current_step)

            if is_main_process:
                batch_size_actual = batch[0][0].size(0)
                total_loss += total_loss_iter.item() * batch_size_actual
                total_samples += batch_size_actual

                if is_train and train_writer:
                    train_writer.add_scalar('total_loss', total_loss_iter.item(), current_step)
                elif not is_train and val_writer:
                    val_writer.add_scalar('total_loss', total_loss_iter.item(), current_step)

                if pbar:
                    pbar.update()
                    pbar.set_postfix({
                        'Loss': f"{total_loss_iter.item():.4f}",
                        'AvgLoss': f"{total_loss/total_samples:.4f}" if total_samples > 0 else "0.0000",
                    })

            current_step += 1

    if pbar:
        pbar.close()

    if is_main_process:
        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.item()
        if isinstance(total_samples, torch.Tensor):
            total_samples = total_samples.item()
        avg_loss = float(total_loss / total_samples) if total_samples > 0 else 0.0
    else:
        avg_loss = 0.0

    return avg_loss, current_step

@click.command("pretrain")
@click.option("--config_path", type=str,
              default='/mnt/DATA/poolab/jepa_dataset/EEG_JEPA/configs/config_set_transformer_contrastive.yaml')
@click.option("--channel_groups_path", type=str,
              default='/mnt/DATA/poolab/jepa_dataset/EEG_JEPA/configs/channel_groups.json')
@click.option("--checkpoint_path", type=str, default=None,
              help="DeepSpeed检查点目录路径，用于恢复训练。若提供，将从该目录加载模型、优化器和调度器状态。")
@click.option("--use_wandb", type=bool, default=False)
@click.option("--use_deepspeed", type=bool, default=True, help="是否使用DeepSpeed训练")
@click.option("--deepspeed_config", type=str,
              default="/mnt/DATA/poolab/jepa_dataset/EEG_JEPA/configs/deepspeed_config.json",
              help="DeepSpeed配置文件路径")
@click.option("--local_rank", type=int, default=-1, help="分布式训练的本地rank")
def pretrain_main(config_path, channel_groups_path, checkpoint_path, use_wandb,
                  use_deepspeed, deepspeed_config, local_rank):
    """
    预训练主函数（支持DeepSpeed，可从检查点恢复）
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=local_rank)
    args = parser.parse_args()

    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

    # 如果提供了检查点目录，则从中读取配置（可选）
    if checkpoint_path and os.path.isdir(checkpoint_path):
        logger.info(f"从检查点目录恢复: {checkpoint_path}")
        config_path_in_ckpt = os.path.join(checkpoint_path, "config.json")
        if os.path.isfile(config_path_in_ckpt):
            config = load_config(config_path_in_ckpt)
            logger.info("已加载检查点中的配置文件")
        output_dir = checkpoint_path  # 继续使用同一目录保存后续检查点
    else:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            config["save_path"],
            f"{config['model']}/{config['mode']}_{config['embed_dim']}_patch_{config['patch_size']}_{current_time}"
        )
        os.makedirs(output_dir, exist_ok=True)

    # 保存当前配置
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = SetTransformer(
        in_channels=config["in_channels"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        latent_dim=config.get("latent_dim", 512),
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        pooling_head=config["pooling_head"],
        dropout=config["dropout"],
        num_patches=config["num_patches"]
    )
    total_layers, total_params = count_parameters(model)
    logger.info(f'可训练参数: {total_params / 1e6:.2f} 百万')
    logger.info(f'层数: {total_layers}')

    # TensorBoard（只在主进程）
    if args.local_rank <= 0:
        tensorboard_dir = os.path.join(output_dir, "tensorboard_logs")
        train_writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, "train"))
        val_writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, "val"))
        logger.info(f"TensorBoard日志保存在: {tensorboard_dir}")
    else:
        train_writer = None
        val_writer = None

    best_val_loss = float('inf')
    global_step = 0
    val_global_step = 0
    start_epoch = 0

    if use_deepspeed:
        logger.info(f"使用DeepSpeed训练，配置文件: {deepspeed_config}")

        config_params = {
            "lr": config.get("lr", 1.5e-4),
            "weight_decay": config.get("weight_decay", 0.01),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 2)
        }

        wrapper, scheduler = build_engine(model, deepspeed_config, args, config_params)

        # 从检查点加载（如果提供了目录）
        if checkpoint_path and os.path.isdir(checkpoint_path):
            logger.info(f"尝试从DeepSpeed检查点加载: {checkpoint_path}")
            # 注意：DeepSpeed的load_checkpoint需要目录路径，tag可以为空（默认加载最新）
            # 这里假设检查点文件直接保存在checkpoint_path下
            load_path, client_state = wrapper.engine.load_checkpoint(
                checkpoint_path,
                tag="best_model",  # 若文件直接放在目录下，tag=None即可
                load_optimizer_states=True,
                load_lr_scheduler_states=True
            )
            if load_path is not None:
                global_step = client_state.get("global_step", 0)
                best_val_loss = client_state.get("best_val_loss", float('inf'))
                start_epoch = client_state.get("epoch", 0) + 1  # 恢复从下一epoch开始
                logger.info(f"成功加载检查点，恢复训练: start_epoch={start_epoch}, global_step={global_step}, best_val_loss={best_val_loss}")
            else:
                logger.warning("检查点加载失败，从头开始训练")
    else:
        # 普通训练模式（略，可根据需要实现）
        logger.info("使用普通训练模式")
        if device.type == "cuda":
            model = torch.nn.DataParallel(model)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1.5e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
            eta_min=1e-6
        )
        wrapper = DeepSpeedWrapper(
            type('obj', (object,), {
                'module': model,
                'optimizer': optimizer,
                'local_rank': 0
            })(),
            is_deepspeed=False,
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1)
        )
        # 若需要从普通检查点加载，可在此添加代码
        # 但此处仅针对DeepSpeed恢复，故略

    # 创建日志文件（主进程）
    if wrapper.local_rank <= 0:
        log_file = os.path.join(output_dir, "training_log.tsv")
        with open(log_file, "w") as f:
            f.write("Epoch\tSplit\tLoss\tLR\n")
    else:
        log_file = None

    # 数据集
    train_dataset = SetTransformerDataset(config, channel_groups, split="pretrain")
    val_dataset = SetTransformerDataset(config, channel_groups, split="validation")

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 训练循环
    for epoch in range(start_epoch, config["epochs"]):
        if wrapper.local_rank <= 0:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{config['epochs']}")
            print(f"{'='*50}")

        num_workers = config.get("num_workers", 4)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            num_workers=num_workers,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True
        )

        train_loss, global_step = train_epoch_deepspeed(
            wrapper=wrapper,
            dataloader=train_dataloader,
            epoch=epoch,
            config=config,
            split="pretrain",
            train_writer=train_writer,
            val_writer=None,
            global_step=global_step
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=True
        )

        val_loss, val_global_step = train_epoch_deepspeed(
            wrapper=wrapper,
            dataloader=val_dataloader,
            epoch=epoch,
            config=config,
            split="validation",
            train_writer=None,
            val_writer=val_writer,
            global_step=0,
            val_global_step_start=epoch * len(val_dataloader)
        )

        if not use_deepspeed:
            scheduler.step()

        # 记录日志（主进程）
        if wrapper.local_rank <= 0:
            current_lr = wrapper.get_lr()[0][0]
            train_loss = float(train_loss)
            val_loss = float(val_loss)

            with open(log_file, "a") as f:
                f.write(f"{epoch+1}\tTrain\t{train_loss:.5f}\t{current_lr:.6f}\n")
                f.write(f"{epoch+1}\tValidation\t{val_loss:.5f}\t{current_lr:.6f}\n")

            print(f"训练平均损失: {train_loss:.4f}")
            print(f"验证平均损失: {val_loss:.4f}")
            print(f"当前学习率: {current_lr:.6f}")
            print(f"训练步数: {global_step}, 验证步数: {val_global_step}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                if use_deepspeed:
                    client_state = {
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'global_step': global_step,
                        'val_global_step': val_global_step,
                        'config': config
                    }
                    wrapper.engine.save_checkpoint(
                        output_dir,
                        tag="best_model",
                        client_state=client_state
                    )
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': wrapper.module.state_dict(),
                        'optimizer_state_dict': wrapper.engine.optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'global_step': global_step,
                        'val_global_step': val_global_step,
                        'best_val_loss': best_val_loss,
                        'config': config
                    }, os.path.join(output_dir, "best_model.pth"))

                print(f"保存最佳模型，验证损失: {val_loss:.4f}")

            if (epoch + 1) % 5 == 0:
                if use_deepspeed:
                    client_state = {
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'global_step': global_step,
                        'val_global_step': val_global_step,
                        'config': config
                    }
                    wrapper.engine.save_checkpoint(
                        output_dir,
                        tag=f"checkpoint_epoch_{epoch+1}",
                        client_state=client_state
                    )
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': wrapper.module.state_dict(),
                        'optimizer_state_dict': wrapper.engine.optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'global_step': global_step,
                        'val_global_step': val_global_step,
                        'best_val_loss': best_val_loss,
                        'config': config
                    }, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"))

                print(f"保存检查点: epoch_{epoch+1}")

    # 保存最终模型
    if wrapper.local_rank <= 0:
        if use_deepspeed:
            client_state = {
                'epoch': config["epochs"],
                'best_val_loss': best_val_loss,
                'global_step': global_step,
                'val_global_step': val_global_step,
                'config': config
            }
            wrapper.engine.save_checkpoint(
                output_dir,
                tag="final_model",
                client_state=client_state
            )
        else:
            torch.save({
                'epoch': config["epochs"],
                'model_state_dict': wrapper.module.state_dict(),
                'optimizer_state_dict': wrapper.engine.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'global_step': global_step,
                'val_global_step': val_global_step,
                'best_val_loss': best_val_loss,
                'config': config
            }, os.path.join(output_dir, "final_model.pth"))

        if train_writer:
            train_writer.close()
        if val_writer:
            val_writer.close()

        print(f"\nTensorBoard日志已保存到 {tensorboard_dir}")
        print(f"运行以下命令查看TensorBoard:")
        print(f"tensorboard --logdir={tensorboard_dir}")
        print(f"\n训练完成！结果保存在: {output_dir}")

    return output_dir

if __name__ == "__main__":
    pretrain_main()