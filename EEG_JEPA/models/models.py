import torch
import torch.nn.functional as F
from torch import nn
from loguru import logger
from collections import OrderedDict
from einops import rearrange
import math
from typing import List, Tuple, Mapping
import torch.nn.utils.rnn as rnn_utils
from typing import Tuple, List, Optional, Dict
import random
from models.vit import JEPAModel 

class Tokenizer(nn.Module):
    def __init__(self, input_size=64, output_size=128):
        super(Tokenizer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.tokenizer = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.ELU(),
            nn.LayerNorm([4, self.input_size//2]),  
            
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.LayerNorm([8, self.input_size//4]),
            
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.LayerNorm([16, self.input_size//8]),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.LayerNorm([32, self.input_size//16]),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.LayerNorm([64, self.input_size//32]),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.LayerNorm([128, self.input_size//64]),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, self.output_size)
        )

    def forward(self, x):
        
        B, C, T = x.shape
        x = x.view(B, C, -1, self.input_size)
        x = x.permute(0, 1, 2, 3).contiguous().view(-1, 1, self.input_size)
        x = self.tokenizer(x)
        x = x.view(B, C, -1, self.output_size)
        
        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=1, dropout=0.2):
        super(AttentionPooling, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, input_dim = x.size()
        
        if key_padding_mask is not None:
            if key_padding_mask.size(1) == 1:
                return x.mean(dim=1)
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(dtype=torch.bool)
        
            transformer_output = self.transformer_layer(x, src_key_padding_mask=key_padding_mask)
            
            # Invert mask (1 for valid, 0 for padding) and handle the hidden dimension
            attention_mask = (~key_padding_mask).unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Calculate masked mean
            pooled_output = (transformer_output * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        else:
            transformer_output = self.transformer_layer(x)
            pooled_output = transformer_output.mean(dim=1)

        return pooled_output


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class SetTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim,  num_heads, num_layers,latent_dim=128,num_patches=60, pooling_head=4, dropout=0.1, max_seq_length=1024):
        super(SetTransformer, self).__init__()
        # self.patch_embedding = PatchEmbeddingLinear(in_channels, patch_size, embed_dim)
        self.patch_embedding = Tokenizer(input_size=patch_size, output_size=embed_dim)  #1-128
        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.jepa_encoder = JEPAModel(
            patch_dim=latent_dim,
            embed_dim=embed_dim,
            predictor_embed_dim=embed_dim,
            encoder_depth=8,
            predictor_depth=8,
            num_heads=8,
            num_patches=num_patches,
            mask_ratio=0.5,
            ema_decay=0.999
        )

        # self.temporal_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

    def forward(self, multi_modal_inputs, multi_modal_mask):
        B0, C,T0,  = multi_modal_inputs[0].shape
        # 1. 分别处理每个模态
        modality_embeddings = []
        for i, input_data in enumerate(multi_modal_inputs):
            x = self.patch_embedding(input_data) #16
            # print(x.dtype)
            B, C, S, E = x.shape
            x = rearrange(x, 'b c s e -> (b s) c e')

            mask = multi_modal_mask[i].unsqueeze(1).expand(-1, S, -1)
            mask = rearrange(mask, 'b t c -> (b t) c')

            x = self.spatial_pooling(x, mask) # [120, 1280] #通道维度的池化
            x = x.view((B, S, E))
            # print(x.dtype)
            x = self.positional_encoding(x) #

            x = x.to(torch.float16)
            x = self.layer_norm(x)

            x = self.transformer_encoder(x) #32, 30, 2048
            modality_embeddings.append(x) 


        x = torch.stack(modality_embeddings, dim=2)# [32, 30, 3, 2048]
        # x = rearrange(x, 'b c s e -> (b c) s e') # s模态间的池化
        # x = self.spatial_pooling(x) #[960,2048]
        # x = x.view((B, S, -1))
        x = x.mean(dim=2)  # [B, S, E] :似乎更优
        x = self.jepa_encoder(x)

        return x


class SetTransformerForSleepStaging(nn.Module):
    """
    用于睡眠分期的SetTransformer，输入30秒epoch，输出分类logits
    基于预训练的SetTransformer结构，增加可配置的池化层和分类头
    """
    def __init__(
        self,
        in_channels,
        patch_size,
        embed_dim,
        num_heads,
        num_layers,
        num_patches,
        jepa_depth,
        latent_dim=128,               # patch_dim for JEPA
        pooling_head=4,
        dropout=0.1,
        max_seq_length=128,
        num_sleep_stages=5,
        pooling_method='mean',         # 可选: 'mean', 'max', 'first', 'attention' concat_fc
        freeze_backbone=False,

    ):
        super().__init__()
        self.patch_embedding = Tokenizer(input_size=patch_size, output_size=embed_dim)  #1-128
        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
     
        # JEPA编码器（预训练模型的主干）
        self.jepa_encoder = JEPAModel(
            patch_dim=latent_dim,
            embed_dim=embed_dim,
            predictor_embed_dim=embed_dim,
            encoder_depth=jepa_depth,
            predictor_depth=jepa_depth,
            num_heads=8,
            num_patches=num_patches,
            mask_ratio=0.5,
            ema_decay=0.999
        )
        
        # 可配置的池化层
        # self.pooling = PoolingLayer(embed_dim, method=pooling_method, dropout=dropout)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_sleep_stages)
        )
        
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """冻结JEPA编码器的参数，其余部分（包括patch_embedding, transformer等）保持可训练"""
        # for name, param in self.named_parameters():
        #     if name.startswith('jepa_encoder'):
        #         param.requires_grad = False
        """冻结特征提取部分（所有除了classifier之外的参数）"""
        for name, param in self.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = False
    
    def forward(self, multi_modal_inputs, multi_modal_mask):
        """
        multi_modal_inputs: list of [B, C, T]  每个模态一个张量  (64,4,3840)
        multi_modal_mask:   list of [B, C]     每个模态的掩码
        Returns: [B, num_sleep_stages]
        """
        B = multi_modal_inputs[0].shape[0]
        device = multi_modal_inputs[0].device
        
        # ---------- 特征提取（与原SetTransformer forward一致）----------
        modality_embeddings = []
        for i, input_data in enumerate(multi_modal_inputs):
            if input_data.shape[1]==0:
                continue
            x = self.patch_embedding(input_data)  # [B, C, S, E]  (S=num_patches)
            B, C, S, E = x.shape
            # 合并batch和通道，对每个patch进行跨通道池化
            x = rearrange(x, 'b c s e -> (b s) c e')
            
            # 扩展掩码到每个patch
            mask = multi_modal_mask[i]  # [B, C]
            mask = mask.unsqueeze(1).expand(-1, S, -1)  # [B, S, C]
            mask = rearrange(mask, 'b s c -> (b s) c')
            
            x = self.spatial_pooling(x, mask)  # [(B*S), E]
            x = x.view(B, S, -1)  # [B, S, E]
            
            x = self.positional_encoding(x)
            x = self.layer_norm(x)
            x = self.transformer_encoder(x)  # [B, S, E]
            modality_embeddings.append(x)
        
        # 多模态融合：沿模态维度堆叠，然后池化
        x = torch.stack(modality_embeddings, dim=2)# [32, 30, 3, 2048]
        
        # x = rearrange(x, 'b c s e -> (b c) s e') # s模态间的池化
        # x = self.spatial_pooling(x) #[960,2048]
        # x = x.view((B, S, -1))

        x = x.mean(dim=2)  # [B, S, E]
     
        # 通过JEPA编码器（预训练主干）
        with torch.set_grad_enabled(not self.freeze_backbone):
            # full_features, _, _, _ = self.jepa_encoder(x)  # [B, S, jepa_embed_dim]
            # 使用 online_encoder 获取完整序列特征
            full_features = self.jepa_encoder.online_encoder(x)  
        
        # 池化序列维度 -> [B, jepa_embed_dim] (均值池化)
        pooled = full_features.mean(dim=1)  # [B, S, E]
        
        # 分类
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits