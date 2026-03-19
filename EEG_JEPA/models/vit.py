import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import List, Optional, Tuple
import numpy as np

# ------------------------------
# JEPA Masking Strategy for Patches
# ------------------------------

def create_jepa_patch_mask(
    batch_size: int, 
    num_patches: int, 
    mask_ratio: float = 0.5,
    min_span: int = 2, 
    max_span: int = 10,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create JEPA-style block masks for patch sequences.
    Returns: mask [B, N] where 1=keep, 0=mask (N = num_patches)
    """
    masks = torch.ones(batch_size, num_patches, device=device)
    
    for b in range(batch_size):
        num_to_mask = int(num_patches * mask_ratio)
        masked_so_far = 0
        
        while masked_so_far < num_to_mask:
            span_len = random.randint(min_span, max_span)
            start = random.randint(0, max(1, num_patches - span_len))
            end = min(start + span_len, num_patches)
            
            masks[b, start:end] = 0
            masked_so_far += (end - start)
            
            if masked_so_far >= num_to_mask:
                break
    
    return masks  # [B, N]

# ------------------------------
# Helper Functions for VisionTransformer
# ------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Tuple[int, int], cls_token: bool = False) -> np.ndarray:
    """Generate 2D sine-cosine positional embedding"""
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Helper function for 2D sine-cosine positional embedding"""
    assert embed_dim % 2 == 0
    
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sine-cosine positional embedding"""
    assert embed_dim % 2 == 0
    
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def apply_masks(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Apply masks to select specific tokens.
    
    Args:
        x: [B, N, D] input tokens
        masks: [B, M] or list of [B, M_i] indices to select (M <= N)
    
    Returns:
        Selected tokens [B_total, M_total, D]
    """
    if masks is None:
        return x
    
    if isinstance(masks, list):
        masks = torch.cat(masks, dim=0)
    
    B, N, D = x.shape
    
    if masks.dim() == 1:
        masks = masks.unsqueeze(0)
    
    device = x.device
    masks = masks.to(device)
    
    out = []
    num_masks = masks.shape[0]
    
    for i in range(B):
        batch_idx = i % num_masks
        batch_tokens = x[i:i+1]
        batch_mask = masks[batch_idx]
        selected = torch.index_select(batch_tokens, 1, batch_mask)
        out.append(selected)
    
    return torch.cat(out, dim=0)

def repeat_interleave_batch(x: torch.Tensor, B: int, repeat: int) -> torch.Tensor:
    """Repeat tensor to match batch dimension"""
    x = x.unsqueeze(0)
    repeat_dims = [1] * x.dim()
    repeat_dims[0] = repeat
    x = x.repeat(*repeat_dims)
    return x

def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> None:
    """Truncated normal initialization"""
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(min=a, max=b)

# ------------------------------
# Basic Blocks for VisionTransformer
# ------------------------------

class PatchEmbed(nn.Module):
    """Simple patch embedding for already patched inputs"""
    def __init__(self, patch_dim: int = 128, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, patch_dim]
        x = self.proj(x)  # [B, N, embed_dim]
        return x

class Mlp(nn.Module):
    """MLP block"""
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, 
                 act_layer: nn.Module = nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """Multi-head attention with masking support"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                 qk_scale: float = None, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # mask: [B, N] or [B_total, N]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """Transformer block"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 qkv_bias: bool = False, qk_scale: float = None, 
                 drop: float = 0.0, attn_drop: float = 0.0,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

# ------------------------------
# JEPA Encoder (VisionTransformer for Patches)
# ------------------------------

class JEPAEncoder(nn.Module):
    """
    JEPA Encoder based on VisionTransformer for patch sequences.
    
    Input: [B, N, D] where N=num_patches, D=patch_dim
    """
    def __init__(
        self,
        patch_dim: int = 128,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_std: float = 0.02,
        out_layers: List[int] = None,
        num_patches: int = 60,
        **kwargs
    ):
        super().__init__()
        
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.num_patches = num_patches
        
        # Patch embedding (from patch_dim to embed_dim)
        self.patch_embed = PatchEmbed(patch_dim=patch_dim, embed_dim=embed_dim)
        
        # Position embedding (1D for patch sequence)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim),
            requires_grad=True
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            ) for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(drop_rate)
        
        # Initialize weights
        self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
        self.apply(self._init_weights)
    
    def _init_pos_embed(self, pos_embed: torch.Tensor) -> None:
        """Initialize 1D positional embedding for patch sequence"""
        embed_dim = pos_embed.size(-1)
        
        # Create 1D grid (patch positions)
        grid = np.arange(self.num_patches, dtype=np.float32)
        sincos = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
        
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, N, D] patch sequence
            masks: indices of patches to mask (remove)
        
        Returns:
            Encoded features
        """
        # Tokenize input
        x = self.patch_embed(x)  # [B, N, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply dropout
        x = self.dropout(x)
        
        # Mask away unwanted tokens (if masks provided)
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)
        
        # Forward through transformer blocks
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=masks)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))
        
        if self.out_layers is not None:
            return outs
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x  # [B, N, embed_dim]

# ------------------------------
# JEPA Predictor (VisionTransformerPredictor for Patches)
# ------------------------------

class JEPAPredictor(nn.Module):
    """
    JEPA Predictor based on VisionTransformer for patch sequences.
    
    Predicts masked patches from context patches.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_std: float = 0.02,
        use_mask_tokens: bool = True,
        num_mask_tokens: int = 1,
        num_patches: int = 60,
        **kwargs
    ):
        super().__init__()
        
        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        self.use_mask_tokens = use_mask_tokens
        
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for _ in range(num_mask_tokens)
            ])
        
        # Position embedding for predictor
        self.num_patches = num_patches
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=True
        )
        
        # Transformer blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            ) for _ in range(depth)
        ])
        
        # Normalize and project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        # Initialize weights
        self._init_pos_embed(self.predictor_pos_embed.data)
        self.init_std = init_std
        
        if use_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        
        self.apply(self._init_weights)
    
    def _init_pos_embed(self, pos_embed: torch.Tensor) -> None:
        """Initialize 1D positional embedding for predictor"""
        embed_dim = pos_embed.size(-1)
        
        # Create 1D grid (patch positions)
        grid = np.arange(self.num_patches, dtype=np.float32)
        sincos = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
        
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        context: torch.Tensor,
        target_indices: torch.Tensor,
        context_indices: torch.Tensor,
        mask_index: int = 0
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            context: [B, N_context, embed_dim] context patches
            target_indices: [B, N_target] indices of target patches
            context_indices: [B, N_context] indices of context patches
            mask_index: which mask token to use
        
        Returns:
            Predicted target patches [B, N_target, embed_dim]
        """
        B = context.shape[0]
        
        # Map context tokens to predictor dimension
        x = self.predictor_embed(context)  # [B, N_context, predictor_embed_dim]
        
        # Add positional embedding to context tokens
        if self.predictor_pos_embed is not None:
            # Get positional embeddings for context indices
            pos_embed_all = self.predictor_pos_embed.expand(B, -1, -1)  # [B, N, D]
            
            # Gather positional embeddings for context indices
            context_pos = torch.gather(
                pos_embed_all, 
                1, 
                context_indices.unsqueeze(-1).expand(-1, -1, pos_embed_all.shape[-1])
            )
            x = x + context_pos
        
        # Prepare target tokens (mask tokens)
        if self.use_mask_tokens:
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]  # [1, 1, D]
            pred_tokens = pred_tokens.expand(B, self.num_patches, -1)  # [B, N, D]
            
            # Gather mask tokens for target indices
            pred_tokens = torch.gather(
                pred_tokens,
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, pred_tokens.shape[-1])
            )  # [B, N_target, D]
        else:
            # If no mask tokens, use zeros
            pred_tokens = torch.zeros(
                B, target_indices.shape[1], x.shape[-1],
                device=x.device, dtype=x.dtype
            )
        
        # Add positional embedding to target tokens
        if self.predictor_pos_embed is not None:
            # Gather positional embeddings for target indices
            target_pos = torch.gather(
                pos_embed_all,
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, pos_embed_all.shape[-1])
            )
            pred_tokens = pred_tokens + target_pos
        
        # Concatenate context and target tokens
        x = torch.cat([x, pred_tokens], dim=1)  # [B, N_context + N_target, D]
        
        # Create mask for attention (context can see context, target can see all)
        # For simplicity, we allow all-to-all attention
        mask = None
        
        # Forward through transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x, mask=mask)
        
        # Normalize
        x = self.predictor_norm(x)
        
        # Extract target predictions
        x_target = x[:, context.shape[1]:]  # [B, N_target, D]
        
        # Project back to original dimension
        x_target = self.predictor_proj(x_target)  # [B, N_target, embed_dim]
        
        return x_target


import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import List, Optional, Tuple

# ------------------------------
# Complete JEPA Model
# ------------------------------

class JEPAModel(nn.Module):
    """
    Complete JEPA model for EEG patch-based representation learning.
    
    Input: [B, N, D] where:
        B = batch_size (e.g., 32)
        N = num_patches (e.g., 60)
        D = patch_dim (e.g., 128)
    """
    def __init__(
        self,
        patch_dim: int = 128,
        embed_dim: int = 1024,
        predictor_embed_dim: int = 1024,
        encoder_depth: int = 12,
        predictor_depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_patches: int = 60,
        ema_decay: float = 0.996,
        mask_ratio: float = 0.5,
        min_blocks: int = 1,
        max_blocks: int = 3,
        **kwargs
    ):
        super().__init__()
        
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        
        # Online encoder (X-encoder, trainable)
        self.online_encoder = JEPAEncoder(
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_patches=num_patches,
            **kwargs
        )
        
        # Target encoder (Y-encoder, EMA, frozen)
        self.target_encoder = JEPAEncoder(
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_patches=num_patches,
            **kwargs
        )
        
        # Predictor network
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_patches=num_patches,
            **kwargs
        )
        
        # Learnable mask token for replacing masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, patch_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Initialize target encoder with online encoder weights
        self._init_target_encoder()
    
    def _init_target_encoder(self) -> None:
        """Initialize target encoder with online encoder weights"""
        with torch.no_grad():
            for param_online, param_target in zip(
                self.online_encoder.parameters(),
                self.target_encoder.parameters()
            ):
                param_target.data.copy_(param_online.data)
    
    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """EMA update of target encoder"""
        decay = decay or self.ema_decay
        
        with torch.no_grad():
            for param_online, param_target in zip(
                self.online_encoder.parameters(),
                self.target_encoder.parameters()
            ):
                param_target.data.mul_(decay).add_(param_online.data, alpha=1.0 - decay)
    
    def _create_masks(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Create JEPA-style multi-block masks for patch sequences.
        
        Returns:
            - full_mask: [B, N] binary mask (1=keep, 0=mask)
            - context_indices: [B, N_context] indices of context patches
            - target_indices: [B, N_target] indices of target patches
            - num_context: number of context patches per sample
            - num_target: number of target patches per sample
        """
        # Fixed mask counts per sample
        num_target = int(self.num_patches * self.mask_ratio)
        num_context = self.num_patches - num_target
        
        full_mask = torch.ones(batch_size, self.num_patches, device=device,dtype=torch.float16)
        
        for b in range(batch_size):
            # Initialize all patches as visible
            mask = torch.ones(self.num_patches, device=device)
            
            # Determine number of blocks (random between min_blocks and max_blocks)
            num_blocks = random.randint(self.min_blocks, self.max_blocks)
            
            # Distribute target patches among blocks
            target_per_block = [num_target // num_blocks] * num_blocks
            for i in range(num_target % num_blocks):
                target_per_block[i] += 1
            
            # Track already masked positions to avoid overlap
            masked_positions = []
            
            for block_idx in range(num_blocks):
                if target_per_block[block_idx] == 0:
                    continue
                
                # Determine block size (at least 2 patches)
                target_size = target_per_block[block_idx]
                max_block_size = min(self.num_patches, target_size * 3)  # Allow some flexibility
                min_block_size = max(2, target_size)
                
                if min_block_size > max_block_size:
                    block_size = max_block_size
                else:
                    block_size = random.randint(min_block_size, max_block_size)
                
                # Find valid start position
                max_start = self.num_patches - block_size
                if max_start <= 0:
                    start = 0
                    block_size = self.num_patches
                else:
                    # Try to find a position with minimal overlap
                    best_start = 0
                    min_overlap = float('inf')
                    
                    # Sample several random starts
                    for _ in range(10):
                        candidate_start = random.randint(0, max_start)
                        candidate_end = candidate_start + block_size
                        candidate_positions = set(range(candidate_start, candidate_end))
                        overlap = len(candidate_positions.intersection(set(masked_positions)))
                        
                        if overlap == 0:
                            best_start = candidate_start
                            break
                        elif overlap < min_overlap:
                            min_overlap = overlap
                            best_start = candidate_start
                    
                    start = best_start
                
                end = min(start + block_size, self.num_patches)
                
                # Mask this block
                mask[start:end] = 0
                masked_positions.extend(list(range(start, end)))
            
            # Adjust to exactly num_target masked patches
            current_masked = (mask == 0).sum().item()
            
            if current_masked > num_target:
                # Randomly unmask some positions
                masked_pos = torch.where(mask == 0)[0]
                keep_indices = torch.randperm(len(masked_pos))[:num_target]
                new_mask = torch.ones(self.num_patches, device=device)
                new_mask[masked_pos[keep_indices]] = 0
                mask = new_mask
            elif current_masked < num_target:
                # Randomly mask additional positions
                unmasked_pos = torch.where(mask == 1)[0]
                additional = num_target - current_masked
                if len(unmasked_pos) > 0:
                    additional = min(additional, len(unmasked_pos))
                    mask_idx = torch.randperm(len(unmasked_pos))[:additional]
                    mask[unmasked_pos[mask_idx]] = 0
            
            full_mask[b] = mask
        
        # Create indices tensors
        context_indices = torch.zeros(batch_size, num_context, device=device,dtype=torch.int64)
        target_indices = torch.zeros(batch_size, num_target, device=device,dtype=torch.int64)
        
        for b in range(batch_size):
            # Get indices for context and target patches
            context_idx = torch.where(full_mask[b] == 1)[0]
            target_idx = torch.where(full_mask[b] == 0)[0]
            
            # Shuffle indices to avoid order bias
            context_idx = context_idx[torch.randperm(len(context_idx))]
            target_idx = target_idx[torch.randperm(len(target_idx))]
            
            context_indices[b] = context_idx
            target_indices[b] = target_idx
        
        return full_mask, context_indices, target_indices, num_context, num_target
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass following the EEG-VJEPA pipeline.
        
        Args:
            x: [B, N, D] input patches
        
        Returns:
            context_features: [B, num_context, embed_dim] features from X-encoder
            predicted_targets: [B, num_target, embed_dim] predicted target features
            target_features: [B, num_target, embed_dim] target features from Y-encoder
            full_mask: [B, N] binary mask used
        """
        B, N, D = x.shape
        device = x.device
        
        # 1. Create masks
        full_mask, context_indices, target_indices, num_context, num_target = self._create_masks(B, device)
        
        # 2. Prepare masked input for online encoder (X-encoder)
        # Replace masked patches with learnable mask tokens
        mask_tokens = self.mask_token.expand(B, N, D)
        mask_3d = full_mask.unsqueeze(-1)  # [B, N, 1]
        x_masked = x * mask_3d + mask_tokens * (1 - mask_3d)
        
        # 3. X-encoder processes masked input
        z_online = self.online_encoder(x_masked)  # [B, N, embed_dim]
        
        # 4. Extract context features (visible patches)
        context_features = torch.gather(z_online, 1, context_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))  # [B, num_context, embed_dim]
        
        # 5. Y-encoder processes full input (no masking)
        with torch.no_grad():
            z_target = self.target_encoder(x)  # [B, N, embed_dim]
            target_features = torch.gather(z_target,1,target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))  # [B, num_target, embed_dim]
        
        # 6. Predictor predicts target features from context
        predicted_targets = self.predictor(
            context=context_features,
            target_indices=target_indices,
            context_indices=context_indices,
            mask_index=0
        )  # [B, num_target, embed_dim]
        
        return context_features, predicted_targets, target_features, full_mask


