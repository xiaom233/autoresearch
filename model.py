"""
Image restoration model — SwinIR-based RestoreNet with pluggable attention.

Architecture:
  1. Shallow feature extraction: 3×3 conv
  2. Deep feature extraction: RSTB stack (window-based transformer)
  3. Reconstruction: conv → conv with global residual

Attention types: swin (default), mdta (channel), ocab (overlap spatial)
Based on: SwinIR (Liang et al., 2021), Restormer (Zamir et al., 2022),
          X-Restormer (Chen et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ======================================================================
# Swin Window Attention (default)
# ======================================================================


class WindowSDPA(nn.Module):
    """Window-based multi-head attention using F.scaled_dot_product_attention.

    Supports regular (W-MSA) and shifted (SW-MSA) window partitioning.
    Uses PyTorch's native SDPA which dispatches to flash-attention on supported GPUs.
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, f"Spatial dims ({H},{W}) must be multiples of window_size {ws}"

        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        nH, nW = H // ws, W // ws
        x = x.view(B, nH, ws, nW, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

        B_win, N, _ = x.shape
        qkv = self.qkv(x).view(B_win, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).contiguous().view(B_win, N, C)
        x = self.proj(x)

        x = x.view(B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        return x.view(B, L, C)


# ======================================================================
# MDTA — Multi-DConv Head Transposed Attention (Restormer channel attn)
# ======================================================================
class MDTA(nn.Module):
    """Channel-wise self-attention from Restormer."""
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1,
                                     padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, C, H, W)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x=H, y=W)
        out = self.project_out(out)
        return out.flatten(2).transpose(1, 2)


class MDTABlock(nn.Module):
    """MDTA + Gated-Dconv FFN."""
    def __init__(self, dim, num_heads, mlp_ratio=2., bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.conv1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)

    def forward(self, x, x_size):
        H, W = x_size
        x = x + self.attn(self.norm1(x), x_size)
        # GDFN
        B, L, C = x.shape
        x_norm = self.norm2(x).transpose(1, 2).view(B, C, H, W)
        x1, x2 = self.conv1(x_norm).chunk(2, dim=1)
        x_norm = self.conv2(self.dwconv(F.gelu(x1) * x2))
        return x + x_norm.flatten(2).transpose(1, 2)


# ======================================================================
# OCAB — Overlapping Cross-Attention Block (X-Restormer spatial attn)
# ======================================================================
class OCAB(nn.Module):
    """Spatial attention with overlapping windows from X-Restormer."""
    def __init__(self, dim, num_heads, window_size=8, overlap_ratio=0.5, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.overlap = int(window_size * overlap_ratio)
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, C, H, W)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        ws = self.window_size
        ov = self.overlap
        # Q on non-overlapping windows
        q_w = rearrange(q, 'b (h c) (n1 w1) (n2 w2) -> (b n1 n2) (w1 w2) (h c)',
                        h=self.num_heads, w1=ws, w2=ws)
        # K,V on overlapping windows via Unfold
        k_ov = F.unfold(k, kernel_size=ws+ov, stride=ws, padding=ov//2)
        v_ov = F.unfold(v, kernel_size=ws+ov, stride=ws, padding=ov//2)
        k_ov = rearrange(k_ov, 'b (h c k) n -> (b n) k (h c)', h=self.num_heads)
        v_ov = rearrange(v_ov, 'b (h c k) n -> (b n) k (h c)', h=self.num_heads)

        attn = (q_w @ k_ov.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v_ov
        out = rearrange(out, '(b n1 n2) (w1 w2) (h c) -> b (h c) (n1 w1) (n2 w2)',
                        n1=H//ws, n2=W//ws, w1=ws, w2=ws, h=self.num_heads)
        out = self.proj(out)
        return out.flatten(2).transpose(1, 2)


class OCABBlock(nn.Module):
    """OCAB + Gated-Dconv FFN."""
    def __init__(self, dim, num_heads, window_size=8, overlap_ratio=0.5, mlp_ratio=2.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OCAB(dim, num_heads, window_size, overlap_ratio)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.conv1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)

    def forward(self, x, x_size):
        H, W = x_size
        x = x + self.attn(self.norm1(x), x_size)
        B, L, C = x.shape
        x_norm = self.norm2(x).transpose(1, 2).view(B, C, H, W)
        x1, x2 = self.conv1(x_norm).chunk(2, dim=1)
        x_norm = self.conv2(self.dwconv(F.gelu(x1) * x2))
        return x + x_norm.flatten(2).transpose(1, 2)


class SwinBlock(nn.Module):
    """Swin Transformer block: window SDPA + MLP, pre-norm style."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2,
                 activation="gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowSDPA(dim, num_heads, window_size, shift_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        if activation == "swiglu":
            self.w1 = nn.Linear(dim, hidden)
            self.w2 = nn.Linear(dim, hidden)
            self.w3 = nn.Linear(hidden, dim)
            self.activation = "swiglu"
        else:
            act_fn = {"gelu": nn.GELU(), "relu": nn.ReLU(),
                      "silu": nn.SiLU()}.get(activation, nn.GELU())
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                act_fn,
                nn.Linear(hidden, dim),
            )
            self.activation = activation

    def forward(self, x, x_size):
        x = x + self.attn(self.norm1(x), x_size)
        if self.activation == "swiglu":
            x_norm = self.norm2(x)
            x = x + self.w3(F.silu(self.w1(x_norm)) * self.w2(x_norm))
        else:
            x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: attention blocks + conv skip connection.

    attention_type: "swin" (window), "mdta" (channel), "ocab" (overlap spatial)
    """

    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2,
                 activation="gelu", window_shift_ratio=0.5, skip_type="standard",
                 conv_kernel=3, attention_type="swin"):
        super().__init__()
        if attention_type == "mdta":
            self.blocks = nn.ModuleList([
                MDTABlock(dim, num_heads, mlp_ratio) for _ in range(depth)
            ])
        elif attention_type == "ocab":
            self.blocks = nn.ModuleList([
                OCABBlock(dim, num_heads, window_size, overlap_ratio=0.5,
                          mlp_ratio=mlp_ratio) for _ in range(depth)
            ])
        else:  # swin
            shift_size = int(window_size * window_shift_ratio)
            self.blocks = nn.ModuleList([
                SwinBlock(dim, num_heads, window_size,
                          shift_size=0 if i % 2 == 0 else shift_size,
                          mlp_ratio=mlp_ratio, activation=activation)
                for i in range(depth)
            ])
        self.conv = nn.Conv2d(dim, dim, conv_kernel, 1, conv_kernel // 2)
        self.skip_type = skip_type
        if skip_type == "learnable":
            self.skip_alpha = nn.Parameter(torch.ones(1))
        elif skip_type == "dense":
            self.skip_conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x, x_size):
        x_in = x
        for blk in self.blocks:
            x = blk(x, x_size)
        B, L, C = x.shape
        H, W = x_size
        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.conv(feat).flatten(2).transpose(1, 2)
        if self.skip_type == "none":
            return x + feat
        elif self.skip_type == "learnable":
            return x + self.skip_alpha * feat
        elif self.skip_type == "dense":
            feat2d = feat.transpose(1, 2).view(B, C, H, W)
            x2d = x_in.transpose(1, 2).view(B, C, H, W)
            dense_feat = self.skip_conv(torch.cat([feat2d, x2d], dim=1))
            return x + dense_feat.flatten(2).transpose(1, 2)
        return x + feat


class RestoreNet(nn.Module):
    """SwinIR for image restoration (no upsampling, same-resolution in/out)."""

    def __init__(self, in_ch=3, embed_dim=64, depths=(2, 2, 2, 2),
                 num_heads=(4, 4, 4, 4), window_size=8, mlp_ratio=2,
                 activation="gelu", head_dim=0, stage_config="uniform",
                 num_stages=4, window_shift_ratio=0.5, skip_type="standard",
                 conv_kernel=3, attention_type="swin"):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim

        if head_dim > 0:
            _num_heads = tuple(max(1, embed_dim // head_dim) for _ in range(num_stages))
        else:
            _num_heads = num_heads
            if len(_num_heads) != num_stages:
                _num_heads = (_num_heads[0],) * num_stages

        if stage_config == "ascending":
            _depths = tuple(range(1, num_stages + 1))
        elif stage_config == "descending":
            _depths = tuple(range(num_stages, 0, -1))
        elif stage_config == "pyramid":
            _depths = depths
            _dims = [int(embed_dim * (0.75 + 0.25 * i / (num_stages - 1))) for i in range(num_stages)] if num_stages > 1 else [embed_dim]
        else:
            _depths = depths
            if len(_depths) != num_stages:
                _depths = (_depths[0],) * num_stages

        if stage_config == "pyramid" and num_stages > 1:
            _dims = []
            for i in range(num_stages):
                raw_dim = int(embed_dim * (0.75 + 0.25 * i / max(1, num_stages - 1)))
                nh = _num_heads[i] if i < len(_num_heads) else _num_heads[-1]
                _dims.append((raw_dim // nh) * nh)
        else:
            _dims = [embed_dim] * num_stages

        self.conv_first = nn.Conv2d(in_ch, _dims[0], 3, 1, 1)

        self.layers = nn.ModuleList()
        for i in range(num_stages):
            dim = _dims[i]
            d = _depths[i] if i < len(_depths) else _depths[-1]
            nh = _num_heads[i] if i < len(_num_heads) else _num_heads[-1]
            if i > 0 and stage_config == "pyramid":
                self.layers.append(nn.Conv2d(_dims[i-1], dim, 1))
            self.layers.append(
                RSTB(dim, d, nh, window_size, mlp_ratio, activation,
                     window_shift_ratio, skip_type, conv_kernel, attention_type)
            )
        self.norm = nn.LayerNorm(_dims[-1])
        self._final_dim = _dims[-1]
        self._stage_config = stage_config
        self._num_stages = num_stages
        self._dims = _dims

        fd = _dims[-1]
        self.conv_after_body = nn.Conv2d(fd, fd, conv_kernel, 1, conv_kernel // 2)
        self.conv_last = nn.Conv2d(fd, in_ch, conv_kernel, 1, conv_kernel // 2)
        self._shallow_proj = nn.Conv2d(_dims[0], fd, 1) if _dims[0] != fd else None

    def _pad_to_window(self, x):
        _, _, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x

    def forward(self, x):
        H_in, W_in = x.shape[2:]
        x = self._pad_to_window(x)

        shallow = self.conv_first(x)
        B, _, H, W = shallow.shape

        x_seq = shallow.flatten(2).transpose(1, 2)

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                x_seq = x_seq.transpose(1, 2).view(B, -1, H, W)
                x_seq = layer(x_seq).flatten(2).transpose(1, 2)
            else:
                x_seq = layer(x_seq, (H, W))

        x_seq = self.norm(x_seq)
        deep_feat = x_seq.transpose(1, 2).view(B, self._final_dim, H, W)

        if self._shallow_proj is not None:
            shallow = self._shallow_proj(shallow)

        out = self.conv_after_body(deep_feat) + shallow
        out = self.conv_last(out) + x[:, :3, :, :]

        return out[:, :, :H_in, :W_in]
