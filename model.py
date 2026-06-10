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


class GlobalChannelModulation(nn.Module):
    """Global channel statistics → per-channel affine modulation (FiLM-style).

    For global degradations (contrast/brightness/saturation), extracts
    global statistics via GAP and predicts per-channel scale + shift.
    Zero-initialized for identity at training start.

    Inspired by: SE-Net (Hu et al. 2018), FiLM (Perez et al. 2018),
    HAT (Chen et al. 2023), NAFNet (Chen et al. 2022)
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        feat = x.transpose(1, 2).view(B, C, H, W)
        params = self.mlp(self.gap(feat).flatten(1))
        scale, shift = params.chunk(2, dim=1)
        feat = feat * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        return feat.flatten(2).transpose(1, 2)


class ChannelCurve(nn.Module):
    """Per-channel non-linear curve correction for gamma/stretch degradations.

    Uses grouped 1x1 Conv (each channel processed independently) to learn
    an arbitrary per-channel curve f_c: R→R. Applied in pixel space before
    the main network.

    For type B global degradations: brightness_gamma (y=x^γ), contrast_stretch.
    """
    def __init__(self, in_ch=3, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * hidden, 1, groups=in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch * hidden, in_ch, 1, groups=in_ch),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class ColorMLP(nn.Module):
    """Per-pixel cross-channel MLP for HSV/YCrCb color space degradations.

    Full 1x1 Conv (cross-channel mixing) → ReLU → 1x1 Conv. A universal
    color-space transform approximator applied independently to each pixel.
    Zero spatial operations — only channel interactions.

    For type C global degradations: brightness_HSV, saturate_HSV/YCrCb.
    """
    def __init__(self, in_ch=3, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, in_ch, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class FreqMod(nn.Module):
    """Frequency-domain global modulation (per-block, based on SFHformer).

    Adapted from SFHformer (ECCV 2024) FourierUnit:
    - Real+Imag concatenation (not amplitude/phase separation)
    - FCPE: depth-wise conv on frequency features with residual connection
    - Simplified FDC: point-wise conv for per-frequency modulation
    - BatchNorm for frequency feature normalization

    The frequency domain naturally separates global (amplitude spectrum)
    from local (phase spectrum) information.
    """
    def __init__(self, dim):
        super().__init__()
        c2 = dim * 2
        self.bn = nn.BatchNorm2d(c2)
        # FCPE: Frequency Conditional Positional Encoding
        self.fcpe = nn.Conv2d(c2, c2, 3, padding=1, groups=c2)
        # Frequency-domain processing: point-wise conv after BN+GELU
        self.freq_conv = nn.Sequential(
            nn.Conv2d(c2, c2, 1),
            nn.GELU(),
            nn.Conv2d(c2, c2, 1),
        )
        nn.init.zeros_(self.freq_conv[-1].weight)
        nn.init.zeros_(self.freq_conv[-1].bias)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        feat = x.transpose(1, 2).view(B, C, H, W).float()

        # Disable AMP autocast for FFT operations (bfloat16 not supported)
        with torch.amp.autocast('cuda', enabled=False):
            # FFT → real + imag concatenation (as in SFHformer)
            ffted = torch.fft.rfft2(feat, norm='ortho')
            f_real = torch.unsqueeze(torch.real(ffted), dim=-1)
            f_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
            ffted = torch.cat((f_real, f_imag), dim=-1)
            ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()

            # BN → FCPE (DWConv residual) → frequency conv
            ffted = self.bn(ffted)
            ffted = self.fcpe(ffted) + ffted
            ffted = self.freq_conv(ffted)

            # Back to complex → IFFT
            ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
            ffted = torch.view_as_complex(ffted)
            out = torch.fft.irfft2(ffted, s=(H, W), norm='ortho')

        return out.to(x.dtype).flatten(2).transpose(1, 2)


class PCP(nn.Module):
    """Shared global modulation in feature space (between RSTB stages).

    Unlike per-block PCP, this is a SINGLE shared module applied between
    RSTB stages in RestoreNet. Keeps ColorPre's low-resolution stability
    design but in feature space, with negligible parameter overhead.

    Design (in feature space, shared):
    AdaptiveAvgPool(8x8) → Conv1x1 → ReLU → GAP → MLP → scale+shift
    """
    def __init__(self, dim, pool_size=8, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        hidden = max(dim // reduction, 8)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),  # 1x1 conv for efficiency
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        feat = x.transpose(1, 2).view(B, C, H, W)
        low = self.pool(feat)
        params = self.net(low)
        scale, shift = params.chunk(2, dim=1)
        feat = feat * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        return feat.flatten(2).transpose(1, 2)


class CSN(nn.Module):
    """Channel Statistics Normalization — InstanceNorm + learnable affine.

    InstanceNorm normalizes per-channel spatial mean/variance, directly
    inverting brightness (mean shift) and contrast (variance scaling).
    Learnable affine parameters (gamma, beta) provide optimal global stats.

    IN is inherently bounded (unlike FiLM-GCM's unbounded MLP output),
    making CSN stable against NaN while providing per-block global correction.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.norm(feat)
        return feat.flatten(2).transpose(1, 2)


class ColorPre(nn.Module):
    """Low-resolution global color processing (based on CSEC design pattern).

    CSEC (CVPR 2024) processes illumination at low resolution (256×256)
    for global color/illumination estimation, then applies correction to
    full resolution. This avoids the input-level interference with spatial
    feature extraction that we observed with ColorMLP.

    Design:
    - Downsample input → small CNN for global color statistics
    - Predict per-channel affine correction (scale + shift)
    - Apply to full-resolution input via residual
    """
    def __init__(self, in_ch=3, hidden=16):
        super().__init__()
        self.down = nn.AdaptiveAvgPool2d(64)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_ch * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        low = self.down(x)
        params = self.net(low)  # (B, 2*in_ch)
        scale, shift = params.chunk(2, dim=1)
        return x * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


class SwinBlock(nn.Module):
    """Swin Transformer block: window SDPA + MLP + optional GCM/FreqMod, pre-norm style."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2,
                 activation="gelu", use_gcm=False, use_freqmod=False,
                 use_csn=False):
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
        self.gcm = GlobalChannelModulation(dim) if use_gcm else None
        self.freqmod = FreqMod(dim) if use_freqmod else None
        self.csn = CSN(dim) if use_csn else None

    def forward(self, x, x_size):
        x = x + self.attn(self.norm1(x), x_size)
        if self.activation == "swiglu":
            x_norm = self.norm2(x)
            x = x + self.w3(F.silu(self.w1(x_norm)) * self.w2(x_norm))
        else:
            x = x + self.mlp(self.norm2(x))
        if self.gcm is not None:
            x = self.gcm(x, x_size)
        if self.freqmod is not None:
            x = self.freqmod(x, x_size)
        if self.csn is not None:
            x = self.csn(x, x_size)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: attention blocks + conv skip connection.

    attention_type: "swin" (window), "mdta" (channel), "ocab" (overlap spatial)
    """

    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=2,
                 activation="gelu", window_shift_ratio=0.5, skip_type="standard",
                 conv_kernel=3, attention_type="swin", use_gcm=False,
                 use_freqmod=False, use_csn=False):
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
                          mlp_ratio=mlp_ratio, activation=activation,
                          use_gcm=use_gcm, use_freqmod=use_freqmod,
                          use_csn=use_csn)
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
                 conv_kernel=3, attention_type="swin", use_gcm=False,
                 use_channel_curve=False, use_color_mlp=False,
                 use_freqmod=False, use_color_pre=False, use_dual_branch=False,
                 use_pcp=False, use_csn=False, use_color_mlp_output=False):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim

        self.channel_curve = ChannelCurve(in_ch) if use_channel_curve else None
        self.color_mlp = ColorMLP(in_ch) if use_color_mlp else None
        self.color_pre = ColorPre(hidden=16) if use_color_pre else None
        self.color_mlp_output = ColorMLP(in_ch, hidden=8) if use_color_mlp_output else None

        # Shared PCP: one instance applied between RSTB stages
        self.pcp = PCP(embed_dim) if use_pcp else None

        self.use_dual_branch = use_dual_branch
        if use_dual_branch:
            # Global branch: GAP on input → MLP → per-channel affine params
            self.global_branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_ch, in_ch * 2),
                nn.ReLU(),
                nn.Linear(in_ch * 2, embed_dim * 2),
            )
            nn.init.zeros_(self.global_branch[-1].weight)
            nn.init.zeros_(self.global_branch[-1].bias)
            # Gating for fusing global modulation with local features
            self.gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, embed_dim),
                nn.Sigmoid(),
            )
        else:
            self.global_branch = None

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
                     window_shift_ratio, skip_type, conv_kernel, attention_type,
                     use_gcm=use_gcm, use_freqmod=use_freqmod,
                     use_csn=use_csn)
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

        if self.channel_curve is not None:
            x = self.channel_curve(x)
        if self.color_mlp is not None:
            x = self.color_mlp(x)
        if self.color_pre is not None:
            x = self.color_pre(x)

        # Global branch: extract global statistics and predict modulation
        global_scale = None
        global_shift = None
        if self.global_branch is not None:
            global_params = self.global_branch(x)
            global_scale, global_shift = global_params.chunk(2, dim=1)
            # global_scale, global_shift: (B, embed_dim)

        shallow = self.conv_first(x)
        B, _, H, W = shallow.shape

        x_seq = shallow.flatten(2).transpose(1, 2)

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                x_seq = x_seq.transpose(1, 2).view(B, -1, H, W)
                x_seq = layer(x_seq).flatten(2).transpose(1, 2)
            else:
                x_seq = layer(x_seq, (H, W))
                # Apply shared PCP after each RSTB
                if self.pcp is not None:
                    x_seq = self.pcp(x_seq, (H, W))
                # Apply global modulation after each RSTB if dual branch
                if self.global_branch is not None and global_scale is not None:
                    feat_2d = x_seq.transpose(1, 2).view(B, -1, H, W)
                    g = self.gate(feat_2d.mean(dim=[2, 3]))
                    modulation = feat_2d * (1.0 + global_scale.unsqueeze(-1).unsqueeze(-1)) + global_shift.unsqueeze(-1).unsqueeze(-1)
                    feat_2d = feat_2d + g.unsqueeze(-1).unsqueeze(-1) * (modulation - feat_2d)
                    x_seq = feat_2d.flatten(2).transpose(1, 2)

        x_seq = self.norm(x_seq)
        deep_feat = x_seq.transpose(1, 2).view(B, self._final_dim, H, W)

        if self._shallow_proj is not None:
            shallow = self._shallow_proj(shallow)

        out = self.conv_after_body(deep_feat) + shallow
        out = self.conv_last(out) + x[:, :3, :, :]

        # Output-level color correction (after spatial restoration)
        if self.color_mlp_output is not None:
            out = self.color_mlp_output(out)

        return out[:, :, :H_in, :W_in]
