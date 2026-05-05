"""
Harmformer Encoder — roto-translation equivariant ViT
Based on: Karella et al., "Harmformer: Harmonic Networks Meet Transformers
for Continuous Roto-Translation Equivariance", arXiv:2411.03794

Features are stored as dict {order_m: (real, imag)} throughout the network,
where order_m ∈ {-1, 0, 1} is the rotation order.

Layer-by-layer correspondence to standard ViT:
    ViT Patch Embedding      → S1 Stem (H-Conv blocks) + S2 Patch Construction
    Absolute Pos Encoding     → Circular Relative Position Encoding (in MSA)
    Linear Q/K/V              → HarmonicLinear (real weights, per order)
    Dot Product QK^T          → Harmonic Dot Product (conjugate, subtracts orders)
    Softmax                   → Softmax on magnitudes (codomain R+)
    Attention × Values        → Real attention × complex values (adds orders)
    Multi-Head Self-Attention → HarmonicMSA (parallel orders, shared A_0)
    Layer Norm                → HarmonicLayerNorm (magnitude-based, per order)
    ReLU / GELU               → C-ReLU: ReLU(a|z|+b) * e^{iθ}
    Residual Connection       → Harmonic Residual (same order streams)
    MLP                       → HarmonicMLP (HLinear → C-ReLU → HLinear)
    GAP + Classification      → |magnitude| → concat orders → GAP → feature vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

ORDERS = [-1, 0, 1]


# ================================================================
#  Utilities
# ================================================================

def complex_conv2d(inp_r, inp_i, w_r, w_i, stride=1, padding=0):
    """Complex 2D convolution via real arithmetic.
    (a + bi) ⊛ (c + di) = (a⊛c − b⊛d) + i(a⊛d + b⊛c)
    """
    out_r = (F.conv2d(inp_r, w_r, stride=stride, padding=padding)
           - F.conv2d(inp_i, w_i, stride=stride, padding=padding))
    out_i = (F.conv2d(inp_r, w_i, stride=stride, padding=padding)
           + F.conv2d(inp_i, w_r, stride=stride, padding=padding))
    return out_r, out_i


# ================================================================
#  Harmonic Filter  (Def. 4.1)
#  W_m(r, θ) = R(r) · e^{i(mθ + β)}
# ================================================================

class HarmonicFilter(nn.Module):
    """Learnable harmonic filter for rotation order `order`.

    R(r) is parameterised as per-pixel learnable weights masked
    to a circular support.  β is a learnable phase per output channel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, order):
        super().__init__()
        center = kernel_size // 2

        # Learnable radial profile + per-channel phase
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels,
                                               kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.phase = nn.Parameter(torch.zeros(out_channels))

        # --- precompute angular basis on the discrete grid ---
        y, x = torch.meshgrid(
            torch.arange(kernel_size, dtype=torch.float32) - center,
            torch.arange(kernel_size, dtype=torch.float32) - center,
            indexing='ij',
        )
        theta = torch.atan2(y, x)
        r = torch.sqrt(x ** 2 + y ** 2)

        mask = (r <= center + 0.5).float()
        if order != 0:                       # order ≠ 0 → zero at r = 0
            mask = mask * (r > 0.5).float()

        self.register_buffer('basis_r', torch.cos(order * theta) * mask)
        self.register_buffer('basis_i', torch.sin(order * theta) * mask)
        self.register_buffer('mask', mask)

    def get_filter(self):
        """Return (filter_real, filter_imag), each [O, I, K, K]."""
        w = self.weight * self.mask
        pr = torch.cos(self.phase).view(-1, 1, 1, 1)
        pi = torch.sin(self.phase).view(-1, 1, 1, 1)
        # e^{i(mθ+β)} = e^{imθ} · e^{iβ}
        cr = self.basis_r * pr - self.basis_i * pi
        ci = self.basis_r * pi + self.basis_i * pr
        return w * cr, w * ci


# ================================================================
#  Harmonic Convolution Layer  (Eq. 9)
#  F_m^out = Σ_{m1+m2=m} F_m1^in ⊛ W_m2
# ================================================================

class HarmonicConvLayer(nn.Module):
    """Full harmonic convolution that mixes all three rotation-order streams.

    `lifting=True` for the very first layer (real image → complex streams).
    """

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 lifting=False):
        super().__init__()
        self.lifting = lifting
        self.padding = kernel_size // 2

        if lifting:
            # input is real (order 0) → three output streams
            self.filters = nn.ModuleDict({
                str(m): HarmonicFilter(in_channels, out_channels,
                                       kernel_size, m)
                for m in ORDERS
            })
        else:
            # 9 (m_in, m_out) pairs, filter order = m_out − m_in
            self.filters = nn.ModuleDict({
                f'{m_in}to{m_out}': HarmonicFilter(
                    in_channels, out_channels, kernel_size, m_out - m_in)
                for m_in in ORDERS for m_out in ORDERS
            })

    def forward(self, x):
        p = self.padding
        out = {}

        if self.lifting:
            for m in ORDERS:
                fr, fi = self.filters[str(m)].get_filter()
                out[m] = (F.conv2d(x, fr, padding=p),
                          F.conv2d(x, fi, padding=p))
        else:
            for m_out in ORDERS:
                acc_r = acc_i = 0
                for m_in in ORDERS:
                    fr, fi = self.filters[f'{m_in}to{m_out}'].get_filter()
                    ir, ii = x[m_in]
                    cr, ci = complex_conv2d(ir, ii, fr, fi, padding=p)
                    acc_r = acc_r + cr
                    acc_i = acc_i + ci
                out[m_out] = (acc_r, acc_i)

        return out


# ================================================================
#  HBatchNorm + C-ReLU  (Def. A.4, fused)
#  output = ReLU(a · BN(|z|) + b) · e^{iθ}
#
#  Operates only on magnitudes → phase untouched → HE preserved.
#  Codomain restricted to R⁺₀ (no negative magnitudes).
# ================================================================

class ComplexBNReLU(nn.Module):
    """For spatial feature maps [B, C, H, W] — used in the stem."""

    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels, affine=False)
        self.a = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = 1e-8

    def forward(self, real, imag):
        mag = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
        phase_r = real / mag
        phase_i = imag / mag

        B, C, H, W = mag.shape
        mag_bn = self.bn(mag.reshape(B, C, -1)).reshape(B, C, H, W)
        mag_act = F.relu(self.a * mag_bn + self.b)

        return mag_act * phase_r, mag_act * phase_i


# ================================================================
#  H-Conv Block  (Fig. 3b)
#  HarmonicConv → HBatchNorm + C-ReLU  (+ residual if same dims)
# ================================================================

class HConvBlock(nn.Module):

    def __init__(self, in_c, out_c, kernel_size=5, lifting=False):
        super().__init__()
        self.conv = HarmonicConvLayer(in_c, out_c, kernel_size, lifting)
        self.act = nn.ModuleDict({
            str(m): ComplexBNReLU(out_c) for m in ORDERS
        })
        self.residual = (not lifting) and (in_c == out_c)

    def forward(self, x):
        h = self.conv(x)
        out = {}
        for m in ORDERS:
            r, i = h[m]
            r, i = self.act[str(m)](r, i)
            if self.residual:
                xr, xi = x[m]
                r, i = r + xr, i + xi
            out[m] = (r, i)
        return out


# ================================================================
#  Harmonic Layer Norm  (Lemma 5.3)
#  Normalise per rotation-order stream over spatial dimension N.
#  µ and σ are rotation-invariant → HE preserved.
# ================================================================

class HarmonicLayerNorm(nn.Module):
    """For sequence features [B, N, D] — used in the encoder."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = 1e-8

    def forward(self, streams):
        out = {}
        for m in ORDERS:
            r, i = streams[m]                             # [B, N, D]
            mu_r = r.mean(dim=1, keepdim=True)
            mu_i = i.mean(dim=1, keepdim=True)
            rc, ic = r - mu_r, i - mu_i
            mag = torch.sqrt(rc ** 2 + ic ** 2 + self.eps)
            sigma = mag.std(dim=1, keepdim=True) + self.eps
            out[m] = (rc / sigma * self.gamma,
                      ic / sigma * self.gamma)
        return out


# ================================================================
#  Harmonic Linear  (Lemma 5.2)
#  F_m · W  with  W ∈ R^{d_in × d_out}  (order 0, real-valued)
#  No bias — a bias would break HE for m ≠ 0.
# ================================================================

class HarmonicLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, real, imag):
        return real @ self.weight, imag @ self.weight


# ================================================================
#  C-ReLU  (Def. A.2)
#  ReLU(a · |z| + b) · e^{iθ}
#  For encoder MLP activation.
# ================================================================

class CReLU(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-8

    def forward(self, real, imag):
        mag = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
        phase_r = real / mag
        phase_i = imag / mag
        mag_new = F.relu(self.a * mag + self.b)
        return mag_new * phase_r, mag_new * phase_i


# ================================================================
#  Harmonic MSA  (Section 5.3, Fig. 4b)
#
#  1) Q_m, K_m, V_m  per order via HarmonicLinear
#  2) A₀ = softmax(|Σ_m Q_m · conj(K_m)ᵀ|)
#     — dot product subtracts orders  (Lemma 5.4):  m − m = 0
#     — softmax on magnitudes, codomain R⁺₀
#  3) out_m = A₀ · V_m
#     — matmul adds orders  (Lemma 5.5):  0 + m = m
# ================================================================

class HarmonicMSA(nn.Module):

    def __init__(self, dim, num_heads, num_patches, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections per rotation order
        self.qkv = nn.ModuleDict()
        for m in ORDERS:
            self.qkv[str(m)] = nn.ModuleDict({
                'q': HarmonicLinear(dim, dim),
                'k': HarmonicLinear(dim, dim),
                'v': HarmonicLinear(dim, dim),
            })

        # output projection per order
        self.proj = nn.ModuleDict({
            str(m): HarmonicLinear(dim, dim) for m in ORDERS
        })

        self.attn_drop = nn.Dropout(dropout)

        # distance-based RPE (rotation-invariant)
        self.rpe = nn.Parameter(
            torch.zeros(1, num_heads, num_patches, num_patches))
        nn.init.trunc_normal_(self.rpe, std=0.02)

    def forward(self, streams):
        B, N, D = streams[0][0].shape
        H, hd = self.num_heads, self.head_dim

        def to_heads(t):
            return t.reshape(B, N, H, hd).transpose(1, 2)  # [B,H,N,hd]

        # ---- generate Q, K, V per order ----
        qkvs = {}
        for m in ORDERS:
            r, i = streams[m]
            qr, qi = self.qkv[str(m)]['q'](r, i)
            kr, ki = self.qkv[str(m)]['k'](r, i)
            vr, vi = self.qkv[str(m)]['v'](r, i)
            qkvs[m] = {
                'q': (to_heads(qr), to_heads(qi)),
                'k': (to_heads(kr), to_heads(ki)),
                'v': (to_heads(vr), to_heads(vi)),
            }

        # ---- attention: Σ_m  Q_m · conj(K_m)^T  → order 0 ----
        attn_r = 0
        attn_i = 0
        for m in ORDERS:
            qr, qi = qkvs[m]['q']
            kr, ki = qkvs[m]['k']
            # (qr + i qi)(kr − i ki)^T = (qr kr^T + qi ki^T) + i(…)
            attn_r = attn_r + (qr @ kr.transpose(-2, -1)
                               + qi @ ki.transpose(-2, -1))
            attn_i = attn_i + (qi @ kr.transpose(-2, -1)
                               - qr @ ki.transpose(-2, -1))

        # softmax on magnitudes (Harmformer softmax, codomain R⁺₀)
        attn_mag = torch.sqrt(attn_r ** 2 + attn_i ** 2 + 1e-8)
        attn_mag = attn_mag * self.scale + self.rpe

        attn = F.softmax(attn_mag, dim=-1)          # [B, H, N, N] real ≥ 0
        attn = self.attn_drop(attn)

        # ---- A₀ × V_m  → output order m ----
        out = {}
        for m in ORDERS:
            vr, vi = qkvs[m]['v']
            o_r = (attn @ vr).transpose(1, 2).reshape(B, N, D)
            o_i = (attn @ vi).transpose(1, 2).reshape(B, N, D)
            o_r, o_i = self.proj[str(m)](o_r, o_i)
            out[m] = (o_r, o_i)

        return out


# ================================================================
#  Harmonic MLP  (Fig. 4a, MLP Part)
#  HLinear → C-ReLU → HLinear   (per order, no cross-stream mixing)
# ================================================================

class HarmonicMLP(nn.Module):

    def __init__(self, dim, expansion=2, dropout=0.0):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.ModuleDict({
            str(m): HarmonicLinear(dim, hidden) for m in ORDERS})
        self.act = nn.ModuleDict({
            str(m): CReLU(hidden) for m in ORDERS})
        self.fc2 = nn.ModuleDict({
            str(m): HarmonicLinear(hidden, dim) for m in ORDERS})
        self.drop = nn.Dropout(dropout)

    def forward(self, streams):
        out = {}
        for m in ORDERS:
            r, i = streams[m]
            r, i = self.fc1[str(m)](r, i)
            r, i = self.act[str(m)](r, i)
            r, i = self.fc2[str(m)](r, i)
            out[m] = (self.drop(r), self.drop(i))
        return out


# ================================================================
#  Harmonic Encoder Block  (Fig. 4a)
#  Pre-norm style:
#     x  →  HLayerNorm → MSA  → + residual
#        →  HLayerNorm → MLP  → + residual
# ================================================================

class HarmonicEncoderBlock(nn.Module):

    def __init__(self, dim, num_heads, num_patches,
                 mlp_expansion=2, dropout=0.0):
        super().__init__()
        self.norm1 = HarmonicLayerNorm(dim)
        self.msa   = HarmonicMSA(dim, num_heads, num_patches, dropout)
        self.norm2 = HarmonicLayerNorm(dim)
        self.mlp   = HarmonicMLP(dim, mlp_expansion, dropout)

    def forward(self, streams):
        # ---- MSA + residual  (Lemma 5.1) ----
        normed = self.norm1(streams)
        attn   = self.msa(normed)
        res1 = {}
        for m in ORDERS:
            sr, si = streams[m]
            ar, ai = attn[m]
            res1[m] = (sr + ar, si + ai)

        # ---- MLP + residual ----
        normed2 = self.norm2(res1)
        mlp_out = self.mlp(normed2)
        res2 = {}
        for m in ORDERS:
            r1r, r1i = res1[m]
            mr, mi = mlp_out[m]
            res2[m] = (r1r + mr, r1i + mi)

        return res2


# ================================================================
#  HarmformerEncoder  —  drop-in replacement for ViTEncoder
#
#  S1  Stem Stage:        H-Conv blocks + AvgPool  (reduces spatial dims)
#  S2  Patch Construction: flatten + HarmonicLinear projection
#  S3  Harmonic Encoder:   k × (HarmonicMSA + HarmonicMLP)
#  S4  Invariant Output:  |magnitude| → concat 3 orders → global avg pool
#
#  Interface:
#      input:  [B, 3, 32, 32]  real
#      output: [B, feature_dim] real   (feature_dim = 3 × encoder_dim)
# ================================================================

class HarmformerEncoder(nn.Module):

    def __init__(
        self,
        img_size: int = 32,
        in_channels: int = 3,
        stem_channels: list = None,
        stem_kernel_size: int = 5,
        convs_per_block: int = 2,
        encoder_dim: int = 64,
        encoder_depth: int = 4,
        num_heads: int = 4,
        mlp_expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if stem_channels is None:
            stem_channels = [16, 32]

        self.encoder_dim = encoder_dim
        self.feature_dim = 3 * encoder_dim          # invariant output dim

        # -------- S1: Stem --------
        self.stem_convs = nn.ModuleList()
        self.pool_after = []                        # conv indices after which to pool

        ch = in_channels
        for block_idx, ch_out in enumerate(stem_channels):
            for conv_idx in range(convs_per_block):
                lifting = (block_idx == 0 and conv_idx == 0)
                c_in = ch if conv_idx == 0 else ch_out
                self.stem_convs.append(
                    HConvBlock(c_in, ch_out, stem_kernel_size, lifting=lifting))
            self.pool_after.append(len(self.stem_convs) - 1)
            ch = ch_out

        # -------- S2: Projection to encoder_dim --------
        self.patch_proj = nn.ModuleDict({
            str(m): HarmonicLinear(stem_channels[-1], encoder_dim)
            for m in ORDERS
        })

        spatial = img_size
        for _ in stem_channels:
            spatial //= 2
        self.num_patches = spatial * spatial         # e.g. 8×8 = 64

        # -------- S3: Encoder --------
        self.encoder = nn.ModuleList([
            HarmonicEncoderBlock(encoder_dim, num_heads, self.num_patches,
                                 mlp_expansion, dropout)
            for _ in range(encoder_depth)
        ])
        self.final_norm = HarmonicLayerNorm(encoder_dim)

        # -------- compatibility with SiamNet --------
        self.preprocess = nn.Identity()

    # -----------------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] real image tensor
        Returns:
            [B, feature_dim] real invariant embedding
        """

        # ---- S1: Stem (H-Conv blocks + AvgPool) ----
        h = x
        for idx, conv in enumerate(self.stem_convs):
            h = conv(h)
            if idx in self.pool_after:
                pooled = {}
                for m in ORDERS:
                    r, i = h[m]
                    pooled[m] = (F.avg_pool2d(r, 2), F.avg_pool2d(i, 2))
                h = pooled

        # ---- S2: Construct 1×1 patches + project ----
        streams = {}
        for m in ORDERS:
            r, i = h[m]                              # [B, C, H', W']
            r = r.flatten(2).transpose(1, 2)         # [B, N, C]
            i = i.flatten(2).transpose(1, 2)
            r, i = self.patch_proj[str(m)](r, i)     # [B, N, encoder_dim]
            streams[m] = (r, i)

        # ---- S3: Harmonic Encoder ----
        for block in self.encoder:
            streams = block(streams)

        streams = self.final_norm(streams)

        # ---- S4: Invariant output ----
        #   |magnitude| per order → concat → global average pool
        mags = []
        for m in ORDERS:
            r, i = streams[m]                        # [B, N, D]
            mags.append(torch.sqrt(r ** 2 + i ** 2 + 1e-8))
        features = torch.cat(mags, dim=-1)           # [B, N, 3D]
        features = features.mean(dim=1)              # [B, 3D]

        return features


# ================================================================
#  Quick test
# ================================================================

if __name__ == '__main__':
    from torchvision.transforms import functional as TF

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = HarmformerEncoder(
        img_size=32,
        stem_channels=[16, 32],
        encoder_dim=64,       # feature_dim = 3 × 64 = 192
        encoder_depth=4,
        num_heads=4,
    ).to(device)

    enc.eval()
    print(f'feature_dim = {enc.feature_dim}')
    print(f'num_patches = {enc.num_patches}')
    print(f'parameters  = {sum(p.numel() for p in enc.parameters()):,}')

    # --- equivariance test ---
    img = torch.randn(1, 3, 32, 32, device=device)
    img_rot = TF.rotate(img, 90, expand=False)

    with torch.no_grad():
        emb      = enc(img)
        emb_rot  = enc(img_rot)

    cos_sim = F.cosine_similarity(emb, emb_rot, dim=-1).item()
    l2_dist = (emb - emb_rot).norm().item()
    print(f'cos_sim(emb, emb_rot90) = {cos_sim:.6f}')
    print(f'L2 dist                 = {l2_dist:.6f}')
    print(f'allclose (atol=0.01)    = {torch.allclose(emb, emb_rot, atol=0.01)}')
