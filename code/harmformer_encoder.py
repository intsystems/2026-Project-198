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

def _magnitude(real, imag, eps=1e-6):
    """sqrt(real² + imag² + eps) computed in fp32 to avoid fp16 overflow / NaN
    on V100 / Turing under autocast. Inputs may be fp16/bf16; output matches input dtype.
    """
    out = torch.sqrt(real.float() ** 2 + imag.float() ** 2 + eps)
    return out.to(real.dtype)


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

    W_m(r, θ) = R(r) · e^{i(mθ + β)}.  R(r) depends ONLY on radius:
    parameterised as a sum of Gaussians centered on radial rings
    (Worrall et al. 2017, H-Net).  β is a learnable phase per output channel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, order):
        super().__init__()
        center = kernel_size // 2
        n_rings = center + 1                  # r = 0, 1, ..., center

        # Learnable radial weights (one scalar per ring per (out, in) pair)
        self.radial = nn.Parameter(torch.empty(out_channels, in_channels, n_rings))
        nn.init.kaiming_uniform_(self.radial, a=math.sqrt(5))
        self.phase = nn.Parameter(torch.zeros(out_channels))

        # --- precompute angular basis and Gaussian ring basis on the grid ---
        y, x = torch.meshgrid(
            torch.arange(kernel_size, dtype=torch.float32) - center,
            torch.arange(kernel_size, dtype=torch.float32) - center,
            indexing='ij',
        )
        theta = torch.atan2(y, x)
        r = torch.sqrt(x ** 2 + y ** 2)

        mask = (r <= center + 0.5).float()
        if order != 0:                        # order ≠ 0 → zero at r = 0
            mask = mask * (r > 0.5).float()

        # Gaussian ring basis: B[k, H, W] = exp(-(r - k)^2 / (2σ²)) · mask
        sigma = 0.5
        rings = torch.arange(n_rings, dtype=torch.float32)
        ring_basis = torch.exp(
            -((r.unsqueeze(0) - rings.view(-1, 1, 1)) ** 2) / (2 * sigma ** 2)
        ) * mask                              # [n_rings, K, K]

        self.register_buffer('basis_r', torch.cos(order * theta) * mask)
        self.register_buffer('basis_i', torch.sin(order * theta) * mask)
        self.register_buffer('ring_basis', ring_basis)
        self.register_buffer('mask', mask)

    def get_filter(self):
        """Return (filter_real, filter_imag), each [O, I, K, K]."""
        # R(r) by combining radial weights with Gaussian rings
        # radial: [O, I, n_rings]  ring_basis: [n_rings, K, K]
        R = torch.einsum('oik,khw->oihw', self.radial, self.ring_basis)

        pr = torch.cos(self.phase).view(-1, 1, 1, 1)
        pi = torch.sin(self.phase).view(-1, 1, 1, 1)
        # e^{i(mθ+β)} = e^{imθ} · e^{iβ}
        cr = self.basis_r * pr - self.basis_i * pi
        ci = self.basis_r * pi + self.basis_i * pr
        return R * cr, R * ci


# ================================================================
#  Harmonic Convolution Layer  (Eq. 9)
#  F_m^out = Σ_{m1+m2=m} F_m1^in ⊛ W_m2
# ================================================================

class HarmonicConvLayer(nn.Module):
    """Full harmonic convolution that mixes all three rotation-order streams.

    Filters are stacked into one big kernel so the whole layer becomes
    2 (lifting) or 4 (non-lifting) `F.conv2d` calls instead of 6 / 36.

    `lifting=True` for the very first layer (real image → complex streams).
    """

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 lifting=False):
        super().__init__()
        self.lifting = lifting
        self.padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels

        if lifting:
            # 3 filters (m ∈ {-1, 0, 1}); input is real
            self.filters = nn.ModuleList([
                HarmonicFilter(in_channels, out_channels, kernel_size, m)
                for m in ORDERS
            ])
        else:
            # 9 filters indexed [m_out_idx * 3 + m_in_idx]
            self.filters = nn.ModuleList([
                HarmonicFilter(in_channels, out_channels, kernel_size,
                               m_out - m_in)
                for m_out in ORDERS for m_in in ORDERS
            ])

    def _big_filter(self):
        """Stack per-pair filters once into a single big kernel.
        lifting:    [3·O, I, K, K]
        non-lift:   [3·O, 3·I, K, K]  with rows = m_out, cols = m_in
        """
        if self.lifting:
            rs, is_ = [], []
            for f in self.filters:
                fr, fi = f.get_filter()
                rs.append(fr); is_.append(fi)
            return torch.cat(rs, dim=0), torch.cat(is_, dim=0)

        rows_r, rows_i = [], []
        for idx_out in range(3):
            cells_r, cells_i = [], []
            for idx_in in range(3):
                fr, fi = self.filters[idx_out * 3 + idx_in].get_filter()
                cells_r.append(fr); cells_i.append(fi)
            rows_r.append(torch.cat(cells_r, dim=1))     # [O, 3·I, K, K]
            rows_i.append(torch.cat(cells_i, dim=1))
        return torch.cat(rows_r, dim=0), torch.cat(rows_i, dim=0)

    def forward(self, x):
        p = self.padding
        O = self.out_channels
        big_r, big_i = self._big_filter()

        if self.lifting:
            # x is real [B, I, H, W] → 2 conv2d calls
            out_r_all = F.conv2d(x, big_r, padding=p)    # [B, 3·O, H, W]
            out_i_all = F.conv2d(x, big_i, padding=p)
            return {m: (out_r_all[:, i*O:(i+1)*O],
                        out_i_all[:, i*O:(i+1)*O])
                    for i, m in enumerate(ORDERS)}

        # Non-lifting: stack inputs along channels → 4 conv2d calls total
        xr_big = torch.cat([x[m][0] for m in ORDERS], dim=1)  # [B, 3·I, H, W]
        xi_big = torch.cat([x[m][1] for m in ORDERS], dim=1)

        # (xr + i·xi) ⊛ (Wr + i·Wi) = (xr⊛Wr − xi⊛Wi) + i(xr⊛Wi + xi⊛Wr)
        rr = F.conv2d(xr_big, big_r, padding=p)
        ii = F.conv2d(xi_big, big_i, padding=p)
        ri = F.conv2d(xr_big, big_i, padding=p)
        ir = F.conv2d(xi_big, big_r, padding=p)

        out_r_all = rr - ii
        out_i_all = ri + ir
        return {m: (out_r_all[:, i*O:(i+1)*O],
                    out_i_all[:, i*O:(i+1)*O])
                for i, m in enumerate(ORDERS)}


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
        mag = _magnitude(real, imag, self.eps)
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
            # Mean only invariant for m = 0; for m ≠ 0 the spatial mean
            # transforms by e^{imφ} under rotation, so we cannot subtract it.
            if m == 0:
                r = r - r.mean(dim=1, keepdim=True)
                i = i - i.mean(dim=1, keepdim=True)
            mag = _magnitude(r, i, self.eps)
            sigma = mag.std(dim=1, keepdim=True) + self.eps
            out[m] = (r / sigma * self.gamma,
                      i / sigma * self.gamma)
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
        mag = _magnitude(real, imag, self.eps)
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

        # Q, K, V projections SHARED across rotation orders (Lemma 5.2):
        # the same real-valued W is applied to every stream m.
        self.q = HarmonicLinear(dim, dim)
        self.k = HarmonicLinear(dim, dim)
        self.v = HarmonicLinear(dim, dim)

        # output projection shared across orders
        self.proj = HarmonicLinear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)

        # Distance-based circular RPE (rotation-invariant):
        # one learnable scalar per (head, integer-distance bin).
        side = int(round(math.sqrt(num_patches)))
        assert side * side == num_patches, "num_patches must be a perfect square"
        ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side),
                                indexing='ij')
        pos = torch.stack([ys.flatten(), xs.flatten()], dim=-1).float()
        d = torch.cdist(pos, pos)              # [N, N]
        dist_idx = torch.round(d).long()
        n_bins = int(dist_idx.max().item()) + 1
        self.register_buffer('dist_idx', dist_idx)
        self.rpe_table = nn.Parameter(torch.zeros(num_heads, n_bins))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, streams):
        B, N, D = streams[0][0].shape
        Hh, hd = self.num_heads, self.head_dim

        # Stack 3 streams: [3, B, N, D]
        r_in = torch.stack([streams[m][0] for m in ORDERS], dim=0)
        i_in = torch.stack([streams[m][1] for m in ORDERS], dim=0)

        # Shared Q/K/V across orders → broadcast matmul on leading dim
        qr = r_in @ self.q.weight; qi = i_in @ self.q.weight
        kr = r_in @ self.k.weight; ki = i_in @ self.k.weight
        vr = r_in @ self.v.weight; vi = i_in @ self.v.weight

        # → heads: [3, B, H, N, hd]
        def to_heads(t):
            return t.reshape(3, B, N, Hh, hd).permute(0, 1, 3, 2, 4)
        qr, qi = to_heads(qr), to_heads(qi)
        kr, ki = to_heads(kr), to_heads(ki)
        vr, vi = to_heads(vr), to_heads(vi)

        # Σ_m Q_m · conj(K_m)^T  — sum collapses the order axis
        # qr/kr: [3, B, H, N, hd]   target: [B, H, N, N]
        attn_r = (torch.einsum('mbhnd,mbhkd->bhnk', qr, kr)
                  + torch.einsum('mbhnd,mbhkd->bhnk', qi, ki))
        attn_i = (torch.einsum('mbhnd,mbhkd->bhnk', qi, kr)
                  - torch.einsum('mbhnd,mbhkd->bhnk', qr, ki))

        # Force fp32 for magnitude + softmax — most NaN-prone block in fp16.
        attn_mag = _magnitude(attn_r, attn_i, 1e-6).float()
        rpe_bias = self.rpe_table[:, self.dist_idx].unsqueeze(0)  # [1,H,N,N]
        attn_mag = attn_mag * self.scale + rpe_bias.float()
        attn = F.softmax(attn_mag, dim=-1).to(attn_r.dtype)
        attn = self.attn_drop(attn)

        # A_0 × V_m — broadcast attn over the order axis
        # attn: [B, H, N, N], v: [3, B, H, N, hd]
        o_r = torch.einsum('bhnk,mbhkd->mbhnd', attn, vr)
        o_i = torch.einsum('bhnk,mbhkd->mbhnd', attn, vi)
        # → [3, B, N, D]
        o_r = o_r.permute(0, 1, 3, 2, 4).reshape(3, B, N, D)
        o_i = o_i.permute(0, 1, 3, 2, 4).reshape(3, B, N, D)

        # Shared output projection
        o_r = o_r @ self.proj.weight
        o_i = o_i @ self.proj.weight

        return {m: (o_r[idx], o_i[idx]) for idx, m in enumerate(ORDERS)}


# ================================================================
#  Harmonic MLP  (Fig. 4a, MLP Part)
#  HLinear → C-ReLU → HLinear   (per order, no cross-stream mixing)
# ================================================================

class HarmonicMLP(nn.Module):
    """Per-order HLinear → C-ReLU → HLinear, vectorised across the 3 orders
    via einsum on stacked weights."""

    def __init__(self, dim, expansion=2, dropout=0.0):
        super().__init__()
        hidden = dim * expansion
        # Stacked per-order weights: [3, in, out]
        self.w1 = nn.Parameter(torch.empty(3, dim, hidden))
        self.w2 = nn.Parameter(torch.empty(3, hidden, dim))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        # CReLU per-order params: [3, hidden]
        self.a = nn.Parameter(torch.ones(3, hidden))
        self.b = nn.Parameter(torch.zeros(3, hidden))
        self.eps = 1e-8
        self.drop = nn.Dropout(dropout)

    def forward(self, streams):
        # Stack 3 streams along leading dim: [3, B, N, D]
        r = torch.stack([streams[m][0] for m in ORDERS], dim=0)
        i = torch.stack([streams[m][1] for m in ORDERS], dim=0)

        # fc1
        r1 = torch.einsum('mbnd,mdh->mbnh', r, self.w1)
        i1 = torch.einsum('mbnd,mdh->mbnh', i, self.w1)

        # C-ReLU per order (a, b broadcast over [B, N])
        mag = _magnitude(r1, i1, self.eps)
        a = self.a.view(3, 1, 1, -1)
        b = self.b.view(3, 1, 1, -1)
        mag_new = F.relu(a * mag + b)
        scale = mag_new / mag
        r1 = r1 * scale
        i1 = i1 * scale

        # fc2
        r2 = torch.einsum('mbnh,mhd->mbnd', r1, self.w2)
        i2 = torch.einsum('mbnh,mhd->mbnd', i1, self.w2)

        return {m: (self.drop(r2[idx]), self.drop(i2[idx]))
                for idx, m in enumerate(ORDERS)}


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
            mags.append(_magnitude(r, i, 1e-6))
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
