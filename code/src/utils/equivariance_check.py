"""Equivariance verification for encoder architectures.

Tests approximate output-space invariance f(g(x)) ≈ f(x) under each
encoder's canonical symmetry group using random network weights.

Groups tested
-------------
D4   — OcticViT (8 elements: 4 rotations x 2 reflections)
Z2   — ShiftEquivariantViT (cyclic pixel shifts)
SO(2)— Harmformer (discrete orbit-averaging; exact on n_rots grid)

ViT baseline is included as a negative control (expected to fail).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import vit_l_16

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EquivarianceResult:
    """Per-encoder invariance check summary."""

    encoder_name: str
    group: str
    mean_error: float
    max_error: float
    passed: bool
    per_transform: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transform generators
# ---------------------------------------------------------------------------

def _d4_transforms(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """All 7 non-identity D4 elements: {r, r2, r3, s, sr, sr2, sr3}."""
    result: Dict[str, torch.Tensor] = {}
    for k in range(1, 4):
        result[f"r{k * 90}"] = TF.rotate(
            x, 90.0 * k, interpolation=TF.InterpolationMode.BILINEAR
        )
    for k in range(4):
        base = (
            x if k == 0
            else TF.rotate(x, 90.0 * k, interpolation=TF.InterpolationMode.BILINEAR)
        )
        result[f"s_r{k * 90}"] = TF.hflip(base)
    return result


def _z2_shifts(
    x: torch.Tensor,
    shift_pixels: List[Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    """Cyclic pixel shifts via torch.roll."""
    return {
        f"shift({dy:+d},{dx:+d})": torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        for dy, dx in shift_pixels
    }


def _so2_rotations(
    x: torch.Tensor,
    angles: List[float],
) -> Dict[str, torch.Tensor]:
    """Discrete rotations in [0, 360)."""
    return {
        f"rot{angle:.0f}": TF.rotate(
            x, angle, interpolation=TF.InterpolationMode.BILINEAR
        )
        for angle in angles
    }


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def _check_invariance(
    model: nn.Module,
    x: torch.Tensor,
    transforms: Dict[str, torch.Tensor],
    tol: float,
) -> Tuple[Dict[str, float], bool]:
    """Measure ||f(g(x)) - f(x)||_2 averaged over the batch for each transform.

    Args:
        model: Encoder in inference mode.
        x: Reference input [B, C, H, W].
        transforms: Mapping from name to transformed input.
        tol: L2 error threshold for a passing verdict.

    Returns:
        per_transform: per-transform mean L2 error.
        passed: True if every error is below tol.
    """
    model.train(False)
    with torch.no_grad():
        z_ref = model(x)

    errors: Dict[str, float] = {}
    for name, x_g in transforms.items():
        with torch.no_grad():
            z_g = model(x_g)
        errors[name] = (z_g - z_ref).norm(dim=-1).mean().item()

    return errors, all(e < tol for e in errors.values())


# ---------------------------------------------------------------------------
# Random-weight encoder constructors
# ---------------------------------------------------------------------------

def _random_vit() -> nn.Module:
    """ViT-L/16 with random weights, classification head replaced by Identity."""
    model = vit_l_16(weights=None)
    model.heads = nn.Identity()
    return model


def _build_harmformer(n_rots: int = 8) -> nn.Module:
    """Harmformer with random ViT-L/16 backbone (orbit-mean pooling)."""
    backbone = _random_vit()

    class _Harmformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.n_rots = n_rots

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            embs = []
            for k in range(self.n_rots):
                angle = 360.0 * k / self.n_rots
                x_r = (
                    x if angle == 0.0
                    else TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR)
                )
                embs.append(self.backbone(x_r))
            return torch.stack(embs, dim=1).mean(dim=1)

    return _Harmformer()


def _build_shift_eq(n_shifts: int = 4) -> Tuple[nn.Module, bool]:
    """Shift-equivariant encoder with random weights.

    Returns:
        (model, is_native): is_native=True when shift_eq_vit library is present.
    """
    try:
        from shift_eq_vit import shift_equivariant_vit_large
        return shift_equivariant_vit_large(), True
    except ImportError:
        logger.warning(
            "shift_eq_vit not installed — Z2 test uses diagonal-averaging fallback."
        )

    backbone = _random_vit()

    class _ShiftAvg(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.n_shifts = n_shifts

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, w = x.shape[-2], x.shape[-1]
            step_h, step_w = h // self.n_shifts, w // self.n_shifts
            embs = []
            for i in range(self.n_shifts):
                shifted = torch.roll(
                    x, shifts=(i * step_h, i * step_w), dims=(-2, -1)
                )
                embs.append(self.backbone(shifted))
            return torch.stack(embs, dim=1).mean(dim=1)

    return _ShiftAvg(), False


def _build_octic() -> Optional[nn.Module]:
    """Octic ViT with random weights, or None if octic_vits is unavailable."""
    try:
        from octic_vits import octic_deit3_large_patch16_224
        return octic_deit3_large_patch16_224()
    except ImportError:
        logger.warning("octic_vits not installed — D4 verification skipped.")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def verify_all_encoders(
    device: Optional[torch.device] = None,
    tol: float = 1e-3,
    input_size: int = 224,
    batch_size: int = 1,
    n_rots: int = 8,
    n_shifts: int = 4,
    seed: int = 42,
) -> Dict[str, EquivarianceResult]:
    """Verify approximate invariance for all equivariant encoders.

    All models use random weights — no pretrained checkpoints are loaded.
    ViT baseline is run as a negative control and always reports ``passed=False``.

    Shift-equivariant fallback (when shift_eq_vit is absent) is tested only
    on the diagonal shifts it actually symmetrizes, not arbitrary shifts.

    Args:
        device: Computation device. Defaults to CPU.
        tol: L2 error threshold to declare a test passed.
        input_size: Spatial resolution of the synthetic input image.
        batch_size: Number of random images per test.
        n_rots: Discrete rotations used by Harmformer orbit averaging.
        n_shifts: Diagonal shifts for the ShiftEq fallback.
        seed: RNG seed for reproducibility.

    Returns:
        Mapping from encoder name to EquivarianceResult.
    """
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    x = torch.rand(batch_size, 3, input_size, input_size, device=device)
    results: Dict[str, EquivarianceResult] = {}

    # --- Harmformer: SO(2) ---
    logger.info("Checking HarmformerEncoder (SO(2)) ...")
    harmformer = _build_harmformer(n_rots=n_rots).to(device)
    harmformer.train(False)
    so2_angles = [360.0 * k / n_rots for k in range(1, n_rots)]
    errors, passed = _check_invariance(harmformer, x, _so2_rotations(x, so2_angles), tol)
    results["harmformer"] = EquivarianceResult(
        encoder_name="harmformer",
        group="SO(2)",
        mean_error=sum(errors.values()) / len(errors),
        max_error=max(errors.values()),
        passed=passed,
        per_transform=errors,
    )
    _log_result(results["harmformer"])

    # --- ShiftEquivariant: Z2 ---
    logger.info("Checking ShiftEquivariantEncoder (Z2) ...")
    shift_enc, is_native = _build_shift_eq(n_shifts=n_shifts)
    shift_enc = shift_enc.to(device)
    shift_enc.train(False)
    if is_native:
        step = input_size // 8
        shift_pixels: List[Tuple[int, int]] = [
            (step, 0), (-step, 0), (0, step), (0, -step),
            (step, step), (2 * step, 3 * step), (-step, 2 * step),
        ]
    else:
        # only the diagonal steps the averaging symmetrizes
        step = input_size // n_shifts
        shift_pixels = [(k * step, k * step) for k in range(1, n_shifts)]
    errors, passed = _check_invariance(shift_enc, x, _z2_shifts(x, shift_pixels), tol)
    results["shift_eq_vit"] = EquivarianceResult(
        encoder_name="shift_eq_vit",
        group="Z2",
        mean_error=sum(errors.values()) / len(errors),
        max_error=max(errors.values()),
        passed=passed,
        per_transform=errors,
    )
    _log_result(results["shift_eq_vit"])

    # --- OcticViT: D4 ---
    logger.info("Checking OcticViTEncoder (D4) ...")
    octic = _build_octic()
    if octic is not None:
        octic = octic.to(device)
        octic.train(False)
        errors, passed = _check_invariance(octic, x, _d4_transforms(x), tol)
        results["octic_vit"] = EquivarianceResult(
            encoder_name="octic_vit",
            group="D4",
            mean_error=sum(errors.values()) / len(errors),
            max_error=max(errors.values()),
            passed=passed,
            per_transform=errors,
        )
        _log_result(results["octic_vit"])

    # --- ViT baseline: negative control ---
    logger.info("Checking ViT baseline (negative control) ...")
    vit = _random_vit().to(device)
    vit.train(False)
    errors, _ = _check_invariance(vit, x, _d4_transforms(x), tol)
    results["vit_baseline"] = EquivarianceResult(
        encoder_name="vit_baseline",
        group="D4 (negative control)",
        mean_error=sum(errors.values()) / len(errors),
        max_error=max(errors.values()),
        passed=False,
        per_transform=errors,
    )
    logger.info(
        "[CONTROL] vit_baseline  mean_err=%.2e  max_err=%.2e  (expected FAIL)",
        results["vit_baseline"].mean_error,
        results["vit_baseline"].max_error,
    )

    return results


def report_equivariance(results: Dict[str, EquivarianceResult]) -> str:
    """Render a plain-text summary table of all equivariance results."""
    header = (
        f"{'Encoder':<25} {'Group':<24} {'Mean err':>12} {'Max err':>12} {'Status':>8}"
    )
    rows = [header, "-" * 84]
    for r in results.values():
        if r.group.endswith("negative control)"):
            status = "n/a"
        else:
            status = "PASSED" if r.passed else "FAILED"
        rows.append(
            f"{r.encoder_name:<25} {r.group:<24}"
            f" {r.mean_error:>12.4e} {r.max_error:>12.4e} {status:>8}"
        )
    return "\n".join(rows)


def _log_result(result: EquivarianceResult) -> None:
    status = "PASSED" if result.passed else "FAILED"
    logger.info(
        "[%s] %s (%s)  mean_err=%.2e  max_err=%.2e",
        status, result.encoder_name, result.group,
        result.mean_error, result.max_error,
    )
    for name, err in result.per_transform.items():
        logger.debug("  %-32s err=%.4e", name, err)
