from typing import Optional

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def validate_prediction_shapes(
    generated_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Validate that prediction and target tensors can be compared directly."""

    if generated_predictions.shape != targets.shape:
        raise ValueError("The generated predictions and targets must be the same shape.")


def resolve_ssim_data_range(
    generated_predictions: torch.Tensor,
    targets: torch.Tensor,
    data_range: Optional[float] = None,
) -> torch.Tensor | float:
    """Return a positive SSIM data range for the current tensors."""

    if data_range is not None:
        return data_range

    batch_max = torch.maximum(generated_predictions.max(), targets.max())
    batch_min = torch.minimum(generated_predictions.min(), targets.min())
    return (batch_max - batch_min).clamp_min(
        torch.finfo(generated_predictions.dtype).eps
    )


def compute_l1_ssim_mean_components(
    generated_predictions: torch.Tensor,
    targets: torch.Tensor,
    ssim_weight: float,
    data_range: Optional[float] = None,
) -> dict[str, torch.Tensor]:
    """Compute mean L1, SSIM loss, and total loss for one batch."""

    validate_prediction_shapes(
        generated_predictions=generated_predictions,
        targets=targets,
    )
    l1 = torch.nn.functional.l1_loss(generated_predictions, targets, reduction="mean")
    ssim = structural_similarity_index_measure(
        preds=generated_predictions,
        target=targets,
        data_range=resolve_ssim_data_range(
            generated_predictions=generated_predictions,
            targets=targets,
            data_range=data_range,
        ),
    )
    ssim_loss = 1.0 - ssim
    total = l1 + ssim_weight * ssim_loss
    return {"l1": l1, "ssim": ssim_loss, "total": total}


def compute_l1_per_sample(
    generated_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute one mean absolute error value per sample."""

    validate_prediction_shapes(
        generated_predictions=generated_predictions,
        targets=targets,
    )
    abs_error = torch.abs(generated_predictions - targets)
    return abs_error.reshape(abs_error.shape[0], -1).mean(dim=1)


def compute_ssim_loss_per_sample(
    generated_predictions: torch.Tensor,
    targets: torch.Tensor,
    data_range: Optional[float] = None,
) -> torch.Tensor:
    """Compute one ``1 - SSIM`` value per sample."""

    validate_prediction_shapes(
        generated_predictions=generated_predictions,
        targets=targets,
    )
    per_sample_ssim = structural_similarity_index_measure(
        preds=generated_predictions,
        target=targets,
        data_range=resolve_ssim_data_range(
            generated_predictions=generated_predictions,
            targets=targets,
            data_range=data_range,
        ),
        reduction="none",
    ).reshape(-1)
    finite_ssim = torch.where(
        torch.isfinite(per_sample_ssim),
        per_sample_ssim,
        torch.tensor(0.0, device=generated_predictions.device, dtype=generated_predictions.dtype),
    )
    return 1.0 - finite_ssim
