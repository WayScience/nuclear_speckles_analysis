from typing import Optional

import torch


def validate_prediction_shapes(
    generated_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Validate that prediction and target tensors can be compared directly."""

    if generated_predictions.shape != targets.shape:
        raise ValueError("The generated predictions and targets must be the same shape.")


def compute_gamma_weighted_l1_per_sample(
    generated_predictions: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    epsilon: float,
) -> torch.Tensor:
    """Compute one gamma-weighted absolute-error loss value per sample."""

    validate_prediction_shapes(
        generated_predictions=generated_predictions,
        targets=targets,
    )
    abs_error = torch.abs(generated_predictions - targets)
    weighted_abs_error = torch.pow(abs_error + epsilon, gamma) * abs_error
    return weighted_abs_error.reshape(weighted_abs_error.shape[0], -1).mean(dim=1)


def compute_gamma_weighted_l1_mean_components(
    generated_predictions: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    """Compute mean gamma-weighted absolute-error loss for one batch."""

    per_sample_loss = compute_gamma_weighted_l1_per_sample(
        generated_predictions=generated_predictions,
        targets=targets,
        gamma=gamma,
        epsilon=epsilon,
    )
    total = per_sample_loss.mean()
    return {"total": total}


def resolve_loss_description(gamma: Optional[float] = None, epsilon: Optional[float] = None) -> str:
    """Return a human-readable description of the gamma-weighted loss."""

    if gamma is None or epsilon is None:
        return "Gamma-weighted absolute-error loss"
    return (
        "Gamma-weighted absolute-error loss: "
        f"mean((abs(y - y_hat) + {epsilon})^{gamma} * abs(y - y_hat))"
    )
