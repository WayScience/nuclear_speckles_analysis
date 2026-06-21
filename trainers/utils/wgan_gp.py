from collections.abc import Callable

import torch


def compute_wgan_gp_primitives(
    critic: Callable[[torch.Tensor], torch.Tensor],
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute interpolation and critic outputs for WGAN-GP.

    Args:
        critic: Critic/discriminator callable that maps samples to scores.
        real_samples: Batch of real samples.
        fake_samples: Batch of generated samples.

    Returns:
        Dictionary containing:
            - gradients: d(critic(interpolated))/d(interpolated) per sample.
            - real_classification_outputs: Critic outputs for real samples.
            - fake_classification_outputs: Critic outputs for fake samples.

    Raises:
        ValueError: If real and fake batch sizes differ.
        ValueError: If critic outputs are empty.
    """

    if real_samples.size(0) != fake_samples.size(0):
        raise ValueError("real_samples and fake_samples must have matching batch size.")

    batch_size = real_samples.size(0)
    alpha_shape = (batch_size,) + (1,) * (real_samples.dim() - 1)
    alpha = torch.rand(alpha_shape, device=real_samples.device, dtype=real_samples.dtype)

    interpolated_samples = (
        alpha * real_samples + (1.0 - alpha) * fake_samples
    ).requires_grad_(True)

    interpolated_outputs = critic(interpolated_samples)
    real_classification_outputs = critic(real_samples)
    fake_classification_outputs = critic(fake_samples)

    if interpolated_outputs.numel() == 0:
        raise ValueError("critic output for interpolated samples must not be empty.")

    grad_outputs = torch.ones_like(interpolated_outputs)
    gradients = torch.autograd.grad(
        outputs=interpolated_outputs,
        inputs=interpolated_samples,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return {
        "gradients": gradients,
        "real_classification_outputs": real_classification_outputs,
        "fake_classification_outputs": fake_classification_outputs,
    }


def compute_wgan_gp_components(
    gradients: torch.Tensor,
    real_classification_outputs: torch.Tensor,
    fake_classification_outputs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute unweighted WGAN-GP components from precomputed primitives.

    Args:
        gradients: d(critic(interpolated))/d(interpolated) per sample.
        real_classification_outputs: Critic outputs for real samples.
        fake_classification_outputs: Critic outputs for fake samples.

    Returns:
        Dictionary containing:
            - wasserstein_term: mean(fake) - mean(real).
            - gradient_penalty_unweighted: mean((||grad||_2 - 1)^2).

    Raises:
        ValueError: If critic output batch sizes do not match gradients batch size.
    """

    batch_size = gradients.size(0)
    if real_classification_outputs.size(0) != batch_size:
        raise ValueError("real_classification_outputs batch size must match gradients.")
    if fake_classification_outputs.size(0) != batch_size:
        raise ValueError("fake_classification_outputs batch size must match gradients.")

    flat_gradients = gradients.view(batch_size, -1)
    gradient_penalty_unweighted = ((flat_gradients.norm(2, dim=1) - 1) ** 2).mean()
    wasserstein_term = torch.mean(fake_classification_outputs) - torch.mean(
        real_classification_outputs
    )

    return {
        "wasserstein_term": wasserstein_term,
        "gradient_penalty_unweighted": gradient_penalty_unweighted,
    }


def compute_wgan_gp_terms(
    critic: Callable[[torch.Tensor], torch.Tensor],
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute WGAN-GP primitives and derived unweighted components.

    This convenience wrapper preserves a one-call API and combines
    ``compute_wgan_gp_primitives`` and ``compute_wgan_gp_components``.
    """

    primitives = compute_wgan_gp_primitives(
        critic=critic,
        real_samples=real_samples,
        fake_samples=fake_samples,
    )
    components = compute_wgan_gp_components(**primitives)
    return primitives | components


def compute_generator_components(
    fake_classification_outputs: torch.Tensor,
    generated_predictions: torch.Tensor,
    targets: torch.Tensor,
    epoch: int = 0,
    use_adversarial_decay: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute unconditional WGAN generator terms with optional epoch decay."""

    if generated_predictions.shape != targets.shape:
        raise ValueError("generated_predictions and targets must have the same shape.")

    batch_size = generated_predictions.size(0)
    if fake_classification_outputs.size(0) != batch_size:
        raise ValueError(
            "fake_classification_outputs batch size must match generated_predictions."
        )

    reconstruction_term = torch.nn.functional.l1_loss(
        generated_predictions, targets, reduction="mean"
    )
    adversarial_term = torch.mean(fake_classification_outputs)
    if use_adversarial_decay:
        adversarial_term = adversarial_term / (epoch + 1)

    return {
        "reconstruction_term": reconstruction_term,
        "adversarial_term": adversarial_term,
    }
