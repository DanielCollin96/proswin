"""Weighted loss functions for the two-stage PROSWIN training protocol.

Stage 1 trains the mu (point-forecast) head using :func:`mean_squared_error_loss`.
Stage 2 trains the sigma (uncertainty) head using
:func:`negative_log_likelihood_loss`. Both functions accept per-sample focal
weights that emphasise hard-to-predict events (e.g. high-speed solar wind).
"""
import torch


def mean_squared_error_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted mean squared error loss.

    Each squared residual is multiplied by a per-sample weight before
    averaging. Weights are typically derived from a focal weighting
    scheme that down-weights easy samples and up-weights hard ones (e.g.
    high-speed events).

    Args:
        target: Ground-truth values, shape ``(B,)``.
        pred: Model predictions, shape ``(B,)``.
        weights: Per-sample loss weights, shape ``(B,)``.

    Returns:
        Scalar weighted MSE loss tensor.
    """
    squared_errors = torch.pow(pred - target, 2)
    weighted_errors = weights * squared_errors
    return torch.mean(weighted_errors)


def negative_log_likelihood_loss(
    target: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted negative log-likelihood loss under a Normal distribution.

    Targets and parameters are expected in the **normalised log-space**
    produced by ``DistributionTransformer``. Fitting a Normal in this
    space is equivalent to fitting a log-normal in the original km/s
    space, which is the intended predictive distribution.

    Each per-sample log-likelihood is multiplied by a focal weight
    before averaging, so the loss emphasises samples that are harder to
    predict (e.g. high-speed events).

    Args:
        target: Ground-truth values in normalised log-space, shape
            ``(B,)``.
        mu: Predicted mean in normalised log-space, shape ``(B,)``.
        sigma: Predicted standard deviation (must be positive), shape
            ``(B,)``.
        weights: Per-sample loss weights, shape ``(B,)``.

    Returns:
        Scalar weighted negative log-likelihood loss tensor.
    """
    normal_dist = torch.distributions.Normal(mu, sigma)
    log_likelihood = normal_dist.log_prob(target)
    weighted_ll = weights * log_likelihood
    return -torch.mean(weighted_ll)
