import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from proswin.data_management.distribution_transformer import DistributionTransformer

logger = logging.getLogger(__name__)

# Supported Swin encoder variants and their timm model strings
_ENCODER_MAP: dict[str, str] = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
}


class SolarSwinTransformer(nn.Module):
    """Probabilistic solar wind speed predictor built on a Swin Transformer.

    Fuses image-based features extracted by a pretrained Swin Transformer with
    hand-crafted physical features. Outputs either a point prediction (mu only)
    or the parameters of a log-normal predictive distribution (mu, sigma) in a
    normalised target space.

    The sigma output is optionally clamped so that the implied distribution
    stays below a physical upper limit at a specified quantile. A bias
    correction is then applied to mu so that the expected value of the
    back-transformed distribution is preserved.

    Args:
        image_encoder_name: Swin variant to use as image backbone. One of
            ``"swin_tiny"``, ``"swin_small"``, ``"swin_base"``.
        use_pretrained_image_encoder_weights: If True, load ImageNet-pretrained
            weights for the Swin backbone on first instantiation (requires a
            network download). Default is True.
        num_physical_features: Number of scalar physical features concatenated
            with image features.
        mu_head_hidden_layer_size: Hidden dimension of the mu prediction MLP.
        sigma_head_hidden_layer_size: Hidden dimension of the sigma prediction
            MLP.
        mu_head_dropout: Dropout probability in the mu head.
        sigma_head_dropout: Dropout probability in the sigma head.
        clamp_sigma: Whether to clamp sigma to respect ``physical_upper_limit``.
        quantile_to_clamp: Quantile (0–1) used to derive the sigma upper bound.
        physical_upper_limit: Hard physical ceiling on the predicted quantity
            (in original units, km/s).
        point_prediction: If True, only mu is predicted and the sigma head is
            omitted.
        use_multiple_gpus: Use ``SyncBatchNorm`` instead of ``BatchNorm2d`` for
            correct statistics across GPUs in DDP training.
        distribution_transformer: Scaler/transformer used to normalise the
            target variable. Required unless ``point_prediction=True`` or
            ``clamp_sigma=False``.
        physical_feature_scaler: Scaler/transformer used to standardize the
            physical input features. It is not used directly in this class,
            but needs to be saved with the model hyperparameters for
            predicting with a saved model.

    Raises:
        ValueError: If ``image_encoder_name`` is not one of the supported
            variants.
        ValueError: If ``distribution_transformer`` is None while
            ``clamp_sigma=True`` and ``point_prediction=False``.
    """

    def __init__(
        self,
        image_encoder_name: str = "swin_tiny",
        use_pretrained_image_encoder_weights: bool = True,
        num_physical_features: int = 63,
        mu_head_hidden_layer_size: int = 168,
        sigma_head_hidden_layer_size: int = 143,
        mu_head_dropout: float = 0.666,
        sigma_head_dropout: float = 0.704,
        clamp_sigma: bool = True,
        quantile_to_clamp: float = 0.99,
        physical_upper_limit: float = 850,
        point_prediction: bool = False,
        use_multiple_gpus: bool = False,
        distribution_transformer: DistributionTransformer | None = None,
        physical_feature_scaler: StandardScaler | None = None,
    ) -> None:
        super().__init__()

        if image_encoder_name not in _ENCODER_MAP:
            raise ValueError(
                f"Unsupported image_encoder '{image_encoder_name}'. "
                f"Choose one of: {list(_ENCODER_MAP.keys())}."
            )

        if not point_prediction and clamp_sigma and distribution_transformer is None:
            raise ValueError(
                "A DistributionTransformer is required when "
                "clamp_sigma=True and point_prediction=False."
            )

        self.image_encoder_name = image_encoder_name
        self.num_physical_features = num_physical_features
        self.mu_head_hidden_layer_size = mu_head_hidden_layer_size
        self.sigma_head_hidden_layer_size = sigma_head_hidden_layer_size
        self.mu_head_dropout = mu_head_dropout
        self.sigma_head_dropout = sigma_head_dropout
        self.min_sigma: float = 1e-1
        self.point_prediction = point_prediction
        self.distribution_transformer = distribution_transformer
        self.physical_feature_scaler = physical_feature_scaler
        self.clamp_sigma = clamp_sigma
        self.quantile_to_clamp = quantile_to_clamp
        self.physical_upper_limit = physical_upper_limit

        # Load pretrained Swin backbone without classification head.
        # NOTE: pretrained=True triggers a network download on first call,
        #       which can take some time.
        #       Pass pretrained=False (or pre-cache weights) to avoid this.
        self.image_encoder = timm.create_model(
            _ENCODER_MAP[self.image_encoder_name],
            pretrained=use_pretrained_image_encoder_weights,
            num_classes=0,   # removes the classification head
            global_pool="avg",
        )

        # Learnable input normalisation (per-channel, spatial BatchNorm).
        # SyncBatchNorm is required for correct statistics across GPUs.
        self.batch_norm: nn.Module = (
            nn.SyncBatchNorm(3) if use_multiple_gpus else nn.BatchNorm2d(3)
        )

        image_feature_dim: int = self.image_encoder.num_features
        combined_feature_dim: int = image_feature_dim + num_physical_features

        # MLP head for the location parameter (mu in normalised log-space)
        self.prediction_head_mu = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(combined_feature_dim, mu_head_hidden_layer_size),
            nn.Dropout(mu_head_dropout),
            nn.LeakyReLU(),
            nn.Linear(mu_head_hidden_layer_size, 1),
        )

        if not self.point_prediction:
            # MLP head for log(sigma); exp() is applied in forward()
            self.prediction_head_sigma = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(combined_feature_dim, sigma_head_hidden_layer_size),
                nn.Dropout(sigma_head_dropout),
                nn.LeakyReLU(),
                nn.Linear(sigma_head_hidden_layer_size, 1),
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            batch: Dict containing:

                - ``"images"``: Float tensor of shape ``(B, 3, 224, 224)``.
                - ``"physical_features"``: Float tensor of shape
                  ``(B, num_physical_features)``.

        Returns:
            Predicted mu of shape ``(B, 1)`` when ``point_prediction=True``,
            or a ``(mu, sigma)`` tuple each of shape ``(B, 1)`` otherwise.
            Both are in normalised log-space.
        """
        images: torch.Tensor = batch["images"]
        physical_features: torch.Tensor = batch["physical_features"]

        # Learnable per-channel normalisation before the transformer
        normalized_images = self.batch_norm(images)

        image_features = self.image_encoder(normalized_images)  # (B, image_feature_dim)

        # Guard against 1-D physical feature tensors (single-sample inference)
        if physical_features.ndim == 1:
            physical_features = physical_features.unsqueeze(0)

        if self.num_physical_features == 0:
            combined_features = image_features
        else:
            combined_features = torch.cat([image_features, physical_features], dim=1)

        output_mu: torch.Tensor = self.prediction_head_mu(combined_features)

        if self.point_prediction:
            return output_mu

        # sigma is predicted in log-space for numerical stability
        output_log_sigma = self.prediction_head_sigma(combined_features)
        output_sigma = torch.exp(output_log_sigma)

        if self.clamp_sigma:
            sigma_bound = self._get_sigma_bound(output_mu)
            output_sigma = torch.minimum(output_sigma, sigma_bound)

        # Applying lower bound after all clamping prevents NLL explosion
        # when sigma_bound itself is very small (large mu batches),
        # so avoiding to let a small output_sigma slip through
        output_sigma = output_sigma.clamp(min=self.min_sigma)

        # Bias correction: ensures E[back-transformed prediction] = exp(mu)
        # For a log-normal, E[X] = exp(mu + sigma^2/2) in the *original* space,
        # so mu must be shifted by -scale * sigma^2 / 2 in normalised space.
        # Only apply bias correction during inference, not during training.
        # During training the NLL is computed in normalised space where targets
        # have no correction applied, so correcting mu here would make the loss
        # try to compensate via sigma, destabilising training.
        if not self.training:
            dist_trafo_scale = torch.tensor(
                self.distribution_transformer.target_scaler.scale_,
                dtype=output_sigma.dtype,
                device=output_sigma.device,
            )
            output_mu = output_mu - dist_trafo_scale * output_sigma.pow(2) / 2

        return output_mu, output_sigma

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def convert_to_probabilistic(
        self,
        sigma_head_hidden_layer_size: int | None = None,
        sigma_head_dropout: float | None = None,
        distribution_transformer: DistributionTransformer | None = None,
    ) -> None:
        """Convert a trained point-prediction model to a probabilistic model in-place.

        Initialises the sigma prediction head while leaving all existing weights
        (encoder + mu head) untouched and fully trainable. Freezing of specific
        parameter groups, if desired, should be done externally after calling
        this method (e.g. by setting ``requires_grad=False`` on selected
        parameters).

        This follows the standard two-stage training protocol:

        - Stage 1: train the deterministic backbone to convergence (point
          prediction mode).
        - Stage 2: extend the model with a distributional head and train
          further, optionally fine-tuning the full network.

        Args:
            sigma_head_hidden_layer_size: Hidden dimension for the sigma MLP
                head. If None, the value already stored on the model
                (``self.sigma_head_hidden_layer_size``) is reused.
            sigma_head_dropout: Dropout probability for the sigma MLP head. If
                None, the value already stored on the model
                (``self.sigma_head_dropout``) is reused.
            distribution_transformer: Scaler used to normalise the target
                variable. If provided, overwrites
                ``self.distribution_transformer``. Required if it was not set
                during point-prediction training and ``clamp_sigma`` is True.

        Raises:
            RuntimeError: If the model is already in probabilistic mode
                (``self.point_prediction is False``).
            ValueError: If ``clamp_sigma=True`` but no
                ``DistributionTransformer`` is available after the conversion.
        """
        if not self.point_prediction:
            raise RuntimeError(
                "Model is already in probabilistic mode "
                "(point_prediction=False). Conversion is a one-way operation."
            )

        # --- Update stored hyperparameters if new values were supplied ----------
        if sigma_head_hidden_layer_size is not None:
            self.sigma_head_hidden_layer_size = sigma_head_hidden_layer_size

        if sigma_head_dropout is not None:
            self.sigma_head_dropout = sigma_head_dropout

        if distribution_transformer is not None:
            self.distribution_transformer = distribution_transformer

        if self.clamp_sigma and self.distribution_transformer is None:
            raise ValueError(
                "A DistributionTransformer is required when clamp_sigma=True. "
                "Pass one via the distribution_transformer argument."
            )

        # --- Build and register the sigma head -----------------------------------
        image_feature_dim: int = self.image_encoder.num_features
        combined_feature_dim: int = image_feature_dim + self.num_physical_features

        self.prediction_head_sigma = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(combined_feature_dim, self.sigma_head_hidden_layer_size),
            nn.Dropout(self.sigma_head_dropout),
            nn.LeakyReLU(),
            nn.Linear(self.sigma_head_hidden_layer_size, 1),
        )

        # Move the new head to the same device as the rest of the model.
        # Inferred from the mu head's first parameter — no stored device needed.
        reference_param = next(self.prediction_head_mu.parameters())
        self.prediction_head_sigma.to(reference_param.device)

        # Flip the mode flag last so the forward() pass activates the sigma branch
        self.point_prediction = False

    def _get_sigma_bound(self, mu: torch.Tensor) -> torch.Tensor:
        """Compute the per-sample maximum sigma that keeps the predictive
        distribution below ``physical_upper_limit`` at ``quantile_to_clamp``.

        The bound is derived by inverting the log-normal quantile formula so
        that the ``quantile_to_clamp``-quantile of the back-transformed
        distribution does not exceed ``physical_upper_limit``.

        Args:
            mu: Predicted mu values in normalised space, shape ``(B, 1)``.

        Returns:
            Per-sample sigma upper bound, shape ``(B, 1)``. Floored at
            ``1e-4`` to avoid degenerate distributions.
        """
        dtype = mu.dtype
        device = mu.device

        # Retrieve normalisation parameters as tensors
        dist_trafo_scale = torch.tensor(
            self.distribution_transformer.target_scaler.scale_,
            dtype=dtype, device=device,
        )
        dist_trafo_mean = torch.tensor(
            self.distribution_transformer.target_scaler.mean_,
            dtype=dtype, device=device,
        )
        dist_trafo_loc_shift = torch.tensor(
            self.distribution_transformer.loc_shift,
            dtype=dtype, device=device,
        )

        # Standard-normal quantile for the chosen probability level
        z_q = torch.tensor(norm.ppf(self.quantile_to_clamp), dtype=dtype, device=device)

        # De-normalise mu to recover the underlying (log) scale parameter
        mu_underlying_norm = mu * dist_trafo_scale + dist_trafo_mean

        # Back-transform: scale=exp(mu) is the log-normal scale parameter
        scale = torch.exp(mu_underlying_norm)
        loc = torch.ones_like(scale) * dist_trafo_loc_shift

        # Maximum log-normal shape s.t. quantile <= physical_upper_limit:
        #   upper_limit = loc + scale * exp(shape * z_q)
        #   => shape = log((upper_limit - loc) / scale) / z_q
        max_shape = torch.log((self.physical_upper_limit - loc) / scale) / z_q

        # Convert shape (in original space) back to normalised sigma units
        sigma_bound = torch.maximum(
            max_shape / dist_trafo_scale,
            torch.tensor(5e-2, dtype=dtype, device=device),  # floor for stability
        )

        return sigma_bound


def save_model(
    model: SolarSwinTransformer,
    path: str | Path,
) -> None:
    """Save a trained model's weights and hyperparameters to a single file.

    The checkpoint contains the model hyperparameters, the full ``state_dict``,
    and the fitted ``DistributionTransformer``, so it can be fully reconstructed
    by :func:`load_model` without any additional arguments.

    Args:
        model: The trained model instance.
        path: Destination path for the checkpoint file (e.g.
            ``"results/run1/model.pt"``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hparams = {
        "image_encoder_name": model.image_encoder_name,
        "num_physical_features": model.num_physical_features,
        "mu_head_hidden_layer_size": model.mu_head_hidden_layer_size,
        "sigma_head_hidden_layer_size": model.sigma_head_hidden_layer_size,
        "mu_head_dropout": model.mu_head_dropout,
        "sigma_head_dropout": model.sigma_head_dropout,
        "clamp_sigma": model.clamp_sigma,
        "quantile_to_clamp": model.quantile_to_clamp,
        "physical_upper_limit": model.physical_upper_limit,
        "point_prediction": model.point_prediction,
    }

    checkpoint = {
        "hyperparameters": hparams,
        "state_dict": model.state_dict(),
        "distribution_transformer": model.distribution_transformer,
        "physical_feature_scaler": model.physical_feature_scaler,
    }

    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to %s  (%.1f MB)", path, path.stat().st_size / 1e6)


def load_model(path: str | Path) -> SolarSwinTransformer:
    """Load a fully trained SolarSwinTransformer from a checkpoint file.

    Reconstructs the model architecture from the saved hyperparameters,
    loads the trained weights, and sets the model to evaluation mode.

    Args:
        path: Path to the checkpoint file produced by :func:`save_model`.

    Returns:
        The reconstructed model in evaluation mode with trained weights loaded.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the checkpoint is missing expected keys (e.g. wrong file
            passed).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at '{path}'.")

    # weights_only=False is required because DistributionTransformer is pickled
    checkpoint = torch.load(path, weights_only=False)

    for key in ("hyperparameters", "state_dict", "distribution_transformer"):
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint at '{path}' is missing key '{key}'."
            )

    hparams: dict = checkpoint["hyperparameters"]
    distribution_transformer: DistributionTransformer = (
        checkpoint["distribution_transformer"]
    )
    physical_feature_scaler= checkpoint["physical_feature_scaler"]

    # Rebuild the architecture — pretrained_backbone=False skips the internet
    # download since weights come entirely from the saved state dict
    model = SolarSwinTransformer(
        use_pretrained_image_encoder_weights=False,
        distribution_transformer=distribution_transformer,
        physical_feature_scaler=physical_feature_scaler,
        **hparams,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # disables dropout and sets BatchNorm to inference mode

    return model

def load_model_without_scaler(path: str | Path) -> SolarSwinTransformer:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at '{path}'.")

    # weights_only=False is required because DistributionTransformer is pickled
    checkpoint = torch.load(path, weights_only=False)

    for key in ("hyperparameters", "state_dict", "distribution_transformer"):
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint at '{path}' is missing key '{key}'."
            )

    hparams: dict = checkpoint["hyperparameters"]
    distribution_transformer: DistributionTransformer = (
        checkpoint["distribution_transformer"]
    )
    #physical_feature_scaler= checkpoint["physical_feature_scaler"]

    # Rebuild the architecture — pretrained_backbone=False skips the internet
    # download since weights come entirely from the saved state dict
    model = SolarSwinTransformer(
        use_pretrained_image_encoder_weights=False,
        distribution_transformer=distribution_transformer,
        physical_feature_scaler=None,
        **hparams,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # disables dropout and sets BatchNorm to inference mode

    return model
