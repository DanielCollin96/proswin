import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from proswin.data_management.distribution_transformer import DistributionTransformer
from proswin.predictions import (
    PointPrediction,
    ProbabilisticPrediction,
)
from proswin.evaluation.solar_wind_prediction_evaluator import PredictionEvaluator
from proswin.model.solar_swin_transformer import (
    SolarSwinTransformer,
    load_model,
    save_model,
)
from proswin.model.solar_wind_predictor import SolarWindPredictor
from proswin.model.losses import (
    mean_squared_error_loss,
    negative_log_likelihood_loss,
)

_VALID_LOSS_FUNCTIONS: dict[str, set[str]] = {
    'mu': {'mse'},
    'sigma': {'nll'},
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_optimizer(
        pred_params: list[torch.nn.Parameter],
        optimizer: str | None = None,
        optimizer_params: dict | None = None,
) -> torch.optim.Optimizer:
    """Instantiate a ``torch.optim`` optimizer by name.

    Args:
        pred_params: List of model parameters to optimise (typically the
            result of filtering ``model.parameters()`` by
            ``requires_grad``).
        optimizer: Class name from ``torch.optim`` (e.g. ``"Adam"``,
            ``"AdamW"``). Defaults to ``"Adam"`` if None.
        optimizer_params: Keyword arguments forwarded to the optimizer
            constructor. The learning rate must be included here by the
            caller.

    Returns:
        Initialised optimizer instance.

    Raises:
        ValueError: If ``optimizer`` is not a valid class in
            ``torch.optim``.
    """
    optim_type = optimizer or "Adam"
    optim_cls = getattr(torch.optim, optim_type, None)
    if optim_cls is None:
        raise ValueError(
            f"Unknown optimizer '{optim_type}'. "
            f"Must be a class name from torch.optim (e.g. 'Adam', 'AdamW')."
        )
    return optim_cls(pred_params, **(optimizer_params or {}))


def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Instantiate a ``torch.optim.lr_scheduler`` by name, or return None.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        lr_scheduler: Class name from ``torch.optim.lr_scheduler``
            (e.g. ``"ReduceLROnPlateau"``). Defaults to
            ``"ReduceLROnPlateau"`` if None. Pass ``"none"``
            (case-insensitive) to disable scheduling entirely.
        lr_scheduler_params: Keyword arguments forwarded to the
            scheduler constructor.

    Returns:
        Initialised scheduler, or ``None`` if scheduling is disabled.

    Raises:
        ValueError: If ``lr_scheduler`` is not a valid class in
            ``torch.optim.lr_scheduler``.
    """
    if lr_scheduler is not None and lr_scheduler.lower() == "none":
        return None
    sched_type = lr_scheduler or "ReduceLROnPlateau"
    sched_cls = getattr(torch.optim.lr_scheduler, sched_type, None)
    if sched_cls is None:
        raise ValueError(
            f"Unknown scheduler '{sched_type}'. "
            f"Must be a class name from torch.optim.lr_scheduler, "
            f"or 'none' to disable."
        )
    return sched_cls(optimizer, **(lr_scheduler_params or {}))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SolarModelTrainer:
    """Trains a SolarSwinTransformer using the two-stage mu → sigma protocol.

    **Stage 1** (``train_mu``): trains the Swin image encoder and the mu MLP
    head jointly using MSE loss on point predictions.

    **Stage 2** (``train_sigma``): reloads the best stage-1 checkpoint, freezes
    the encoder and mu head, then trains only the sigma MLP head using NLL loss
    to fit the log-normal predictive distribution.

    Both stages support early stopping, optional gradient clipping, AMP
    mixed-precision training, multi-GPU DDP, and optional W&B experiment
    tracking.

    Args:
        model: Freshly constructed ``SolarSwinTransformer``. Must be in
            ``point_prediction=True`` mode at construction time.
        device: Device to run training on.
        model_dir: Directory for saving best-checkpoint ``.pth`` files.
        pred_dir: Directory for saving test-set prediction CSV files.
        epoch_losses_dir: Directory for the per-epoch training log CSV.
        use_multiple_gpus: Wrap model in DDP for distributed training.
            Default ``False``.
        gradient_clip_max_norm: Maximum gradient L2 norm for clipping;
            set to ``0`` to disable. Default ``1.0``.
        use_mixed_precision: Enable AMP ``autocast`` + ``GradScaler`` on
            CUDA. Automatically disabled on CPU. Default ``True``.
        wandb_project: W&B project name. If ``None``, W&B logging is
            skipped entirely.
        wandb_run_name: W&B run display name. Defaults to
            ``model_name``.
        wandb_group: W&B group name, useful for grouping CV folds under
            a shared experiment.
        train_config_to_save_in_wandb: Flat config dict logged to W&B at
            initialisation. Use :meth:`build_wandb_config` to construct
            it from the YAML config.
        model_name: Stem used for checkpoint filenames and the training
            log. Defaults to ``"proswin"``.
        logger: Optional external logger. Falls back to the module logger
            if None.
    """

    def __init__(
            self,
            model: SolarSwinTransformer,
            device: torch.device,
            model_dir: Path | str,
            pred_dir: Path | str,
            epoch_losses_dir: Path | str,
            use_multiple_gpus: bool = False,
            gradient_clip_max_norm: float = 1.0,
            use_mixed_precision: bool = True,
            wandb_project: str | None = None,
            wandb_run_name: str | None = None,
            wandb_group: str | None = None,
            train_config_to_save_in_wandb: dict | None = None,
            model_name: str | None = None,
            logger: logging.Logger | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self.device = device
        self._logger.info(f"Training device: {self.device}")

        self.model_name = model_name or "proswin"

        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._pred_dir = Path(pred_dir)
        self._pred_dir.mkdir(parents=True, exist_ok=True)
        self._epoch_losses_dir = Path(epoch_losses_dir)
        self._epoch_losses_dir.mkdir(parents=True, exist_ok=True)

        self._grad_clip: float = float(gradient_clip_max_norm)
        self._use_amp: bool = use_mixed_precision and device.type == "cuda"
        self._gradient_scaler = torch.amp.GradScaler(enabled=self._use_amp)
        self._use_multiple_gpus = use_multiple_gpus
        self._wandb_group = wandb_group

        self._model_hparams = {
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

        # Only rank-0 writes checkpoints, logs metrics, and initialises W&B
        # to avoid duplicate I/O in distributed runs.
        self._is_primary: bool = (
            not use_multiple_gpus or dist.get_rank() == 0
        )

        self._core_model: SolarSwinTransformer = model.to(device)
        self._model = self._wrap_ddp(use_multiple_gpus)

        self._log_parameter_counts()

        # Clear any existing epoch log for this model to avoid stale rows
        log_path = self._epoch_losses_dir / f"{self.model_name}_training_log.csv"
        if log_path.exists():
            log_path.unlink()

        self._wandb_project = wandb_project
        if wandb_run_name is None:
            wandb_run_name = self.model_name
        if self._is_primary:
            self._init_wandb(wandb_run_name, train_config_to_save_in_wandb)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train_mu(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader | None = None,
            learning_rate: float = 1e-3,
            optimizer: str = "Adam",
            optimizer_params: dict | None = None,
            epochs: int = 200,
            early_stopping_epochs: int = 20,
            loss_function: str = "mse",
    ) -> None:
        """Train the image encoder and mu head (stage 1).

        Trains all model parameters jointly using MSE loss until
        validation loss stops improving. The best checkpoint is saved to
        ``<model_dir>/<model_name>_mu_best.pth`` and, if ``test_loader``
        is provided, evaluated immediately after training.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader used for early stopping.
            test_loader: Optional held-out test DataLoader. If provided,
                the best checkpoint is evaluated after training.
            learning_rate: Initial learning rate. Default ``1e-3``.
            optimizer: Optimizer class name (e.g. ``"Adam"``). See
                :func:`_build_optimizer`. Default ``"Adam"``.
            optimizer_params: Additional keyword arguments for the
                optimizer constructor (learning rate is injected
                automatically).
            epochs: Maximum number of training epochs. Default ``200``.
            early_stopping_epochs: Stop after this many consecutive
                epochs without validation improvement. Default ``20``.
            loss_function: Must be ``"mse"``. Default ``"mse"``.

        Raises:
            RuntimeError: If the model is not in
                ``point_prediction=True`` mode.
        """
        if not self._core_model.point_prediction:
            raise RuntimeError(
                "train_mu() requires a point-prediction model "
                "(point_prediction=True). Construct a fresh model with "
                "point_prediction=True."
            )

        self._logger.info("=" * 60)
        self._logger.info("Training stage 1: training image encoder + mu head")
        self._logger.info("=" * 60)

        self._run_training(
            training_stage="mu",
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            optimizer=optimizer,
            epochs=epochs,
            early_stopping_epochs=early_stopping_epochs,
            loss_function=loss_function,
            optimizer_params=optimizer_params,
        )

        if test_loader is not None:
            self._run_test_evaluation(test_loader, "mu")

    def train_sigma(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader | None = None,
            learning_rate: float = 1e-4,
            optimizer: str = "Adam",
            optimizer_params: dict | None = None,
            epochs: int = 200,
            early_stopping_epochs: int = 20,
            loss_function: str = "nll",
            pretrained_mu_checkpoint: str | Path | None = None,
    ) -> None:
        """Train the sigma head (stage 2).

        Reloads the best mu-stage checkpoint, freezes the image encoder
        and mu head, then trains only the sigma MLP head using NLL loss to
        fit the log-normal predictive distribution. The best checkpoint is
        saved to ``<model_dir>/<model_name>_sigma_best.pth``.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader used for early stopping.
            test_loader: Optional held-out test DataLoader evaluated after
                training.
            learning_rate: Initial learning rate. Default ``1e-4``.
            optimizer: Optimizer class name. Default ``"Adam"``.
            optimizer_params: Additional keyword arguments for the
                optimizer.
            epochs: Maximum training epochs. Default ``200``.
            early_stopping_epochs: Early-stopping patience. Default
                ``20``.
            loss_function: Must be ``"nll"``. Default ``"nll"``.
            pretrained_mu_checkpoint: Path to a mu-stage checkpoint. If
                None, defaults to
                ``<model_dir>/<model_name>_mu_best.pth``.

        Raises:
            RuntimeError: If the model is not in
                ``point_prediction=True`` mode before reloading.
            FileNotFoundError: If the mu-stage checkpoint is missing.
        """
        if not self._core_model.point_prediction:
            raise RuntimeError(
                "train_sigma() expects the model to be in point-prediction mode "
                "before reloading — conversion to probabilistic happens internally."
            )

        checkpoint_to_load = Path(
            pretrained_mu_checkpoint
            if pretrained_mu_checkpoint is not None
            else self._model_dir / f"{self.model_name}_mu_best.pth"
        )

        self._logger.info("=" * 60)
        self._logger.info("Training stage 2: training sigma head")
        self._logger.info("=" * 60)

        self._reload_best_checkpoint(checkpoint_to_load)
        self._freeze_mu_and_encoder()
        self._log_parameter_counts()

        self._run_training(
            training_stage="sigma",
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=learning_rate,
            optimizer=optimizer,
            epochs=epochs,
            early_stopping_epochs=early_stopping_epochs,
            loss_function=loss_function,
            optimizer_params=optimizer_params,
        )

        if test_loader is not None:
            self._run_test_evaluation(test_loader, "sigma")

    def finish_logging(self) -> None:
        """Finish the W&B run. Call explicitly after all training stages complete."""
        if self._is_primary and _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

    # ------------------------------------------------------------------
    # Private: training loop
    # ------------------------------------------------------------------

    def _run_training(
            self,
            training_stage: str,
            train_loader: DataLoader,
            val_loader: DataLoader,
            learning_rate: float,
            optimizer: str,
            epochs: int,
            early_stopping_epochs: int,
            loss_function: str,
            optimizer_params: dict | None = None,
            lr_scheduler: str = 'ReduceLROnPlateau',
            lr_scheduler_params: dict | None = None,
    ) -> None:
        """Execute the training loop for one stage.

        Builds the optimizer and LR scheduler, then iterates over epochs.
        Each epoch runs a full training pass and a validation pass, steps
        the scheduler, and saves a checkpoint when the validation loss
        improves. Stops early if no improvement is seen for
        ``early_stopping_epochs`` consecutive epochs.

        Args:
            training_stage: ``"mu"`` or ``"sigma"``.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            learning_rate: Initial learning rate (merged into
                ``optimizer_params`` before being passed to the
                optimizer).
            optimizer: Optimizer class name (e.g. ``"Adam"``).
            epochs: Maximum number of epochs.
            early_stopping_epochs: Early-stopping patience.
            loss_function: ``"mse"`` for stage 1; ``"nll"`` for
                stage 2.
            optimizer_params: Additional keyword arguments for the
                optimizer constructor.
            lr_scheduler: LR scheduler class name. Default
                ``"ReduceLROnPlateau"``.
            lr_scheduler_params: Keyword arguments for the scheduler.

        Raises:
            ValueError: If ``training_stage`` or ``loss_function`` is
                invalid.
        """
        if training_stage not in ('mu', 'sigma'):
            raise ValueError("Training stage must be 'mu' or 'sigma'.")

        valid = _VALID_LOSS_FUNCTIONS.get(training_stage, set())
        if loss_function not in valid:
            raise ValueError(
                f"Loss function '{loss_function}' is not valid for training "
                f"stage '{training_stage}'. Expected one of: {valid}."
            )

        # Inject learning rate into the optimizer kwargs
        optimizer_params = (optimizer_params or {}) | {"lr": learning_rate}
        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        opt = _build_optimizer(trainable_params, optimizer, optimizer_params)
        scheduler = _build_scheduler(opt, lr_scheduler, lr_scheduler_params)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        # Reset the AMP scaler at the start of each training stage so state
        # from a previous stage is not carried over.
        self._gradient_scaler = torch.amp.GradScaler(enabled=self._use_amp)

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self._train_epoch(opt, train_loader, loss_function)
            val_loss, val_rmse, val_crps = self._validate_epoch(
                val_loader, loss_function
            )

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()

            epoch_duration = time.perf_counter() - epoch_start

            if self._is_primary:
                self._log_epoch(
                    epoch, train_loss, val_loss, val_rmse, val_crps,
                    opt, training_stage, epoch_duration,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if self._is_primary:
                    self._save_checkpoint(training_stage)
                    self._logger.info(
                        f"  => Best checkpoint saved (val_loss={best_val_loss:.6f})"
                    )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_epochs:
                if self._is_primary:
                    self._logger.info(
                        f"Early stopping at epoch {epoch} — no improvement "
                        f"for {early_stopping_epochs} epochs."
                    )
                break

    # ------------------------------------------------------------------
    # Private: forward + loss
    # ------------------------------------------------------------------

    def _train_epoch(
            self,
            optimizer: torch.optim.Optimizer,
            train_loader: DataLoader,
            loss_function: str,
    ) -> float:
        """Run a single training epoch over all batches.

        Performs the forward pass, loss computation, and backward pass
        with AMP scaling and gradient clipping for each batch.

        Args:
            optimizer: Active optimizer.
            train_loader: DataLoader for the training split.
            loss_function: ``"mse"`` or ``"nll"``.

        Returns:
            Mean batch loss over the full epoch.
        """
        self._model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="  Training", unit="batch", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            targets = batch["target"].float()
            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self._use_amp
            ):
                output = self._model(batch)
                loss = self._compute_loss(
                    output, targets, loss_function,
                    batch["loss_function_weights"],
                )

            self._gradient_scaler.scale(loss).backward()
            if self._grad_clip > 0.0:
                self._gradient_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self._model.parameters(), self._grad_clip
                )
            self._gradient_scaler.step(optimizer)
            self._gradient_scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / len(train_loader)

    def _validate_epoch(
            self,
            val_loader: DataLoader,
            loss_function: str,
    ) -> tuple[float, float, float]:
        """Run a single validation epoch and compute evaluation metrics.

        Accumulates loss and predictions over all batches, gathers across
        DDP ranks, inverse-transforms to original units, and computes
        RMSE and CRPS.

        Args:
            val_loader: DataLoader for the validation split.
            loss_function: ``"mse"`` or ``"nll"``.

        Returns:
            A 3-tuple ``(val_loss, val_rmse, val_crps)`` where
            ``val_loss`` is the mean batch loss, ``val_rmse`` is RMSE in
            original target units, and ``val_crps`` is mean CRPS in
            original target units (NaN for point predictions).
        """
        self._model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = torch.tensor(0, device=self.device)
        all_mu: list[torch.Tensor] = []
        all_sigma: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        # Dataset length before DDP padding — used to trim gathered tensors.
        n_val_samples = len(val_loader.dataset)

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc="  Validating", unit="batch", leave=False
            ):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                targets = batch["target"].float()

                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self._use_amp
                ):
                    output = self._model(batch)
                    loss = self._compute_loss(
                        output, targets, loss_function,
                        batch["loss_function_weights"],
                    )

                total_loss += loss
                num_batches += 1
                all_targets.append(targets.detach().reshape(-1))

                if isinstance(output, tuple):
                    mu, sigma = output
                    all_mu.append(mu.detach().reshape(-1))
                    all_sigma.append(sigma.detach().reshape(-1))
                else:
                    all_mu.append(output.detach().reshape(-1))

        if self._use_multiple_gpus:
            # Synchronise the scalar loss across all ranks so every rank
            # reports the same mean validation loss.
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

        val_loss = (total_loss / num_batches).item()

        # Gather tensors across DDP ranks and trim any padding added by
        # DistributedSampler to make the dataset evenly divisible.
        mu_cat = self._gather_tensor_if_ddp(torch.cat(all_mu))[:n_val_samples]
        targets_cat = (
            self._gather_tensor_if_ddp(torch.cat(all_targets))[:n_val_samples]
        )
        sigma_cat = (
            self._gather_tensor_if_ddp(torch.cat(all_sigma))[:n_val_samples]
            if all_sigma else None
        )

        mu_orig, targets_orig, pred_df_orig = self._inverse_transform(
            mu_cat, targets_cat, sigma_cat
        )

        targets_series = pd.Series(targets_orig)
        pred = (
            ProbabilisticPrediction(pred_df_orig)
            if pred_df_orig is not None
            else PointPrediction(pd.Series(mu_orig))
        )

        rmse = float(np.sqrt(np.mean(
            (pred.get_expected_values().values - targets_series.values) ** 2
        )))
        crps = (
            float(pred.get_crps(targets_series).mean())
            if pred_df_orig is not None
            else float("nan")
        )

        return val_loss, rmse, crps

    def _compute_loss(
            self,
            output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
            targets: torch.Tensor,
            loss_function: str,
            weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the weighted batch loss for a single forward pass.

        Routes to the MSE or NLL loss function based on ``loss_function``
        and validates that the model output type matches.

        Args:
            output: Model output — a single tensor ``(mu,)`` for MSE, or
                a ``(mu, sigma)`` tuple for NLL.
            targets: Ground-truth target tensor of shape ``(B,)``.
            loss_function: ``"mse"`` or ``"nll"``.
            weights: Per-sample loss weights of shape ``(B,)``.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If the output type does not match the expected
                loss function.
            NotImplementedError: If ``loss_function`` is not recognised.
        """
        if loss_function == "mse":
            if isinstance(output, tuple):
                raise ValueError(
                    "MSE loss expects point-prediction output (mu only), "
                    "got (mu, sigma)."
                )
            return mean_squared_error_loss(targets, output, weights=weights)
        elif loss_function == "nll":
            if not isinstance(output, tuple):
                raise ValueError(
                    "NLL loss expects probabilistic output (mu, sigma), "
                    "got a single tensor."
                )
            mu, sigma = output
            return negative_log_likelihood_loss(targets, mu, sigma, weights=weights)
        else:
            raise NotImplementedError(
                f"Loss function '{loss_function}' is not implemented."
            )

    def _gather_tensor_if_ddp(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather a tensor across DDP ranks, or return it unchanged.

        Args:
            tensor: Local tensor on the current rank.

        Returns:
            Concatenated tensor from all ranks when DDP is active; the
            input tensor unchanged in single-GPU mode.
        """
        if not self._use_multiple_gpus:
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered)

    # ------------------------------------------------------------------
    # Private: parameter freezing and reloading
    # ------------------------------------------------------------------

    def _freeze_mu_and_encoder(self) -> None:
        """Freeze the image encoder, batch norm, and mu head.

        Sets ``requires_grad = False`` on all parameters of the encoder,
        batch normalisation layer, and mu MLP head so they are excluded
        from optimiser updates during sigma-head training (stage 2).
        """
        components_to_freeze = [
            self._core_model.image_encoder,
            self._core_model.batch_norm,
            self._core_model.prediction_head_mu,
        ]
        for component in components_to_freeze:
            for param in component.parameters():
                param.requires_grad = False

    def _check_architecture_compatibility(
            self, loaded_model: SolarSwinTransformer
    ) -> None:
        """Raise ValueError if the checkpoint architecture differs from the current model.

        Args:
            loaded_model: Model loaded from checkpoint.

        Raises:
            ValueError: If any key architectural hyperparameter does not
                match.
        """
        keys_to_check = [
            "image_encoder_name",
            "num_physical_features",
            "mu_head_hidden_layer_size",
            "mu_head_dropout",
        ]
        mismatches = [
            f"  {key}: checkpoint={getattr(loaded_model, key)}, "
            f"current={self._model_hparams[key]}"
            for key in keys_to_check
            if getattr(loaded_model, key) != self._model_hparams[key]
        ]
        if mismatches:
            raise ValueError(
                "Architecture mismatch between checkpoint and current model:\n"
                + "\n".join(mismatches)
            )

    def _reload_best_checkpoint(self, checkpoint_path: Path) -> None:
        """Reload a stage-1 checkpoint and convert the model to probabilistic mode.

        Loads the saved mu-stage weights, verifies architecture
        compatibility, copies the weights into the current model, calls
        :meth:`~SolarSwinTransformer.convert_to_probabilistic` to attach
        the sigma head, and re-wraps in DDP with
        ``find_unused_parameters=True`` (required because the encoder and
        mu head will be frozen).

        Args:
            checkpoint_path: Path to the ``.pth`` file produced by
                :meth:`_save_checkpoint` during stage 1.

        Raises:
            FileNotFoundError: If ``checkpoint_path`` does not exist.
            ValueError: If the checkpoint is already probabilistic, or if
                the architecture does not match the current model.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: '{checkpoint_path}'. "
                "Run train_mu() first, or pass a valid pretrained_mu_checkpoint."
            )
        self._logger.info(f"Reloading weights from {checkpoint_path} ...")

        if self._use_multiple_gpus:
            dist.barrier()  # ensure all ranks wait for primary to finish saving

        loaded_model = load_model(checkpoint_path)

        if not loaded_model.point_prediction:
            raise ValueError(
                f"Checkpoint at '{checkpoint_path}' is already a probabilistic "
                "model. train_sigma() expects a mu-stage "
                "(point_prediction=True) checkpoint."
            )

        self._check_architecture_compatibility(loaded_model)

        self._core_model.load_state_dict(loaded_model.state_dict())
        self._core_model.convert_to_probabilistic()

        for param in self._core_model.parameters():
            param.requires_grad = True

        # find_unused_parameters=True is required here because the encoder and
        # mu head will be frozen after this call, so DDP would otherwise
        # complain about parameters that never receive gradients.
        self._model = self._wrap_ddp(
            use_multiple_gpus=self._use_multiple_gpus,
            find_unused_parameters=True,
        )

    # ------------------------------------------------------------------
    # Private: checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, training_stage: str) -> None:
        """Save the current model weights as the best checkpoint for a stage.

        Args:
            training_stage: ``"mu"`` or ``"sigma"``; used to construct
                the filename ``<model_name>_<stage>_best.pth``.
        """
        save_model(
            model=self._core_model,
            path=self._model_dir / f"{self.model_name}_{training_stage}_best.pth",
        )

    # ------------------------------------------------------------------
    # Private: test evaluation
    # ------------------------------------------------------------------

    def _inverse_transform(
            self,
            mu: torch.Tensor,
            targets: torch.Tensor,
            sigma: torch.Tensor | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray, pd.DataFrame | None]:
        """Inverse-transform normalised model outputs and targets.

        For point predictions (``sigma=None``), applies the target
        inverse transform to both ``mu`` and ``targets``. For
        probabilistic predictions, converts the ``(mu, sigma)`` pair
        using
        :meth:`~DistributionTransformer.inverse_transform_prediction`
        to produce log-normal parameters (shape, scale, loc).

        Args:
            mu: Normalised mu output tensor, shape ``(N,)``.
            targets: Normalised target tensor, shape ``(N,)``.
            sigma: Normalised sigma tensor of shape ``(N,)``, or
                ``None`` for point predictions.

        Returns:
            A 3-tuple ``(mu_orig, targets_orig, pred_df_orig)`` where:

            - ``mu_orig``: inverse-transformed mu array for point
              predictions, or ``None`` in probabilistic mode (expected
              values are then obtained via
              ``ProbabilisticPrediction.get_expected_values()``).
            - ``targets_orig``: inverse-transformed ground-truth array.
            - ``pred_df_orig``: DataFrame of log-normal parameters
              (shape, scale, loc), or ``None`` for point predictions.
        """
        dt = self._core_model.distribution_transformer
        mu_np = mu.cpu().numpy().reshape(-1).astype(float)
        targets_np = targets.cpu().numpy().reshape(-1).astype(float)
        targets_orig = dt.inverse_transform_target(targets_np)

        if sigma is not None:
            sigma_np = sigma.cpu().numpy().reshape(-1).astype(float)
            pred_df = pd.DataFrame({"mu": mu_np, "sigma": sigma_np})
            pred_df_orig = dt.inverse_transform_prediction(pred_df)
            # Expected values are not needed here: ProbabilisticPrediction
            # computes them via get_expected_values() using lognorm.mean().
            return None, targets_orig, pred_df_orig

        return dt.inverse_transform_target(mu_np), targets_orig, None

    def _run_test_evaluation(
            self, test_loader: DataLoader, training_stage: str
    ) -> None:
        """Load the best checkpoint and evaluate it on the test set.

        Reloads the best saved checkpoint for ``training_stage``, runs
        :class:`~SolarWindPredictor` over the test loader, saves
        predictions to CSV, and logs evaluation metrics. W&B metrics are
        reported under the ``test/`` namespace.

        Args:
            test_loader: DataLoader for the held-out test split.
            training_stage: ``"mu"`` or ``"sigma"``; used to locate the
                checkpoint and name the output CSV.
        """
        self._logger.info("Running test evaluation ...")

        if self._use_multiple_gpus:
            dist.barrier()  # ensure checkpoint is fully written before loading

        best_model = load_model(
            self._model_dir / f"{self.model_name}_{training_stage}_best.pth"
        )
        predictor = SolarWindPredictor(best_model, device=self.device)
        prediction, target_series = predictor.predict(
            test_loader,
            output_path=(
                self._pred_dir / f"{self.model_name}_{training_stage}_test.csv"
            ),
        )

        if target_series is not None:
            evaluator = PredictionEvaluator(prediction, target_series)
            results = evaluator.evaluate()
            if isinstance(results, tuple):
                results = results[0]
            self._logger.info("Test evaluation results:")
            self._logger.info(f"\n{results.to_string()}")

            if _WANDB_AVAILABLE and wandb.run is not None:
                for metric in results.index:
                    wandb.log({f"test/{metric}": results.loc[metric]})

    # ------------------------------------------------------------------
    # Private: logging and DDP helpers
    # ------------------------------------------------------------------

    def _log_parameter_counts(self) -> None:
        """Log total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self._core_model.parameters())
        trainable = sum(
            p.numel() for p in self._core_model.parameters() if p.requires_grad
        )
        frozen = total - trainable
        self._logger.info(
            f"Model parameters — total: {total:,} | "
            f"trainable: {trainable:,} | frozen: {frozen:,}"
        )

    def _log_epoch(
            self,
            epoch: int,
            train_loss: float,
            val_loss: float,
            val_rmse: float,
            val_crps: float,
            optimizer: torch.optim.Optimizer,
            training_stage: str,
            epoch_duration: float,
    ) -> None:
        """Log epoch metrics to the console, CSV, and W&B.

        Appends one row to the per-model training log CSV (created on
        the first call, since ``__init__`` deletes any pre-existing log).
        Also logs to W&B under the ``<stage>/`` namespace if a run is
        active.

        Args:
            epoch: Current epoch number (1-indexed).
            train_loss: Mean training loss for this epoch.
            val_loss: Mean validation loss for this epoch.
            val_rmse: RMSE in original target units on the validation
                set.
            val_crps: Mean CRPS in original target units (NaN for point
                predictions).
            optimizer: Active optimizer; used to read the current
                learning rate from ``param_groups[0]["lr"]``.
            training_stage: ``"mu"`` or ``"sigma"``; used as the W&B
                metric prefix.
            epoch_duration: Wall-clock time for the epoch in seconds.
        """
        lr = optimizer.param_groups[0]["lr"]
        self._logger.info(
            f"Epoch {epoch:>4d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | val_rmse={val_rmse:.4f} | "
            f"val_crps={val_crps:.4f} | lr={lr:.2e} | "
            f"time={epoch_duration:.1f}s"
        )

        row = {
            "epoch": epoch,
            "stage": training_stage,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_crps": val_crps,
            "lr": lr,
            "epoch_duration_s": round(epoch_duration, 1),
        }
        log_path = self._epoch_losses_dir / f"{self.model_name}_training_log.csv"
        pd.DataFrame([row]).to_csv(
            log_path, mode="a", header=not log_path.exists(), index=False,
        )

        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                f"{training_stage}/epoch": epoch,
                f"{training_stage}/train_loss": train_loss,
                f"{training_stage}/val_loss": val_loss,
                f"{training_stage}/val_rmse": val_rmse,
                f"{training_stage}/val_crps": val_crps,
                f"{training_stage}/lr": lr,
                f"{training_stage}/epoch_duration_s": epoch_duration,
            })

    def _wrap_ddp(
            self, use_multiple_gpus: bool, find_unused_parameters: bool = False
    ) -> nn.Module:
        """Wrap the core model in DDP or return it unwrapped.

        Args:
            use_multiple_gpus: If True, wrap ``self._core_model`` in
                ``DistributedDataParallel`` using the current device.
            find_unused_parameters: Passed directly to DDP. Set to True
                when some parameters are frozen (e.g. during sigma-head
                training) to avoid DDP errors on parameters that never
                receive gradients. Default ``False``.

        Returns:
            DDP-wrapped module when ``use_multiple_gpus`` is True,
            otherwise the unwrapped ``self._core_model``.
        """
        if use_multiple_gpus:
            return DDP(
                self._core_model,
                device_ids=[self.device],
                find_unused_parameters=find_unused_parameters,
            )
        return self._core_model

    def _init_wandb(
            self,
            run_name: str,
            train_config_to_save_in_wandb: dict | None = None,
    ) -> None:
        """Initialise a W&B run with model and training configuration.

        No-ops if W&B is not installed or ``wandb_project`` is None.
        Logs model hyperparameters, trainer settings, and any external
        config dict to the run. Calls ``wandb.watch`` to log gradients
        every 100 steps.

        Args:
            run_name: Display name for the W&B run.
            train_config_to_save_in_wandb: Additional key-value pairs to
                include in the W&B config (e.g. from
                :meth:`build_wandb_config`).
        """
        if not _WANDB_AVAILABLE or self._wandb_project is None:
            return

        model_config = {f"model/{k}": v for k, v in self._model_hparams.items()}
        trainer_config = {
            "training/gradient_clip_max_norm": self._grad_clip,
            "training/use_mixed_precision": self._use_amp,
            "training/use_multiple_gpus": self._use_multiple_gpus,
        }
        external_config = train_config_to_save_in_wandb or {}

        wandb.init(
            project=self._wandb_project,
            name=run_name,
            group=self._wandb_group,
            config=model_config | trainer_config | external_config,
        )
        wandb.watch(self._model, log="gradients", log_freq=100)

        self._logger.info(
            f"W&B run initialised: "
            f"project='{wandb.run.project}' | "
            f"group='{wandb.run.group}' | "
            f"name='{wandb.run.name}' | "
            f"url={wandb.run.url}"
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_wandb_config(
            train_config: dict,
            output_scaler: DistributionTransformer,
    ) -> dict:
        """Build a flat W&B config dict from training config and fitted scaler.

        Assembles a single flat dictionary suitable for passing as
        ``train_config_to_save_in_wandb`` to the constructor.

        Args:
            train_config: Training configuration dict loaded from YAML.
            output_scaler: Fitted ``DistributionTransformer`` from
                ``DataLoaderCreator.output_scaler``.

        Returns:
            Flat config dict ready to pass as
            ``train_config_to_save_in_wandb``.
        """
        def _stage(stage: str) -> dict:
            cfg = train_config[stage]
            return {
                f"{stage}/optimizer": cfg["optimizer"],
                f"{stage}/batch_size": cfg["batch_size"],
                f"{stage}/learning_rate": cfg["learning_rate"],
                f"{stage}/epochs": cfg["epochs"],
                f"{stage}/early_stopping_epochs": cfg["early_stopping_epochs"],
                f"{stage}/loss_function": cfg["loss_function"],
                f"{stage}/loss_weights_alpha": cfg["loss_weights_alpha"],
                f"{stage}/loss_weights_gamma": cfg["loss_weights_gamma"],
            }

        return {
            "train_test_scheme": train_config["train_test_scheme"],
            "cross_validation/n_folds": train_config["cross_validation"]["n_folds"],
            "cross_validation/cv_split": train_config["cross_validation"]["split"],
            "cross_validation/gap_days": train_config["cross_validation"]["gap_days"],
            **_stage("mu"),
            **_stage("sigma"),
            "model/distribution_transformer/loc_shift": output_scaler.loc_shift,
            "model/distribution_transformer/target_scaler_mean": float(
                output_scaler.target_scaler.mean_[0]
            ),
            "model/distribution_transformer/target_scaler_scale": float(
                output_scaler.target_scaler.scale_[0]
            ),
        }
