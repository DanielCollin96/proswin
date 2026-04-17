import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from proswin.predictions import (
    PointPrediction,
    ProbabilisticPrediction,
    SolarWindPrediction,
)
from proswin.model.solar_swin_transformer import SolarSwinTransformer
from proswin.data_management.distribution_transformer import DistributionTransformer


class SolarWindPredictor:
    """Runs inference with a trained SolarSwinTransformer and saves results.

    Accepts either a plain model or a DDP-wrapped model; DDP mode is detected
    automatically. Mixed-precision inference (``autocast``) is enabled
    automatically on CUDA.

    Args:
        model: Trained model. DDP wrapping is detected and unwrapped
            automatically.
        device: Device to run inference on. Must match the device the model
            is on.
        logger: Optional logger for status messages.
    """

    def __init__(
        self,
        model: SolarSwinTransformer | DistributedDataParallel,
        device: torch.device | str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.logger = logger

        # Detect DDP automatically — no caller-side flag needed
        self._use_ddp: bool = isinstance(model, DistributedDataParallel)
        self._world_size: int = dist.get_world_size() if self._use_ddp else 1
        self._rank = torch.distributed.get_rank() if self._use_ddp else 0

        # Unwrap DDP to access model attributes directly
        self._core_model: SolarSwinTransformer = (
            model.module if self._use_ddp else model  # type: ignore[assignment]
        )
        self._core_model.to(self.device)

        self._point_prediction: bool = self._core_model.point_prediction
        self._distribution_transformer: DistributionTransformer = (
            self._core_model.distribution_transformer
        )

        # autocast accelerates inference on CUDA with negligible precision loss
        self._use_autocast: bool = self.device.type == "cuda"

    def predict(
        self,
        dataloader: DataLoader,
        output_path: str | Path | None = None,
    ) -> tuple[SolarWindPrediction, pd.Series | None]:
        """Run inference over a dataloader and optionally save results to CSV.

        Args:
            dataloader: Yields batches with keys ``"images"``,
                ``"physical_features"``, and optionally ``"target"`` and
                ``"target_date"``.
            output_path: Destination CSV path. Predictions and targets are
                written to a single file. Parent directories are created
                automatically.

        Returns:
            A 2-tuple ``(pred, target_series)`` where ``pred`` is a
            datetime-indexed ``SolarWindPrediction`` in original target units
            and ``target_series`` is a datetime-indexed ground-truth
            ``pd.Series`` in original target units, or ``None`` if
            ``"target"`` is absent from the dataloader.

        Raises:
            RuntimeError: If the model returns an unexpected output type.
        """
        self._core_model.eval()

        raw = self._collect_batches(dataloader)
        tensors = self._gather_ddp(raw)
        pred_df, target_series = self._build_dataframes(tensors)
        pred, target_series = self._postprocess(pred_df, target_series)

        if output_path is not None:
            self._save_csv(pred, target_series, output_path)

        return pred, target_series

    def _collect_batches(
        self,
        dataloader: DataLoader,
    ) -> dict[str, list[torch.Tensor]]:
        """Run forward passes and accumulate raw output tensors.

        Args:
            dataloader: Source of input batches.

        Returns:
            Lists of per-batch tensors keyed by ``"mu"``, and optionally
            ``"sigma"``, ``"targets"``, ``"target_dates"``.

        Raises:
            RuntimeError: If the model returns an unexpected output type.
        """
        raw = {
            "mu": [], "sigma": [], "targets": [], "target_dates": []
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
                if "target" in batch and "target_date" in batch:
                    raw["targets"].append(batch["target"].float().to(self.device))
                    raw["target_dates"].append(batch["target_date"].to(self.device))

                batch_on_device = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k in ("images", "physical_features")
                }

                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self._use_autocast
                ):
                    output = self._core_model(batch_on_device)

                if isinstance(output, tuple):
                    mu, sigma = output
                    raw["sigma"].append(sigma.reshape(-1))
                elif isinstance(output, torch.Tensor):
                    mu = output
                else:
                    raise RuntimeError(
                        f"Unexpected model output type: {type(output)}. "
                        "Expected torch.Tensor or a 2-tuple of tensors."
                    )

                raw["mu"].append(mu.reshape(-1))

        return raw

    def _gather_ddp(
        self,
        raw: dict[str, list[torch.Tensor]],
    ) -> dict[str, torch.Tensor | None]:
        """Concatenate batch lists and, if DDP, gather across all processes.

        Args:
            raw: Per-batch tensor lists as returned by
                :meth:`_collect_batches`.

        Returns:
            Concatenated (and gathered) tensors keyed by ``"mu"``,
            ``"sigma"``, ``"targets"``, ``"target_dates"``. Values are
            ``None`` when the corresponding list was empty (e.g. no targets
            in the dataloader).
        """
        def _cat_or_none(lst: list[torch.Tensor]) -> torch.Tensor | None:
            return torch.cat(lst) if lst else None

        tensors = {k: _cat_or_none(v) for k, v in raw.items()}

        if self._use_ddp:
            tensors = {
                k: _all_gather_tensor(v, self._world_size) if v is not None else None
                for k, v in tensors.items()
            }

        return tensors

    def _build_dataframes(
        self,
        tensors: dict[str, torch.Tensor | None],
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Convert gathered tensors to a datetime-indexed DataFrame and Series.

        Duplicate rows introduced by DDP padding are removed, and results
        are sorted chronologically.

        Args:
            tensors: Gathered tensors keyed by ``"mu"``, ``"sigma"``,
                ``"targets"``, ``"target_dates"``.

        Returns:
            A 2-tuple ``(pred_df, target_series)`` where ``pred_df`` is a
            DataFrame with columns ``["mu"]`` or ``["mu", "sigma"]``, and
            ``target_series`` is a ground-truth Series or ``None``.
        """
        mu = tensors["mu"].cpu().numpy().reshape(-1)  # type: ignore[union-attr]

        if tensors["target_dates"] is not None:
            index = pd.to_datetime(
                tensors["target_dates"].cpu().numpy().reshape(-1), unit="s"
            )
        else:
            index = np.arange(len(mu))

        if tensors["sigma"] is not None:
            sigma = tensors["sigma"].cpu().numpy().reshape(-1)
            pred_df = pd.DataFrame(
                np.stack([mu, sigma], axis=1), index=index, columns=["mu", "sigma"]
            )
        else:
            pred_df = pd.DataFrame(mu, index=index, columns=["mu"])

        target_series: pd.Series | None = None
        if tensors["targets"] is not None:
            target_series = pd.Series(
                tensors["targets"].cpu().numpy().reshape(-1), index=index
            )

        # Remove DistributedSampler padding duplicates, then sort
        if self._use_ddp:
            mask = pred_df.index.duplicated(keep="first")
            pred_df = pred_df.loc[~mask]
            if target_series is not None:
                target_series = target_series.loc[~mask]

        return pred_df.sort_index(), (
            target_series.sort_index() if target_series is not None else None
        )

    def _postprocess(
        self,
        pred_df: pd.DataFrame,
        target_series: pd.Series | None,
    ) -> tuple[PointPrediction | ProbabilisticPrediction, pd.Series | None]:
        """Inverse-transform predictions and targets to original target units.

        Args:
            pred_df: Normalised predictions with columns ``["mu"]`` or
                ``["mu", "sigma"]``.
            target_series: Normalised ground-truth values, or ``None``.

        Returns:
            A 2-tuple ``(pred, target_series)`` in original target units.
        """
        if self._distribution_transformer is None:
            raise ValueError('Distribution Transformer not initialized.')

        if self._point_prediction:
            pred_df["mu"] = self._distribution_transformer.inverse_transform_target(
                pred_df["mu"].values.reshape(-1).astype(float)
            )
            pred = PointPrediction(pred_df["mu"])

        else:
            # Handles mu and sigma jointly for the log-normal parameterisation
            pred_df = self._distribution_transformer.inverse_transform_prediction(pred_df)
            pred = ProbabilisticPrediction(pred_df)

        if target_series is not None:
            target_series = pd.Series(
                self._distribution_transformer.inverse_transform_target(
                    target_series.values
                ),
                index=target_series.index,
            )

        return pred, target_series

    def _save_csv(
        self,
        pred: SolarWindPrediction,
        target_series: pd.Series | None,
        output_path: str | Path,
    ) -> None:
        """Write predictions (and targets if available) to a CSV file.

        Args:
            pred: Predictions to save.
            target_series: If provided, appended as an ``"observation"``
                column.
            output_path: Destination path. Parent directories are created if
                absent.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        obs = target_series.values if target_series is not None else None

        if self._point_prediction:
            output_df = pd.DataFrame({
                'date': pred.index,
                'predicted_mean': pred.mean,
                'observation': obs,
            })
        else:
            output_df = pd.DataFrame({
                'date': pred.index,
                'predicted_0.025_quantile': pred.get_quantiles(0.025).values,
                'predicted_mean': pred.get_expected_values().values,
                'predicted_0.975_quantile': pred.get_quantiles(0.975).values,
                'predicted_lognorm_shape': pred.shape.values,
                'predicted_lognorm_scale': pred.scale.values,
                'predicted_lognorm_loc': pred.loc.values,
                'observation': obs,
            })

        if self._rank == 0:
            output_df.to_csv(output_path, index=False)
            if self.logger is not None:
                self.logger.info(f"Predictions saved to {output_path}  ({len(output_df)} samples)")


def _all_gather_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Gather a 1-D tensor from all DDP processes and concatenate.

    Args:
        tensor: Local tensor on the current process.
        world_size: Total number of processes in the DDP group.

    Returns:
        Concatenated tensor containing contributions from all processes.
    """
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered)
