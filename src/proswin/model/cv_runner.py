import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from proswin.data_management.data_loader_creator import DataLoaderCreator
from proswin.data_management.solar_dataset import SolarDataset
from proswin.predictions import (
    PointPrediction,
    ProbabilisticPrediction,
)
from proswin.evaluation.hss_finder import HSSFinder
from proswin.evaluation.solar_wind_prediction_evaluator import PredictionEvaluator
from proswin.model.solar_model_trainer import SolarModelTrainer
from proswin.model.solar_swin_transformer import SolarSwinTransformer
from proswin.utils.configurator import ConfigLoader


class CrossValidationRunner:
    """Runs a full cross-validation training and evaluation pipeline.

    For each fold:

    - Creates a fresh model and ``DataLoaderCreator`` (scalers fitted on
      fold-specific training data only to prevent leakage).
    - Runs mu and/or sigma training via ``SolarModelTrainer``.
    - Saves per-fold predictions and metrics.
    - Logs each fold as a separate W&B run grouped under the model name.

    After all folds:

    - Concatenates per-fold predictions into a single chronological time
      series covering the full evaluation period.
    - Evaluates and saves the aggregated results.
    - Logs aggregated metrics to a dedicated W&B summary run.

    Args:
        dataset: Fully constructed dataset covering the full date range.
        cv_split: Output of ``SequentialCVSplitGenerator.compute_split()``.
            Keys ``"train"``, ``"val"``, ``"test"``, each a list of
            ``pd.DatetimeIndex`` objects, one per fold.
        cv_split_missing: Corresponding missing-date indices for each
            fold and split (same structure as ``cv_split``). Used by
            ``HSSFinder`` to fill gaps before peak detection.
        config: Loaded configuration object.
        device: Device to run training on.
        logger: Logger for progress messages. Falls back to the module
            logger if None.

    Raises:
        ValueError: If ``config.train_config['fit_prediction_head']``
            contains a stage name other than ``"mu"`` or ``"sigma"``.
    """

    def __init__(
            self,
            dataset: SolarDataset,
            cv_split: dict,
            cv_split_missing: dict,
            config: ConfigLoader,
            device: torch.device,
            logger: logging.Logger | None = None,
    ) -> None:
        self._dataset = dataset
        self._cv_split = cv_split
        self._cv_split_missing = cv_split_missing
        self._config = config
        self._device = device
        self._logger = logger or logging.getLogger(__name__)
        self._n_folds = len(cv_split['train'])
        self._model_name = config.model_config['model_name']
        self._fit_stages: list[str] = config.train_config['fit_prediction_head']

        self._logger.info(
            f"W&B project: {self._config.train_config.get('wandb_project')}"
        )

        # Validate requested stages
        for stage in self._fit_stages:
            if stage not in ('mu', 'sigma'):
                raise ValueError(
                    f"fit_prediction_head contains unknown stage '{stage}'. "
                    "Must be 'mu', 'sigma', or both."
                )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> pd.Series:
        """Run the full cross-validation pipeline.

        Iterates over all folds, training and evaluating each one.
        Failed folds are logged and skipped; if all folds fail a
        ``RuntimeError`` is raised. After all successful folds, per-fold
        predictions are concatenated and evaluated as a single time
        series.

        Returns:
            Aggregated evaluation metrics across all folds as a
            ``pd.Series`` indexed by metric name.

        Raises:
            RuntimeError: If every fold fails and there are no
                predictions to aggregate.
        """
        start_time = time.perf_counter()
        self._logger.info(
            f"Starting {self._n_folds}-fold cross-validation for model "
            f"'{self._model_name}'."
        )

        fold_predictions: list[pd.DataFrame] = []
        fold_observations: list[pd.Series] = []
        fold_metrics: list[pd.Series] = []
        failed_folds: list[int] = []

        for fold_idx in range(self._n_folds):
            self._logger.info(
                f"\n{'=' * 60}\nFold {fold_idx + 1} / {self._n_folds}\n{'=' * 60}"
            )
            try:
                pred_df, obs, metrics = self._run_fold(fold_idx)
                fold_predictions.append(pred_df)
                fold_observations.append(obs)
                fold_metrics.append(metrics)
            except Exception as exc:
                self._logger.error(
                    f"Fold {fold_idx + 1} failed with error: {exc}. "
                    "Continuing with remaining folds.",
                    exc_info=True,
                )
                failed_folds.append(fold_idx + 1)

        if not fold_predictions:
            raise RuntimeError(
                "All folds failed. No predictions to aggregate. "
                f"Failed folds: {failed_folds}"
            )

        if failed_folds:
            self._logger.warning(
                f"The following folds failed and are excluded from aggregation: "
                f"{failed_folds}"
            )

        aggregated_metrics = self._aggregate_and_evaluate(
            fold_predictions, fold_observations, fold_metrics
        )

        elapsed = time.perf_counter() - start_time
        self._logger.info(
            f"Cross-validation complete in {elapsed / 60:.1f} min. "
            f"Successful folds: {self._n_folds - len(failed_folds)}/{self._n_folds}."
        )

        return aggregated_metrics

    # ------------------------------------------------------------------
    # Private: single fold
    # ------------------------------------------------------------------

    def _run_fold(
            self, fold_idx: int
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Run training and evaluation for a single fold.

        Args:
            fold_idx: Zero-based fold index.

        Returns:
            A 3-tuple ``(pred_df, obs, metrics)`` where ``pred_df``
            contains raw prediction parameters (shape/scale/loc or mu)
            for this fold's test set with a datetime index, ``obs`` is
            the corresponding ground-truth series, and ``metrics`` is
            the fold-level evaluation metrics.
        """
        fold_name = f"fold_{fold_idx + 1:02d}"
        fold_model_name = f"{self._model_name}_{fold_name}"

        # One flat directory per model — fold name is used as the filename
        # prefix so all folds' outputs coexist without subdirectories.
        model_model_dir = self._config.paths['models'] / self._model_name
        model_pred_dir = self._config.paths['predictions'] / self._model_name
        model_epoch_dir = self._config.paths['epoch_losses'] / self._model_name
        model_eval_dir = self._config.paths['evaluation'] / self._model_name

        for d in (model_model_dir, model_pred_dir, model_epoch_dir, model_eval_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Fresh DataLoaderCreator per fold — scalers are fitted on this fold's
        # training data only, preventing data leakage across folds.
        loader_creator = DataLoaderCreator(
            dataset=self._dataset,
            train_dates=self._cv_split['train'][fold_idx],
            validation_dates=self._cv_split['val'][fold_idx],
            test_dates=self._cv_split['test'][fold_idx],
            use_multiple_gpus=self._config.train_config['use_multiple_gpus'],
        )


        # Fresh model for each fold — no weight sharing across folds
        model = self._build_model(loader_creator)

        wandb_config = SolarModelTrainer.build_wandb_config(
            train_config=self._config.train_config,
            output_scaler=loader_creator.output_scaler,
        )
        # Add fold index to W&B config for easy filtering across folds
        wandb_config['fold'] = fold_idx + 1

        trainer = SolarModelTrainer(
            model=model,
            device=self._device,
            model_dir=model_model_dir,
            pred_dir=model_pred_dir,
            epoch_losses_dir=model_epoch_dir,
            use_multiple_gpus=self._config.train_config['use_multiple_gpus'],
            wandb_project=self._config.train_config.get('wandb_project'),
            # Group name links all folds of this model in the W&B UI
            wandb_run_name=f"{self._model_name}_{fold_name}",
            wandb_group=self._model_name,
            model_name=fold_model_name,
            train_config_to_save_in_wandb=wandb_config,
            logger=self._logger,
        )

        self._train_fold(trainer, loader_creator, fold_idx)
        trainer.finish_logging()

        # Load and evaluate predictions saved by the trainer
        final_stage = self._fit_stages[-1]
        pred_path = model_pred_dir / f"{fold_model_name}_{final_stage}_test.csv"
        pred_df, obs = self._load_predictions(pred_path, final_stage)
        metrics = self._evaluate_fold(
            pred_df, obs, fold_idx, fold_model_name, final_stage, model_eval_dir
        )

        self._logger.info(f"Fold {fold_idx + 1} metrics:\n{metrics.to_string()}")

        return pred_df, obs, metrics

    def _train_fold(
            self,
            trainer: SolarModelTrainer,
            loader_creator: DataLoaderCreator,
            fold_idx: int,
    ) -> None:
        """Run the configured training stages for one fold.

        Executes whichever stages are listed in
        ``config.train_config['fit_prediction_head']`` (``"mu"``,
        ``"sigma"``, or both in that order). For each stage the focal
        loss weights are applied to the shared dataset before the
        DataLoaders are constructed, so each stage uses its own
        weighting scheme.

        Args:
            trainer: Initialised ``SolarModelTrainer`` for this fold.
            loader_creator: Fold-specific ``DataLoaderCreator`` with
                scalers already fitted on this fold's training data.
            fold_idx: Zero-based fold index (currently unused; reserved
                for future per-fold configuration overrides).
        """
        mu_cfg = self._config.train_config['mu']
        sigma_cfg = self._config.train_config['sigma']

        if 'mu' in self._fit_stages:
            # Update focal loss weights on the shared dataset before
            # constructing loaders — mu and sigma use different alpha/gamma.
            self._dataset.set_loss_function_weights(
                weights_alpha=mu_cfg['loss_weights_alpha'],
                weights_gamma=mu_cfg['loss_weights_gamma'],
            )
            train_loader, val_loader, test_loader = loader_creator.get_loaders(
                batch_size=mu_cfg['batch_size'],
                drop_last=True,
            )
            trainer.train_mu(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                learning_rate=mu_cfg['learning_rate'],
                optimizer=mu_cfg['optimizer'],
                epochs=mu_cfg['epochs'],
                early_stopping_epochs=mu_cfg['early_stopping_epochs'],
                loss_function=mu_cfg['loss_function'],
            )

        if 'sigma' in self._fit_stages:
            self._dataset.set_loss_function_weights(
                weights_alpha=sigma_cfg['loss_weights_alpha'],
                weights_gamma=sigma_cfg['loss_weights_gamma'],
            )
            train_loader, val_loader, test_loader = loader_creator.get_loaders(
                batch_size=sigma_cfg['batch_size'],
                drop_last=True,
            )
            trainer.train_sigma(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                learning_rate=sigma_cfg['learning_rate'],
                optimizer=sigma_cfg['optimizer'],
                epochs=sigma_cfg['epochs'],
                early_stopping_epochs=sigma_cfg['early_stopping_epochs'],
                loss_function=sigma_cfg['loss_function'],
            )

    def _build_model(self, loader_creator: DataLoaderCreator) -> SolarSwinTransformer:
        """Construct a fresh ``SolarSwinTransformer`` for a fold.

        Always initialises in ``point_prediction=True`` mode so that
        stage-1 (mu) training can proceed immediately. If sigma training
        is also requested, ``SolarModelTrainer.train_sigma`` converts the
        model to probabilistic mode internally after reloading the best
        mu checkpoint.

        Args:
            loader_creator: Used to supply the fold-specific fitted
                ``DistributionTransformer`` (output scaler)  and the
                physical_feature_scaler (input scaler) to the model.

        Returns:
            Freshly initialised ``SolarSwinTransformer``.
        """
        mc = self._config.model_config
        return SolarSwinTransformer(
            image_encoder_name=mc['image_encoder'],
            use_pretrained_image_encoder_weights=mc.get(
                'use_pretrained_image_encoder_weights', False
            ),
            num_physical_features=len(self._dataset.physical_features.columns),
            mu_head_hidden_layer_size=mc['mu']['hidden_layer_size'],
            sigma_head_hidden_layer_size=mc['sigma']['hidden_layer_size'],
            mu_head_dropout=mc['mu']['dropout'],
            sigma_head_dropout=mc['sigma']['dropout'],
            clamp_sigma=mc['clamp_sigma'],
            quantile_to_clamp=mc['quantile_to_clamp'],
            physical_upper_limit=mc['physical_upper_limit'],
            point_prediction=True,
            use_multiple_gpus=self._config.train_config['use_multiple_gpus'],
            distribution_transformer=loader_creator.output_scaler,
            physical_feature_scaler=loader_creator.input_scaler,
        )

    # ------------------------------------------------------------------
    # Private: prediction loading and evaluation
    # ------------------------------------------------------------------

    def _load_predictions(
            self, pred_path: Path, stage: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Load saved predictions from a CSV written by ``SolarWindPredictor``.

        For sigma-stage predictions, the three log-normal parameter
        columns are renamed to ``["shape", "scale", "loc"]`` to match
        the ``ProbabilisticPrediction`` constructor's expected column
        names.

        Args:
            pred_path: Path to the prediction CSV.
            stage: Training stage, either ``"mu"`` or ``"sigma"``.

        Returns:
            A 2-tuple ``(pred_df, obs)`` where ``pred_df`` contains
            prediction parameters indexed by datetime and ``obs`` is the
            corresponding ground-truth series.

        Raises:
            FileNotFoundError: If the prediction CSV does not exist.
        """
        if not pred_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: '{pred_path}'. "
                "Ensure training completed successfully."
            )

        df = pd.read_csv(pred_path, index_col=0, parse_dates=[0])
        obs = df['observation']

        if stage == 'sigma':
            pred_df = df[['predicted_lognorm_shape', 'predicted_lognorm_scale',
                          'predicted_lognorm_loc']]
            pred_df.columns = ['shape', 'scale', 'loc']
        else:
            pred_df = df[['predicted_mean']]

        return pred_df, obs

    def _evaluate_fold(
            self,
            pred_df: pd.DataFrame,
            obs: pd.Series,
            fold_idx: int,
            fold_model_name: str,
            stage: str,
            eval_dir: Path,
    ) -> pd.Series:
        """Evaluate predictions for a single fold and save metrics to CSV.

        Args:
            pred_df: Prediction parameters indexed by datetime.
            obs: Ground-truth observations indexed by datetime.
            fold_idx: Zero-based fold index; used to look up the
                corresponding missing-date index for ``HSSFinder``.
            fold_model_name: Full model name including fold suffix
                (e.g. ``"proswin_fold_01"``); used as the output
                filename stem.
            stage: Final training stage (``"mu"`` or ``"sigma"``).
            eval_dir: Directory to write metric CSV files into.

        Returns:
            Evaluation metrics for this fold as a ``pd.Series``.
        """
        pred = self._build_prediction_object(pred_df, stage)
        hss_finder = HSSFinder(self._config.paths['solar_event_dir'])
        interpolation_dates = self._cv_split_missing['test'][fold_idx]

        evaluator = PredictionEvaluator(pred, obs, hss_finder, interpolation_dates)
        results = evaluator.evaluate()
        if isinstance(results, tuple):
            metrics, bs_table = results
            bs_table.to_csv(eval_dir / f"{fold_model_name}_brier_scores.csv")
        else:
            metrics = results

        metrics.to_csv(eval_dir / f"{fold_model_name}_metrics.csv")
        return metrics

    @staticmethod
    def _build_prediction_object(
            pred_df: pd.DataFrame, stage: str
    ) -> PointPrediction | ProbabilisticPrediction:
        """Construct the appropriate prediction container from a DataFrame.

        Args:
            pred_df: Prediction parameters with standardised column
                names (``["shape", "scale", "loc"]`` for sigma,
                ``["predicted_mean"]`` for mu).
            stage: ``"mu"`` for a point prediction or ``"sigma"`` for a
                probabilistic (log-normal) prediction.

        Returns:
            ``ProbabilisticPrediction`` for sigma-stage output;
            ``PointPrediction`` for mu-stage output.
        """
        if stage == 'sigma':
            return ProbabilisticPrediction(pred_df)
        return PointPrediction(pred_df.iloc[:, 0])

    # ------------------------------------------------------------------
    # Private: aggregation
    # ------------------------------------------------------------------

    def _aggregate_and_evaluate(
            self,
            fold_predictions: list[pd.DataFrame],
            fold_observations: list[pd.Series],
            fold_metrics: list[pd.Series],
    ) -> pd.Series:
        """Concatenate fold predictions, evaluate, and log aggregated results.

        Sorts all per-fold predictions and observations into a single
        chronological time series, re-evaluates over the full span
        (including HSS peak metrics), writes summary CSVs, and logs to
        W&B if configured.

        Args:
            fold_predictions: Per-fold prediction DataFrames.
            fold_observations: Per-fold observation Series.
            fold_metrics: Per-fold metric Series (used to compute
                mean/std across folds in the summary table).

        Returns:
            Aggregated evaluation metrics as a ``pd.Series``.
        """
        self._logger.info("Aggregating predictions across all folds ...")

        model_pred_dir = self._config.paths['predictions'] / self._model_name
        model_eval_dir = self._config.paths['evaluation'] / self._model_name

        # Concatenate and sort chronologically
        all_preds = pd.concat(fold_predictions).sort_index()
        all_obs = pd.concat(fold_observations).sort_index()

        # Save aggregated predictions
        agg_pred_path = (
            model_pred_dir / f"{self._model_name}_all_folds_predictions.csv"
        )
        agg_df = all_preds.copy()
        agg_df['observation'] = all_obs
        agg_df.to_csv(agg_pred_path)
        self._logger.info(f"Aggregated predictions saved to {agg_pred_path}")

        # Evaluate aggregated predictions
        final_stage = self._fit_stages[-1]
        pred = self._build_prediction_object(all_preds, final_stage)
        hss_finder = HSSFinder(self._config.paths['solar_event_dir'])
        # Merge missing-date indices from all test folds so the HSSFinder
        # can correctly fill gaps spanning fold boundaries.
        all_missing = pd.DatetimeIndex(
            np.concatenate([m.values for m in self._cv_split_missing['test']])
        ).unique()

        evaluator = PredictionEvaluator(pred, all_obs, hss_finder, all_missing)
        results = evaluator.evaluate()
        if isinstance(results, tuple):
            agg_metrics, bs_table = results
            bs_table.to_csv(
                model_eval_dir / f"{self._model_name}_all_folds_brier_scores.csv"
            )
        else:
            agg_metrics = results

        agg_metrics.to_csv(
            model_eval_dir / f"{self._model_name}_all_folds_metrics.csv"
        )

        # Build a summary table with one row per fold plus mean, std, and
        # the aggregated score (which differs from the per-fold mean because
        # it is computed over the full concatenated time series).
        metrics_df = pd.DataFrame(fold_metrics)
        metrics_df.index = [f"fold_{i + 1:02d}" for i in range(len(fold_metrics))]
        metrics_df.loc['mean'] = metrics_df.mean()
        metrics_df.loc['std'] = metrics_df.std()
        metrics_df.loc['aggregated'] = agg_metrics

        summary_path = (
            model_eval_dir / f"{self._model_name}_cv_metrics_summary.csv"
        )
        metrics_df.to_csv(summary_path)
        self._logger.info(
            f"Cross-validation metrics summary:\n{metrics_df.to_string()}"
        )

        # Log aggregated metrics to a dedicated W&B summary run
        self._log_aggregated_wandb(agg_metrics, metrics_df, summary_path)

        return agg_metrics

    def _log_aggregated_wandb(
            self,
            agg_metrics: pd.Series,
            metrics_df: pd.DataFrame,
            summary_path: Path,
    ) -> None:
        """Log aggregated CV metrics and a summary artifact to W&B.

        Creates a dedicated W&B run (grouped under the model name)
        for the cross-validation summary, logs per-fold mean/std and
        aggregated metrics, and attaches the summary CSV as an artifact.
        No-ops if W&B is not installed or no project is configured.

        Args:
            agg_metrics: Evaluation metrics computed over the full
                concatenated prediction time series.
            metrics_df: Summary table with per-fold rows plus ``mean``,
                ``std``, and ``aggregated`` rows.
            summary_path: Path to the summary CSV to upload as a W&B
                artifact.
        """
        wandb_project = self._config.train_config.get('wandb_project')
        if not _WANDB_AVAILABLE or wandb_project is None:
            return

        wandb.init(
            project=wandb_project,
            name=f"{self._model_name}_cv_summary",
            group=self._model_name,
            config={"model_name": self._model_name, "n_folds": self._n_folds},
        )

        for metric, value in agg_metrics.items():
            wandb.log({f"aggregated/{metric}": value})

        for metric in metrics_df.columns:
            wandb.log({
                f"cv_mean/{metric}": metrics_df.loc['mean', metric],
                f"cv_std/{metric}": metrics_df.loc['std', metric],
            })

        artifact = wandb.Artifact(
            name=f"{self._model_name}_cv_summary",
            type="evaluation",
        )
        artifact.add_file(str(summary_path))
        wandb.log_artifact(artifact)
        wandb.finish()
