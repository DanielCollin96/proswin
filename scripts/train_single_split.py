"""Train the PROSWIN model on a fixed temporal split.

Provides a non-cross-validated training run using hardcoded study-period
date ranges (2010-06-01 – 2024-06-30 split into train / val / test). Use
``scripts/cross_validation.py`` for the full CV pipeline instead.
"""
import pathlib
from datetime import datetime

import pandas as pd
import torch

from proswin.data_management.data_loader_creator import DataLoaderCreator
from proswin.data_management.solar_dataset import SolarDataset
from proswin.predictions import ProbabilisticPrediction
from proswin.evaluation.hss_finder import HSSFinder
from proswin.evaluation.solar_wind_prediction_evaluator import PredictionEvaluator
from proswin.model.solar_model_trainer import SolarModelTrainer
from proswin.model.solar_swin_transformer import SolarSwinTransformer
from proswin.utils.configurator import ConfigLoader


def main() -> None:
    """Entry point for single-split training and evaluation.

    Builds the dataset, trains the PROSWIN model in two stages (mu head
    first with MSE loss, then sigma head with NLL loss), evaluates the
    resulting predictions on the test split, and logs the results. DDP is
    torn down in the ``finally`` block even if initialisation or training
    fails.
    """
    project_root = pathlib.Path(__file__).parent.parent

    # Initialise config before the try-block so that the finally-clause can
    # safely guard on `config is not None` even if ConfigLoader raises.
    config = None
    try:
        config = ConfigLoader(project_root)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = SolarDataset(
            solar_image_dir=config.paths['solar_image_dir'],
            solar_event_dir=config.paths['solar_event_dir'],
            physical_features_dir=config.paths['physical_features_dir'],
            ml_data_dir=config.paths['ml_data_dir'],
            dataset_interval=config.dataset_config['dataset_interval'],
            channels=config.dataset_config['channels'],
            physical_input_data=config.dataset_config['physical_features'],
            solar_wind_speed_persistence_interval=(
                config.dataset_config['solar_wind_speed_persistence']['interval']
            ),
            solar_wind_speed_persistence_cadence=(
                config.dataset_config['solar_wind_speed_persistence']['cadence']
            ),
            sunspot_number_months=config.dataset_config['sunspot_number_months'],
            precompute_and_reuse_dataset=(
                config.dataset_config['precompute_and_reuse_dataset']
            ),
            rank=config.rank,
            logger=config.logger,
        )

        # Fixed study-period boundaries for this non-CV training run.
        # Intersection with dataset.target_dates removes any timestamps
        # that the dataset does not actually contain (e.g. missing obs).
        train_dates = pd.date_range(
            start=datetime(2010, 6, 1), end=datetime(2020, 12, 31, 23), freq="h"
        )
        val_dates = pd.date_range(
            start=datetime(2021, 1, 1), end=datetime(2022, 12, 31, 23), freq="h"
        )
        test_dates = pd.date_range(
            start=datetime(2023, 1, 1), end=datetime(2024, 6, 30, 23), freq="h"
        )


        train_dates = train_dates.intersection(dataset.target_dates)
        val_dates = val_dates.intersection(dataset.target_dates)
        test_dates = test_dates.intersection(dataset.target_dates)

        # Identify the missing hourly timestamps in the test split that the
        # evaluator should interpolate to facilitate a continuous HSS association.
        missing_test = pd.date_range(
            start=test_dates[0], end=test_dates[-1], freq="1h"
        ).difference(test_dates)

        loader_creator = DataLoaderCreator(
            dataset=dataset,
            train_dates=train_dates,
            validation_dates=val_dates,
            test_dates=test_dates,
            use_multiple_gpus=config.train_config['use_multiple_gpus'],
        )

        model = SolarSwinTransformer(
            image_encoder_name=config.model_config['image_encoder'],
            use_pretrained_image_encoder_weights=False,
            num_physical_features=len(dataset.physical_features.columns),
            mu_head_hidden_layer_size=(
                config.model_config['mu']['hidden_layer_size']
            ),
            sigma_head_hidden_layer_size=(
                config.model_config['sigma']['hidden_layer_size']
            ),
            mu_head_dropout=config.model_config['mu']['dropout'],
            sigma_head_dropout=config.model_config['sigma']['dropout'],
            clamp_sigma=config.model_config['clamp_sigma'],
            quantile_to_clamp=config.model_config['quantile_to_clamp'],
            physical_upper_limit=config.model_config['physical_upper_limit'],
            point_prediction=True,
            use_multiple_gpus=config.train_config['use_multiple_gpus'],
            distribution_transformer=loader_creator.output_scaler,
            physical_feature_scaler=loader_creator.input_scaler,
        )

        wandb_config = SolarModelTrainer.build_wandb_config(
            train_config=config.train_config,
            output_scaler=loader_creator.output_scaler,
        )

        model_name = config.model_config['model_name']
        trainer = SolarModelTrainer(
            model=model,
            device=device,
            model_dir=config.paths['results'] / 'models',
            pred_dir=config.paths['predictions'],
            epoch_losses_dir=config.paths['results'] / 'epoch_losses',
            use_multiple_gpus=config.train_config['use_multiple_gpus'],
            wandb_project=config.train_config.get('wandb_project'),
            wandb_run_name=None,
            model_name=model_name,
            train_config_to_save_in_wandb=wandb_config,
            logger=config.logger,
        )

        # --- Stage 1: train the mu (point-forecast) head with MSE loss ------
        dataset.set_loss_function_weights(
            weights_alpha=config.train_config['mu']['loss_weights_alpha'],
            weights_gamma=config.train_config['mu']['loss_weights_gamma'],
        )
        train_loader, val_loader, test_loader = loader_creator.get_loaders(
            batch_size=config.train_config['mu']['batch_size'],
            num_workers=None,
            drop_last=True,
        )
        trainer.train_mu(
            train_loader,
            val_loader,
            test_loader,
            learning_rate=config.train_config['mu']['learning_rate'],
            optimizer=config.train_config['mu']['optimizer'],
            epochs=config.train_config['mu']['epochs'],
            early_stopping_epochs=config.train_config['mu']['early_stopping_epochs'],
            loss_function=config.train_config['mu']['loss_function'],
        )

        # --- Stage 2: freeze encoder + mu head, train sigma with NLL loss ---
        dataset.set_loss_function_weights(
            weights_alpha=config.train_config['sigma']['loss_weights_alpha'],
            weights_gamma=config.train_config['sigma']['loss_weights_gamma'],
        )
        train_loader, val_loader, test_loader = loader_creator.get_loaders(
            batch_size=config.train_config['sigma']['batch_size'],
            num_workers=None,
            drop_last=True,
        )
        trainer.train_sigma(
            train_loader,
            val_loader,
            test_loader,
            learning_rate=config.train_config['sigma']['learning_rate'],
            optimizer=config.train_config['sigma']['optimizer'],
            epochs=config.train_config['sigma']['epochs'],
            early_stopping_epochs=(
                config.train_config['sigma']['early_stopping_epochs']
            ),
            loss_function=config.train_config['sigma']['loss_function'],
        )
        trainer.finish_logging()

        # --- Evaluate on the test split -------------------------------------
        csv_path = (
            config.paths['predictions']
            / f"{config.model_config['model_name']}_sigma_test.csv"
        )
        saved_prediction_data = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
        lognorm_cols = [
            'predicted_lognorm_shape',
            'predicted_lognorm_scale',
            'predicted_lognorm_loc',
        ]
        predictions = ProbabilisticPrediction(
            saved_prediction_data.loc[:, lognorm_cols]
        )
        observations = saved_prediction_data.loc[:, 'observation']
        hss_finder = HSSFinder(config.paths['solar_event_dir'])

        evaluator = PredictionEvaluator(
            predictions, observations, hss_finder, missing_test
        )
        results, bs_table = evaluator.evaluate()
        config.logger.info("Evaluation results:\n%s", results)
        config.logger.info("Brier score table:\n%s", bs_table)

    finally:
        if config is not None:
            config.terminate_ddp()


if __name__ == "__main__":
    main()
