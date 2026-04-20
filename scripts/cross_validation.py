"""Run the full cross-validation training and evaluation pipeline.

Loads configuration from ``configs/config.yaml``, builds the dataset and CV
split, then delegates to :class:`CrossValidationRunner` to train and evaluate
the PROSWIN model across all folds. Designed to run on a single node or under
DDP on HPC/SLURM.
"""
import pathlib

import torch

from proswin.data_management.cv_split_generator import SequentialCVSplitGenerator
from proswin.data_management.solar_dataset import SolarDataset
from proswin.model.cv_runner import CrossValidationRunner
from proswin.utils.configurator import ConfigLoader


def main() -> None:
    """Entry point for cross-validated training and evaluation.

    Resolves the project root from this file's location, loads the central
    configuration, builds the :class:`SolarDataset`, generates the CV split,
    and runs all folds via :class:`CrossValidationRunner`. DDP is always torn
    down in the ``finally`` block — even if initialisation fails — to avoid
    leaving stale process groups.
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

        cv_generator = SequentialCVSplitGenerator(
            dataset_dates=dataset.target_dates,
            gaps_in_dataset=dataset.missing_dates,
        )
        cv_split, cv_split_missing = cv_generator.compute_split(
            save_split=True,
            ml_data_dir=config.paths['ml_data_dir'],
        )

        runner = CrossValidationRunner(
            dataset=dataset,
            cv_split=cv_split,
            cv_split_missing=cv_split_missing,
            config=config,
            device=device,
            logger=config.logger,
        )
        runner.run()

    finally:
        if config is not None:
            config.terminate_ddp()


if __name__ == "__main__":
    main()
