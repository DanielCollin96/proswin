"""Run inference with a trained PROSWIN model over a specified date range.

This script loads a trained SolarSwinTransformer checkpoint, builds a
prediction DataLoader for a user-defined time window, runs inference, and
saves the resulting probabilistic solar wind speed (SWS) forecasts to disk.

Typical usage::

    python scripts/compare_models.py

The model path and all data/output directories are resolved from the project
configuration file (``config.yaml`` at the repository root).  To predict with
a different model, edit the ``path_to_model`` variable near the top of
``main()``.

Pipeline position: run *after* training (``train.py`` / CV runner) and
*before* any downstream evaluation or plotting scripts.
"""
import pathlib
from datetime import datetime

import pandas as pd
import torch

from proswin.data_management.data_loader_creator import get_prediction_loader
from proswin.data_management.solar_dataset import SolarDataset
from proswin.model.solar_swin_transformer import load_model
from proswin.model.solar_wind_predictor import SolarWindPredictor
from proswin.utils.configurator import ConfigLoader


def main() -> None:
    """Load a trained model and run inference over a fixed date range.
    To change the model or the prediction window, edit the ``path_to_model``
    variable and the ``pd.date_range(...)`` call inside this function.

    Raises:
        FileNotFoundError: If ``path_to_model`` does not point to an existing
            checkpoint file.  Raised implicitly by :func:`load_model`.
    """
    project_root = pathlib.Path(__file__).parent.parent
    config = ConfigLoader(project_root)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    config.logger.info("Using device: %s", device)

    # Model — edit this path to switch between different trained checkpoints.
    # Trained models are conventionally stored under config.paths['models'].
    path_to_model: pathlib.Path = (
        config.paths["models"] / "171-211_full_model_fold_01.pth"
    )
    config.logger.info("Loading model from: %s", path_to_model)
    model = load_model(path_to_model)


    # Construct the dataset using data given in the data directory
    dataset = SolarDataset(
        solar_image_dir=config.paths["solar_image_dir"],
        solar_event_dir=config.paths["solar_event_dir"],
        physical_features_dir=config.paths["physical_features_dir"],
        ml_data_dir=config.paths["ml_data_dir"],
        dataset_interval=config.dataset_config["dataset_interval"],
        channels=config.dataset_config["channels"],
        physical_input_data=config.dataset_config["physical_features"],
        solar_wind_speed_persistence_interval=(
            config.dataset_config["solar_wind_speed_persistence"]["interval"]
        ),
        solar_wind_speed_persistence_cadence=(
            config.dataset_config["solar_wind_speed_persistence"]["cadence"]
        ),
        sunspot_number_months=config.dataset_config["sunspot_number_months"],
        precompute_and_reuse_dataset=(
            config.dataset_config["precompute_and_reuse_dataset"]
        ),
        rank=config.rank,
        logger=config.logger,
    )

    # Edit start/end dates to change the prediction dates.
    prediction_dates = pd.date_range(
        start=datetime(2020, 1, 1, 0),
        end=datetime(2020, 1, 1, 23),
        freq="1h",
    )
    config.logger.info(
        "Prediction window: %s → %s (%d steps)",
        prediction_dates[0],
        prediction_dates[-1],
        len(prediction_dates),
    )

    pred_loader = get_prediction_loader(
        dataset=dataset,
        dates=prediction_dates,
        input_scaler=model.physical_feature_scaler,
        output_scaler=model.distribution_transformer,
        batch_size=16,
        num_workers=None,
    )

    # Run inference
    output_path = config.paths["predictions"] / "new_predictions.csv"
    predictor = SolarWindPredictor(model, device=device, logger=config.logger)

    config.logger.info("Running inference ...")
    predictions, targets = predictor.predict(pred_loader, output_path=output_path)
    config.logger.info("Predictions saved to: %s", output_path)

    # Show information extracted from the probabilistic predictions.
    config.logger.info(
        "Predicted expected SWS values:\n%s",
        predictions.get_expected_values(),
    )
    config.logger.info(
        "Predicted 95th-percentile SWS values:\n%s",
        predictions.get_quantiles(0.95),
    )


if __name__ == "__main__":
    main()