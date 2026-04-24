"""Run inference with a trained PROSWIN model over a specified date range.

This script loads a trained SolarSwinTransformer checkpoint, builds a
prediction DataLoader for a user-defined time window, runs inference, and
saves the resulting probabilistic solar wind speed (SWS) forecasts to disk.
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

# Enter here the path to the model you want to use for inference.
# The path is relative to the results/models folder.
MODEL_PATH = "full_model/171-211_fold_01.pth"

# Enter here the start and end dates for inference.
# The dates must be contained in the dataset and the prediction frequency is
# one hour.
PREDICTION_START = datetime(2020, 1, 1, 0)
PREDICTION_END = datetime(2020, 1, 1, 23)

# Enter here the path to the file where the computed predictions should be
# stored. The path is relative to results/predictions.
PREDICTION_PATH = "new_predictions.csv"

def main() -> None:
    """Load a trained model and run inference over a fixed date range.

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

    # ------------------------------------------------------------------
    # Loading model and dataset
    # ------------------------------------------------------------------

    # Load a trained model checkpoint.
    # Trained models are conventionally stored under results/models.
    path_to_model: pathlib.Path = (
        config.paths["models"] / pathlib.Path(MODEL_PATH)
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

    # ------------------------------------------------------------------
    # Set up dataloader
    # ------------------------------------------------------------------

    # Compute the prediction dates.
    prediction_dates = pd.date_range(
        start=PREDICTION_START,
        end=PREDICTION_END,
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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    output_path = config.paths["predictions"] / PREDICTION_PATH
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