"""Load and evaluate saved model predictions.

Demonstrates how to read a prediction CSV produced by the training pipeline
and run the full evaluation suite (CRPS, RMSE, Brier scores, HSS metrics).
To evaluate a different model, replace ``'test_model'`` in
``path_to_predictions`` with the desired model name.
"""
import logging
import pathlib

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from proswin.predictions import ProbabilisticPrediction
from proswin.data_management.cv_split_generator import SequentialCVSplitGenerator
from proswin.evaluation.hss_finder import HSSFinder
from proswin.evaluation.solar_wind_prediction_evaluator import PredictionEvaluator
from proswin.evaluation.plotter import (SolarWindPredictionPlotter,
                                        plot_crps_illustration,
                                        plot_cv_split)
from proswin.utils.configurator import ConfigLoader


def main() -> None:
    """Load a saved prediction CSV, evaluate it, and generate all figures.

    Performs the following steps:

    1. Resolve the project root and load all configuration via
       :class:`~proswin.utils.configurator.ConfigLoader`.
    2. Read the prediction CSV and reconstruct a
       :class:`~proswin.predictions.ProbabilisticPrediction` object.
    3. Run the full evaluation suite via
       :class:`~proswin.evaluation.solar_wind_prediction_evaluator.PredictionEvaluator`
       and log all metrics.
    4. Generate all paper figures via
       :class:`~proswin.evaluation.plotter.SolarWindPredictionPlotter`.
    5. Plot a single-timestep CRPS illustration.
    6. Visualise the cross-validation split for the prediction period.

    To evaluate a different model, edit ``path_to_predictions``.

    Notes:
        ``interpolation_dates`` controls which temporal gaps are filled before
        HSS peak association:

        - ``None`` — interpolate all gaps (required for continuous HSS
          detection).
        - ``pd.DatetimeIndex([])`` — skip interpolation entirely.
        - A specific :class:`pandas.DatetimeIndex` — interpolate only those
          timestamps.
    """

    project_root = pathlib.Path(__file__).parent.parent
    config = ConfigLoader(project_root)

    # Replace 'test_model' with the name of the model you want to evaluate.
    # The CSV is written by SolarModelTrainer under config.paths['predictions'].
    path_to_predictions = (
        config.paths['predictions']
        / '171-211_full_model.csv'
    )

    saved_prediction_data = pd.read_csv(
        path_to_predictions, index_col=0, parse_dates=[0]
    )
    # The CSV stores the three log-normal distribution parameters per timestep.
    # ProbabilisticPrediction reconstructs the full predictive distribution
    # from these parameters on demand.
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

    # None interpolates all gaps; HSS association requires a continuous series.
    # Use pd.DatetimeIndex([]) to disable interpolation, or pass a specific
    # DatetimeIndex to interpolate only selected timestamps.
    interpolation_dates = None

    evaluator = PredictionEvaluator(
        predictions, observations, hss_finder, interpolation_dates
    )
    results, bs_table = evaluator.evaluate()
    config.logger.info("Evaluation results:\n%s", results)
    config.logger.info("Brier score table:\n%s", bs_table)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    figure_dir = config.paths["figures"] / config.model_config["model_name"]

    plotter = SolarWindPredictionPlotter(
        predictions=predictions,
        observations=observations,
        figure_dir=figure_dir,
        solar_event_dir=config.paths["solar_event_dir"],
        physical_feature_dir=config.paths["physical_features_dir"],
        interpolation_dates=interpolation_dates,
        hss_finder=hss_finder,
    )
    plotter.plot_all(show=False)

    # Single-timestep CRPS illustration for the paper's methods section.
    crps_date = datetime(2016, 11, 26)
    plot_crps_illustration(
        prediction=predictions.slice([crps_date]),
        observation=observations.loc[crps_date],
        figure_dir=figure_dir,
    )

    # CV split visualisation
    full_hourly_range = pd.date_range(
        predictions.index[0], predictions.index[-1], freq="h"
    )
    gaps_in_dataset = full_hourly_range.difference(predictions.index)
    cv_generator = SequentialCVSplitGenerator(
        dataset_dates=predictions.index,
        gaps_in_dataset=gaps_in_dataset,
    )
    cv_split, _ = cv_generator.compute_split(save_split=False)
    plot_cv_split(cv_split, figure_dir)


    plt.show()

if __name__ == "__main__":
    main()
