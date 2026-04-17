"""Load and evaluate saved model predictions.

Demonstrates how to read a prediction CSV produced by the training pipeline
and run the full evaluation suite (CRPS, RMSE, Brier scores, HSS metrics).
To evaluate a different model, replace ``'test_model'`` in
``path_to_predictions`` with the desired model name.
"""
import logging
import pathlib

import pandas as pd

from proswin.predictions import ProbabilisticPrediction
from proswin.evaluation.hss_finder import HSSFinder
from proswin.evaluation.solar_wind_prediction_evaluator import PredictionEvaluator
from proswin.utils.configurator import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Load a saved prediction CSV and evaluate it against observations.

    The model name in ``path_to_predictions`` acts as a user-configurable
    placeholder — replace ``'test_model'`` with the name of any model whose
    predictions have been saved to ``config.paths['predictions']``.

    ``interpolation_dates`` controls which temporal gaps are filled before
    HSS association:

    - ``None`` — interpolate all gaps (required for continuous HSS detection).
    - ``pd.DatetimeIndex([])`` — skip interpolation entirely.
    - A specific ``pd.DatetimeIndex`` — interpolate only those timestamps.
    """
    project_root = pathlib.Path(__file__).parent.parent
    config = ConfigLoader(project_root)

    # Replace 'test_model' with the name of the model you want to evaluate.
    # The CSV is written by SolarModelTrainer under config.paths['predictions'].
    path_to_predictions = (
        config.paths['predictions']
        / 'test_model'
        / 'test_model_all_folds_predictions.csv'
    )

    saved_prediction_data = pd.read_csv(
        path_to_predictions, index_col=0, parse_dates=[0]
    )
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
    logger.info("Evaluation results:\n%s", results)
    logger.info("Brier score table:\n%s", bs_table)


if __name__ == "__main__":
    main()
