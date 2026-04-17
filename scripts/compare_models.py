"""Compare PROSWIN model variants and baselines against literature models.

Evaluates all PROSWIN image-channel configurations, computes baseline
predictions (climatology, persistence, CH area baseline), reads literature model
predictions, and produces plot comparison plots via
:class:`ComparisonPlotter`.
"""
import logging
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from proswin.data_management.cv_split_generator import SequentialCVSplitGenerator
from proswin.predictions import (
    PointPrediction,
    ProbabilisticPrediction,
)
from proswin.evaluation.baselines import (
    BaselinePredictor,
    assign_constant_variance_prediction,
)
from proswin.evaluation.hss_finder import HSSFinder
from proswin.evaluation.model_comparison import ComparisonPlotter
from proswin.evaluation.solar_wind_prediction_evaluator import PredictionEvaluator
from proswin.utils.configurator import ConfigLoader
from proswin.utils.interpolation_dates import interpolate_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full model comparison pipeline.

    Three comparison plots are produced:

    1. **Per-group PROSWIN**: one scatter plot per model group
       (``image_only``, ``full_model``), comparing all channel
       configurations within that group.
    2. **All PROSWIN**: both groups overlaid on a single plot.
    3. **Literature**: the best PROSWIN variant against baselines and
       published models, saved as both a standalone plot and combined
       with the PROSWIN results.

    All figures are saved to ``config.paths['figures']``. The literature
    comparison table is also written to
    ``config.paths['evaluation'] / 'literature_comparison.csv'``.
    """
    project_root = pathlib.Path(__file__).parent.parent
    config = ConfigLoader(project_root, enable_logging=False, initialize_ddp=False)

    # Model groups and channel configurations to evaluate.
    groups = ['image_only', 'full_model']
    model_names = [
        '171', '193', '211', 'HMI',
        '171-193', '171-211', '193-211',
        '171-193-211', '171-193-HMI', '171-211-HMI', '193-211-HMI',
    ]

    # Column names for the log-normal distribution parameters saved by
    # SolarModelTrainer in each prediction CSV.
    lognorm_cols = [
        'predicted_lognorm_shape',
        'predicted_lognorm_scale',
        'predicted_lognorm_loc',
    ]

    results_dict_all_model_groups = {}

    # Read one representative model CSV to recover the dataset's datetime
    # index and gap structure, then recompute the CV split. All model
    # CSVs share the same index, so the choice of model is arbitrary.
    data = pd.read_csv(
        config.paths['predictions'] / f"{model_names[0]}_{groups[0]}.csv",
        index_col=0,
        parse_dates=[0],
    ).sort_index()
    dataset_dates = data.index
    gaps_in_dataset = pd.date_range(
        start=dataset_dates[0], end=dataset_dates[-1], freq='h'
    ).difference(dataset_dates)
    cv_generator = SequentialCVSplitGenerator(
        dataset_dates=dataset_dates, gaps_in_dataset=gaps_in_dataset
    )
    cv_split, cv_split_missing = cv_generator.compute_split(save_split=False)

    # Complete hourly grid covering the full study period, including gaps.
    all_dates = dataset_dates.union(gaps_in_dataset)

    # Collect missing timestamps from every test fold so the evaluator
    # can interpolate them before HSS association (which requires a
    # continuous time series).
    interpolation_dates = pd.DatetimeIndex([])
    for i in range(len(cv_split_missing['test'])):
        interpolation_dates = interpolation_dates.union(cv_split_missing['test'][i])
    interpolation_dates = interpolation_dates.sort_values()

    # --- PROSWIN model group comparison -----------------------------------
    for group in groups:

        hss_finder = HSSFinder(config.paths['solar_event_dir'])
        results_dict = {}

        for model in model_names:
            data = pd.read_csv(
                config.paths['predictions'] / f"{model}_{group}.csv",
                index_col=0,
                parse_dates=[0],
            ).sort_index()

            pred = ProbabilisticPrediction(data.loc[:, lognorm_cols])
            target = data.loc[:, 'observation']

            evaluator = PredictionEvaluator(
                pred, target, hss_finder,
                interpolation_dates=interpolation_dates,
            )
            results = evaluator.evaluate()
            results_dict[model] = results[0]

        model_results_table = pd.DataFrame(results_dict)
        logger.info("Results for group '%s':\n%s", group, model_results_table)
        comp_plotter = ComparisonPlotter(
            model_results_table, config.paths['figures'], save_label=group
        )
        comp_plotter.pred_score_comparison(preset=group)
        results_dict_all_model_groups[group] = model_results_table

    comp_plotter = ComparisonPlotter(
        results_dict_all_model_groups,
        config.paths['figures'],
        save_label='all_proswin',
    )
    comp_plotter.pred_score_comparison(preset='all_proswin')

    # --- Literature comparison --------------------------------------------
    pred_dict = {}
    sws = pd.read_csv(
        config.paths['physical_features_dir'] / 'solar_wind_speed.csv',
        index_col=0,
        parse_dates=[0],
    ).squeeze()

    # Best-performing PROSWIN configuration chosen for head-to-head
    # comparison with published models. Replace with a different model
    # name to compare against a different configuration.
    our_model = "171-211_full_model"
    data = pd.read_csv(
        config.paths['predictions'] / f"{our_model}.csv",
        index_col=0,
        parse_dates=[0],
    ).sort_index()
    our_pred = ProbabilisticPrediction(data.loc[:, lognorm_cols])

    # PROSWIN variants with constant variance (Normal and
    # log-normal), estimated from training data via CV.
    pred_dict['Ours'] = our_pred
    pred_dict['Constant sigma normal'] = assign_constant_variance_prediction(
        our_pred, sws, cv_split, distribution='normal'
    )
    pred_dict['Constant sigma lognormal'] = assign_constant_variance_prediction(
        our_pred, sws, cv_split, distribution='lognormal'
    )

    # Standard solar wind baselines: climatology, 27-day persistence, and
    # a linear regression on coronal-hole area features.
    ch_area = pd.read_csv(
        config.paths['physical_features_dir'] / 'ch_area.csv',
        index_col=0,
        parse_dates=[0],
    )
    ch_area = ch_area.loc[all_dates]
    base_predictor = BaselinePredictor(ch_area, sws)
    pred_clima, pred_persistence, pred_ch_base = base_predictor.compute_predictions(
        cv_split
    )

    pred_dict['Climatological'] = pred_clima
    pred_dict['Persistence'] = pred_persistence
    pred_dict['CH baseline'] = pred_ch_base

    # Literature models are stored as point forecasts; wrap each with a
    # constant Normal uncertainty from an 11-year pre-study period.
    literature_models = pd.read_csv(
        config.paths['predictions'] / 'literature_models.csv',
        index_col=0,
        parse_dates=[0],
    )
    for model in literature_models.columns:
        pred = PointPrediction(literature_models.loc[:, model].dropna())
        pred_dict[model] = assign_constant_variance_prediction(
            pred, sws, distribution='normal'
        )

    # Interpolate small gaps in every prediction, then restrict all
    # predictions to the common date range so metrics are comparable.
    intersection_dates = sws.index
    for model in pred_dict.keys():
        pred_dict[model] = interpolate_predictions(
            pred_dict[model], gap_len_threshold=pd.Timedelta(days=5)
        )
        intersection_dates = intersection_dates.intersection(pred_dict[model].index)

    target = sws.loc[intersection_dates]
    hss_finder = HSSFinder(config.paths['solar_event_dir'])
    results_dict = {}

    for model_name in pred_dict.keys():
        pred_dict[model_name] = pred_dict[model_name].slice(intersection_dates)
        evaluator = PredictionEvaluator(
            pred_dict[model_name],
            target.loc[intersection_dates],
            hss_finder,
            interpolation_dates=pd.DatetimeIndex([]),
        )
        results = evaluator.evaluate()
        results_dict[model_name] = results[0]

    model_results_table = pd.DataFrame(results_dict)
    model_results_table.transpose().to_csv(
        config.paths['evaluation'] / 'literature_comparison.csv'
    )
    print(model_results_table)
    comp_plotter = ComparisonPlotter(
        model_results_table, config.paths['figures'], save_label='literature'
    )
    comp_plotter.pred_score_comparison(preset='literature')

    results_dict_all_model_groups['literature'] = model_results_table

    comp_plotter = ComparisonPlotter(
        results_dict_all_model_groups,
        config.paths['figures'],
        save_label='everything',
    )
    comp_plotter.pred_score_comparison(preset='everything')

    plt.show()


if __name__ == "__main__":
    main()
