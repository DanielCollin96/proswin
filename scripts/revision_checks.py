"""Load and evaluate saved model predictions.

Demonstrates how to read a prediction CSV produced by the training pipeline
and run the full evaluation and plotting pipeline.
"""
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


# Enter here the path to the file where the predictions are
# stored. The path is relative to results/predictions.
PREDICTION_PATH = "predictions_best_model.csv"

# Enter here the directory where the evaluation results should
# be stored. The path is relative to results/evaluation.
EVALUATION_DIRECTORY = "new_evaluation"

# Enter here the directory where the figures should
# stored. The path is relative to results/figures.
FIGURE_DIRECTORY = "new_figures"

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

    # ------------------------------------------------------------------
    # Loading predictions and observations
    # ------------------------------------------------------------------

    # Replace 'test_model' with the name of the model you want to evaluate.
    # The CSV is written by SolarModelTrainer under config.paths['predictions'].
    path_to_predictions = (
        config.paths['predictions']
        / pathlib.Path(PREDICTION_PATH)
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

    ##################################
    # Quick check: does the model's predicted sigma or mu react to CMEs?
    # cme_list defines each CME's disturbance window (start/end) — these
    # events are not represented in the model's input features (see
    # cme_filter.delete_cmes). Predicted sigma is the actual lognormal std
    # in km/s (not the raw shape parameter); the mu effect is checked via
    # the residual (obs - predicted mean). Uses the full available period.
    import numpy as np
    from scipy.stats import lognorm as _lognorm, mannwhitneyu
    from proswin.data_management.cme_filter import delete_cmes

    cme_list = HSSFinder(config.paths['solar_event_dir']).cme_list
    _, mask_no_cme = delete_cmes(
        observations, cme_list, return_mask=True, delete_next_rotation=False
    )
    mask_cme = ~mask_no_cme

    predicted_sigma = pd.Series(
        _lognorm.std(predictions.shape, loc=predictions.loc, scale=predictions.scale),
        index=predictions.index,
    )
    residual = observations - predictions.get_expected_values()
    crps_per_step = predictions.get_crps(observations)

    def _cme_summary(mask: np.ndarray, label: str) -> None:
        print(f"--- {label} (n={mask.sum()}) ---")
        print(f"  mean predicted sigma:    {predicted_sigma[mask].mean():.4f}")
        print(f"  mean residual (obs-mu):  {residual[mask].mean():.4f}")
        print(f"  mean |residual|:         {residual[mask].abs().mean():.4f}")
        print(f"  mean CRPS:               {crps_per_step[mask].mean():.4f}")

    _cme_summary(mask_cme, "CME windows")
    _cme_summary(~mask_cme, "Non-CME")

    _, p_sigma = mannwhitneyu(
        predicted_sigma[mask_cme], predicted_sigma[~mask_cme], alternative="two-sided"
    )
    _, p_res = mannwhitneyu(
        residual[mask_cme].abs(), residual[~mask_cme].abs(), alternative="two-sided"
    )
    print(f"Mann-Whitney U p-value (sigma, CME vs non-CME):      {p_sigma:.4g}")
    print(f"Mann-Whitney U p-value (|residual|, CME vs non-CME): {p_res:.4g}")
    ###############################

    ##################################
    # restrict to June-December 2020
    #observations = observations.loc[datetime(2020,6,1):datetime(2020,12,31,23)]
    observations = observations.loc[:datetime(2019,11,30,23)]

    predictions = predictions.slice(observations.index)
    print(predictions.index)
    print(observations)

    # Quick check: forecast skill vs. an in-sample empirical climatology.
    # Climatology = the observed values themselves, used unchanged as the
    # ensemble forecast at every timestep. CRPS is computed via scoringrules'
    # ensemble estimator (empirical CDF), no parametric distribution fitted.
    import numpy as np
    from scoringrules import crps_ensemble

    n = len(observations)
    #climatology_ensemble = np.tile(observations.values, (n, 1))
    #crps_climatology = crps_ensemble(observations.values, climatology_ensemble).mean()
    crps_model = predictions.get_crps(observations).mean()
    #forecast_skill = 1 - crps_model / crps_climatology

    print(f"CRPS model: {crps_model:.4f}")
    #print(f"CRPS climatology: {crps_climatology:.4f}")
    #print(f"Forecast skill vs climatology: {forecast_skill:.4f}")

    ###############################
    '''
    import numpy as np

    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    import matplotlib.pyplot as plt

    def residual_autocorrelation(y_true: pd.Series, y_pred: pd.Series, nlags: int = 48):
        """
        Compute and visualize autocorrelation of prediction residuals.

        Assumes y_true and y_pred share a regular hourly DatetimeIndex, with no
        time gaps (reindex/interpolate beforehand if needed, and check for gaps
        explicitly).

        Parameters
        ----------
        y_true, y_pred : pd.Series
            Observed and predicted values (e.g. predicted mean/median), aligned index.
        nlags : int
            Number of lags (hours) to compute ACF for.

        Returns
        -------
        dict with 'acf_values', 'ljung_box' results, and residuals.
        """
        residuals = (y_true - y_pred).dropna()

        # sanity check: warn if index has gaps (breaks ACF interpretation)
        inferred_freq = pd.infer_freq(residuals.index)
        if inferred_freq is None:
            print("Warning: index has no consistent frequency — check for gaps "
                  "before trusting ACF results.")

        # ACF plot
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(residuals, lags=nlags, ax=ax)
        ax.set_title("Autocorrelation of prediction residuals")
        ax.set_xlabel("Lag (hours)")
        plt.tight_layout()

        # ACF plot
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(y_pred, lags=nlags, ax=ax)
        ax.set_title("Autocorrelation of predictions")
        ax.set_xlabel("Lag (hours)")
        plt.tight_layout()

        # Ljung-Box test: formal test of "residuals are white noise" (H0)
        # Rejecting H0 (small p-value) = significant autocorrelation present
        lb_results = acorr_ljungbox(residuals, lags=[1, 6, 12, 24, 48], return_df=True)

        return {
            "residuals": residuals,
            "ljung_box": lb_results,
            "figure": fig,
        }

    exp_val = predictions.get_expected_values()
    # usage
    result = residual_autocorrelation(observations, exp_val, nlags=48)
    print(result["ljung_box"])
    # check the actual magnitude, not just significance
    for lag in [1, 6, 12, 24, 48]:
        print(f"Residuals: lag {lag}: acf = {result['residuals'].autocorr(lag=lag):.4f}")
    for lag in [1, 6, 12, 24, 48]:
        print(f"Predictions: lag {lag}: acf = {exp_val.autocorr(lag=lag):.4f}")
    plt.show()
    exit()
    '''
    ##############################

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    hss_finder = HSSFinder(config.paths['solar_event_dir'])

    # None interpolates all gaps; HSS association requires a continuous series.
    # Use pd.DatetimeIndex([]) to disable interpolation, or pass a specific
    # DatetimeIndex to interpolate only selected timestamps.
    interpolation_dates = None

    evaluator = PredictionEvaluator(
        predictions, observations, hss_finder, interpolation_dates
    )
    results, bs_table = evaluator.evaluate()


    #p,o = hss_finder.find(predictions, interpolation_dates)
    #print(o.loc[:,'peak_value'].sort_values())

    eval_dir = config.paths['evaluation'] / pathlib.Path(EVALUATION_DIRECTORY)
    eval_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(eval_dir / 'metrics.csv')
    bs_table.to_csv(eval_dir / 'brier_scores.csv')

    config.logger.info("Evaluation results:\n%s", results)
    config.logger.info("Brier score table:\n%s", bs_table)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    figure_dir = config.paths["figures"] / pathlib.Path(FIGURE_DIRECTORY)

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
