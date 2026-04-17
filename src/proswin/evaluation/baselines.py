from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import lognorm
from sklearn.linear_model import LinearRegression

from proswin.predictions import (
    PointPrediction,
    ProbabilisticNormalPrediction,
    ProbabilisticPrediction,
    SolarWindPrediction,
)
from proswin.utils.interpolation_dates import get_interpolation_dates


def _merge_train_val_for_fold(
    sws: pd.Series,
    cv_split: dict,
    fold: int,
) -> tuple[pd.Series, pd.Index]:
    """Merge train and validation observations for one CV fold.

    The gap between train and validation splits was intentionally excluded
    during model training to prevent data leakage (the gap is ~90 days,
    well below the 100-day threshold used here). Those gap timestamps still
    represent valid observations and are included when estimating the
    distributional spread.

    Args:
        sws: Full observed solar wind speed time series.
        cv_split: CV split dict with keys ``"train"``, ``"val"``,
            ``"test"``, each containing a list of ``pd.DatetimeIndex``
            objects, one per fold.
        fold: Zero-based fold index.

    Returns:
        A 2-tuple ``(train_full, test_index)`` where ``train_full`` is
        the merged train + val + leakage-gap observations and
        ``test_index`` is the ``DatetimeIndex`` of the test split for
        this fold.
    """
    train_idx = cv_split['train'][fold]
    val_idx = cv_split['val'][fold]
    test_idx = cv_split['test'][fold]

    train_target = sws.loc[train_idx]
    val_target = sws.loc[val_idx]
    test_index = sws.loc[test_idx].index

    # Combine train and val, then fill the leakage gap with observed sws.
    # gap_len_threshold=100 days: train/val gaps are ~90 days (fill them),
    # while gaps to the test set are much larger (leave them alone).
    train_val = pd.concat([train_target, val_target], axis=0).sort_index()
    gap_result = get_interpolation_dates(
        train_val.index, gap_len_threshold=pd.Timedelta(days=100)
    )
    train_full = pd.concat(
        [train_val, sws.loc[gap_result.missing_dates]], axis=0
    ).sort_index()

    return train_full, test_index


def _estimate_normal_spread(train_data: pd.Series) -> float:
    """Estimate population standard deviation from training observations.

    Uses population variance (no Bessel correction) — the training set
    is treated as the reference population, not a sample from a larger
    one.

    Args:
        train_data: Training observations.

    Returns:
        Population standard deviation.
    """
    sample_mean = train_data.mean()
    return float(np.sqrt(np.mean((train_data.values - sample_mean) ** 2)))


def _make_normal_params(
    mu: float | np.ndarray | pd.Series,
    sigma: float,
    index: pd.Index,
) -> pd.DataFrame:
    """Construct a ``(loc, scale)`` parameter DataFrame for a Gaussian prediction.

    Args:
        mu: Time-varying mean. Scalar, array, or Series — all are
            broadcast to ``len(index)``.
        sigma: Constant standard deviation applied to all timesteps.
        index: DatetimeIndex for the output DataFrame.

    Returns:
        DataFrame with columns ``["loc", "scale"]``, indexed by
        ``index``.
    """
    n = len(index)
    mu_arr = np.broadcast_to(np.asarray(mu, dtype=float), (n,))
    sig_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (n,))
    return pd.DataFrame(
        np.stack([mu_arr, sig_arr], axis=1),
        index=index,
        columns=["loc", "scale"],
    )


def _make_lognormal_params(
    train_data: pd.Series,
    pred_slice: SolarWindPrediction,
) -> pd.DataFrame:
    """Fit constant log-normal spread from training data.

    ``shape`` (sigma_log) and ``loc`` are fixed by MLE on the training
    observations. ``scale`` is back-computed per prediction step so that
    the log-normal mean equals the point forecast's expected value:

        E[X] = loc + scale * exp(shape² / 2)
        => scale = (E[X] − loc) / exp(shape² / 2)

    This keeps the distributional shape constant while anchoring the
    predicted mean to the point forecast.

    Args:
        train_data: Training observations used to fit ``shape`` and
            ``loc`` via MLE.
        pred_slice: Point prediction whose expected values drive the
            per-step ``scale``.

    Returns:
        DataFrame with columns ``["shape", "scale", "loc"]``, indexed
        by ``pred_slice.index``.
    """
    shape, loc, _ = lognorm.fit(train_data.values)
    exp_vals = pred_slice.get_expected_values().values
    scale_pred = (exp_vals - loc) / np.exp(shape ** 2 / 2)
    n = len(exp_vals)
    return pd.DataFrame(
        np.stack(
            [shape * np.ones(n), scale_pred, loc * np.ones(n)], axis=1
        ),
        index=pred_slice.index,
        columns=["shape", "scale", "loc"],
    )


def assign_constant_variance_prediction(
    pred: SolarWindPrediction | pd.Series,
    sws: pd.Series,
    cv_split: dict | None = None,
    distribution: Literal["normal", "lognormal"] = "normal",
) -> ProbabilisticNormalPrediction | ProbabilisticPrediction:
    """Assign a constant distributional spread to a point prediction.

    Converts a point forecast into a probabilistic one by fixing the
    spread of the predictive distribution to a constant estimated from
    training data, while keeping the location time-varying and equal to
    the point forecast.

    Two operating modes are supported:

    **Climatological baseline** (``cv_split=None``): the spread is
    estimated from the 11 years immediately before the first timestamp
    of ``pred``, ensuring zero look-ahead.

    **Cross-validation mode** (``cv_split`` provided): for each fold,
    the spread is estimated from the merged train + validation
    observations (including leakage-gap timestamps). Predictions are
    assembled across all test folds.

    Args:
        pred: Point forecast. Either a ``SolarWindPrediction`` (expected
            values used as loc) or a plain ``pd.Series`` (values used
            directly as loc). Its first index timestamp defines the
            study start in climatological mode.
        sws: Full observed solar wind speed time series. May be
            arbitrarily long — only the relevant sub-period is selected
            internally.
        cv_split: CV split dict with keys ``"train"``, ``"val"``,
            ``"test"``, each a list of ``pd.DatetimeIndex`` per fold.
            If ``None``, the climatological baseline mode is used.
        distribution: Parametric family — ``"normal"`` returns a
            ``ProbabilisticNormalPrediction``; ``"lognormal"`` returns a
            ``ProbabilisticPrediction`` (log-normal).

    Returns:
        Probabilistic prediction with constant spread and time-varying
        location equal to the input point forecast.

    Raises:
        ValueError: If ``distribution`` is not ``"normal"`` or
            ``"lognormal"``.
        TypeError: If ``pred`` is not a ``SolarWindPrediction`` or
            ``pd.Series``.
    """
    if distribution not in ("normal", "lognormal"):
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            "Choose 'normal' or 'lognormal'."
        )
    if not isinstance(pred, (SolarWindPrediction, pd.Series)):
        raise TypeError(
            f"pred must be a SolarWindPrediction or pd.Series, "
            f"got {type(pred).__name__}."
        )

    def _get_mu(
        pred: SolarWindPrediction | pd.Series,
        index: pd.Index,
    ) -> np.ndarray:
        """Extract expected values from ``pred``, aligned to ``index``."""
        if isinstance(pred, SolarWindPrediction):
            return pred.slice(index).get_expected_values().values
        return pred.loc[index].values

    def _get_index(pred: SolarWindPrediction | pd.Series) -> pd.Index:
        """Return the datetime index of ``pred``."""
        return pred.index

    # ------------------------------------------------------------------
    # Cross-validation mode
    # ------------------------------------------------------------------
    if cv_split is not None:
        n_folds = len(cv_split['train'])
        fold_params = []

        # Restrict sws to the study period spanned by all training folds
        # to avoid including data from outside the CV window.
        all_dates = pd.DatetimeIndex([])
        for i in range(n_folds):
            all_dates = all_dates.union(cv_split['train'][i])
        all_dates = all_dates.sort_values()
        sws = sws.loc[all_dates[0]:all_dates[-1]]

        for i in range(n_folds):
            train_full, test_index = _merge_train_val_for_fold(sws, cv_split, i)
            pred_index = test_index.intersection(_get_index(pred))

            if distribution == "normal":
                sigma = _estimate_normal_spread(train_full)
                mu = _get_mu(pred, pred_index)
                params = _make_normal_params(mu, sigma, pred_index)
            else:
                pred_slice = (
                    pred.slice(pred_index)
                    if isinstance(pred, SolarWindPrediction)
                    else PointPrediction(pred.loc[pred_index])
                )
                params = _make_lognormal_params(train_full, pred_slice)

            fold_params.append(params)

        prob_pred_param = pd.concat(fold_params, axis=0).sort_index()

    # ------------------------------------------------------------------
    # Climatological baseline mode
    # ------------------------------------------------------------------
    else:
        study_start = _get_index(pred)[0]
        baseline_start = study_start - relativedelta(years=11)
        sws_baseline = sws.loc[
            (sws.index >= baseline_start) & (sws.index < study_start)
        ]

        if distribution == "normal":
            sigma = _estimate_normal_spread(sws_baseline)
            mu = _get_mu(pred, _get_index(pred))
            params = _make_normal_params(mu, sigma, _get_index(pred))
        else:
            pred_slice = (
                pred if isinstance(pred, SolarWindPrediction)
                else PointPrediction(pred)
            )
            params = _make_lognormal_params(sws_baseline, pred_slice)

        prob_pred_param = params

    if distribution == "normal":
        return ProbabilisticNormalPrediction(prob_pred_param)
    return ProbabilisticPrediction(prob_pred_param)


class BaselinePredictor:
    """Compute cross-validated probabilistic baseline predictions.

    Three baselines are produced for each CV fold:

    - **Climatology**: constant mu equal to the training-set mean,
      constant sigma equal to the training-set standard deviation.
    - **Persistence**: mu taken from the 27-day-lagged solar wind speed
      (one Carrington rotation period), constant sigma from training
      data.
    - **CH baseline**: mu from a linear regression on coronal-hole area
      features and 27-day-lagged sws fitted on training data, constant
      sigma from training data.

    The study period is inferred from the overlap of
    ``ch_area_features`` and ``sws`` — no hardcoded date constants.

    Args:
        ch_area_features: Time series or DataFrame of coronal-hole area
            features, indexed by datetime.
        sws: Observed solar wind speed time series, indexed by datetime.
    """

    # Solar rotation period used for the persistence baseline.
    _PERSISTENCE_LAG = timedelta(days=27)

    def __init__(
        self,
        ch_area_features: pd.Series | pd.DataFrame,
        sws: pd.Series,
    ) -> None:
        # Infer the study period as the overlap between the two inputs.
        # This avoids hardcoded date constants and makes the class reusable
        # for different study windows.
        study_start = max(ch_area_features.index.min(), sws.index.min())
        study_end = min(ch_area_features.index.max(), sws.index.max())

        # Restrict sws and features to the common study period.
        self.target: pd.Series = sws.loc[
            (sws.index >= study_start) & (sws.index <= study_end)
        ]

        # 27-day lag: shift sws forward by one Carrington rotation so that
        # sw27[t] = sws[t − 27 days], the canonical persistence baseline
        # for solar wind forecasting.
        sw27 = sws.copy()
        sw27.index = sw27.index + self._PERSISTENCE_LAG
        sw27 = sw27.loc[
            (sw27.index >= study_start) & (sw27.index <= study_end)
        ]
        sw27.name = "sws27"

        ch = ch_area_features.loc[
            (ch_area_features.index >= study_start)
            & (ch_area_features.index <= study_end)
        ]

        # Input feature matrix: CH area + 27-day-lagged sws.
        self.input_features: pd.DataFrame = pd.concat([ch, sw27], axis=1)

    def compute_predictions(
        self, cv_split: dict
    ) -> tuple[
        ProbabilisticNormalPrediction,
        ProbabilisticNormalPrediction,
        ProbabilisticNormalPrediction,
    ]:
        """Run cross-validated baseline predictions for all folds.

        For each fold, the spread (sigma) is estimated from the merged
        train + validation observations. Fold-level parameter DataFrames
        are concatenated and sorted chronologically before constructing
        the final prediction objects.

        Args:
            cv_split: CV split dict with keys ``"train"``, ``"val"``,
                ``"test"``, each a list of ``pd.DatetimeIndex`` per
                fold.

        Returns:
            A 3-tuple ``(climatology, persistence, ch_baseline)`` of
            ``ProbabilisticNormalPrediction`` objects covering all test
            folds, where:

            - ``climatology``: constant mu = training-set mean.
            - ``persistence``: mu = 27-day-lagged solar wind speed.
            - ``ch_baseline``: mu = linear regression on CH area +
              27-day-lagged sws features.
        """
        n_folds = len(cv_split['train'])
        params_clima = []
        params_persistence = []
        params_ch = []

        for i in range(n_folds):
            train_full, test_index = _merge_train_val_for_fold(
                self.target, cv_split, i
            )
            sigma = _estimate_normal_spread(train_full)

            # -- Climatology: mu is the training-set mean, constant across
            #    all test timesteps. This is the simplest possible baseline.
            mu_clima = train_full.mean()
            params_clima.append(
                _make_normal_params(mu_clima, sigma, test_index)
            )

            # -- Persistence: mu is the 27-day-lagged solar wind speed.
            #    Based on the ~27-day Carrington rotation period, which
            #    causes recurrent solar wind structures.
            mu_persistence = self.input_features.loc[test_index, "sws27"]
            params_persistence.append(
                _make_normal_params(mu_persistence, sigma, test_index)
            )

            # -- CH baseline: mu from a linear regression on CH area features
            #    and 27-day-lagged sws, fitted on training data only.
            train_input = self.input_features.loc[train_full.index]
            test_input = self.input_features.loc[test_index]

            lr = LinearRegression()
            lr.fit(train_input.values, train_full.values)
            mu_ch = lr.predict(test_input.values)
            params_ch.append(
                _make_normal_params(mu_ch, sigma, test_index)
            )

        return (
            ProbabilisticNormalPrediction(
                pd.concat(params_clima, axis=0).sort_index()
            ),
            ProbabilisticNormalPrediction(
                pd.concat(params_persistence, axis=0).sort_index()
            ),
            ProbabilisticNormalPrediction(
                pd.concat(params_ch, axis=0).sort_index()
            ),
        )
