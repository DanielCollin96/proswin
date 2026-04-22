from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proswin.predictions import (
    PointPrediction,
    SolarWindPrediction,
)
from proswin.evaluation.hss_finder import HSSFinder


class PredictionEvaluator:
    """Compute evaluation metrics for solar wind predictions against observations.

    Supported metrics:

    - **RMSE**: root mean squared error of the expected value.
    - **mean CRPS**: mean continuous ranked probability score. ``NaN`` for
      ``PointPrediction``.
    - **HSS peak RMSE** (optional): RMSE computed only on matched HSS peak
      values.
    - **HSS peak mean CRPS** (optional): mean CRPS at matched HSS peak dates.
      Returns ``NaN`` for ``PointPrediction``.
    - **Binary event metrics** (optional): POD, FAR, TS, BS from the
      hit/miss/false-alarm counts of matched HSS peaks.
    - **Threshold Brier scores** (probabilistic only): Brier score and skill
      score at fixed speed thresholds.

    HSS metrics are only computed when an ``HSSFinder`` instance is provided.

    Args:
        prediction: Any concrete subclass of ``SolarWindPrediction``.
        observations: Ground-truth values aligned with the prediction index.
        hss_finder: Pre-constructed finder for HSS peak detection and
            association. If None, HSS metrics are omitted from
            :meth:`evaluate`.
        interpolation_dates: Missing datetimes to fill before HSS peak
            detection. Passed directly to :meth:`HSSFinder.find`. If None,
            gaps are inferred automatically.

    Raises:
        ValueError: If ``prediction.index`` and ``observations.index`` do not
            align.
    """

    def __init__(
        self,
        prediction: SolarWindPrediction,
        observations: pd.Series,
        hss_finder: HSSFinder | None = None,
        interpolation_dates: pd.DatetimeIndex | None = None,
    ) -> None:
        if not prediction.index.equals(observations.index):
            raise ValueError(
                "Prediction and observation indices do not align. "
                "Ensure both cover the same datetimes."
            )
        self._prediction = prediction
        self._observations = observations
        self._hss_finder = hss_finder
        self._interpolation_dates = interpolation_dates

        # HSS peak DataFrames are computed lazily and cached after first call
        self._pred_peaks: pd.DataFrame | None = None
        self._obs_peaks: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public metrics
    # ------------------------------------------------------------------

    def rmse(
        self,
        pred: SolarWindPrediction | None = None,
        obs: pd.Series | np.ndarray | None = None,
    ) -> float:
        """Compute RMSE of the expected value against observations.

        If both ``pred`` and ``obs`` are None, the instance's stored
        prediction and observations are used.

        Args:
            pred: Prediction to evaluate. If None, uses the stored prediction.
            obs: Observations to compare against. If None, uses the stored
                observations.

        Returns:
            Root mean squared error in original target units.

        Raises:
            RuntimeError: If exactly one of ``pred`` / ``obs`` is None.
        """
        if pred is None and obs is None:
            pred = self._prediction
            obs = self._observations
        elif (pred is None) != (obs is None):
            raise RuntimeError(
                "Either both 'pred' and 'obs' must be provided, "
                "or both must be None."
            )

        expected_values = pred.get_expected_values().values
        residuals = expected_values - obs.values
        return float(np.sqrt(np.mean(residuals ** 2)))

    def crps(
        self,
        pred: SolarWindPrediction | None = None,
        obs: pd.Series | np.ndarray | None = None,
    ) -> float:
        """Compute mean CRPS of the predictive distribution against observations.

        Returns NaN for ``PointPrediction``, which returns NaN per sample
        from :meth:`~SolarWindPrediction.get_crps`.

        If both ``pred`` and ``obs`` are None, the instance's stored
        prediction and observations are used.

        Args:
            pred: Prediction to evaluate. If None, uses the stored prediction.
            obs: Observations to compare against. If None, uses the stored
                observations.

        Returns:
            Mean CRPS as a scalar float.

        Raises:
            RuntimeError: If exactly one of ``pred`` / ``obs`` is None.
        """
        if pred is None and obs is None:
            pred = self._prediction
            obs = self._observations
        elif (pred is None) != (obs is None):
            raise RuntimeError(
                "Either both 'pred' and 'obs' must be provided, "
                "or both must be None."
            )

        return float(pred.get_crps(obs).mean())

    def calibration_errors(
            self,
            pred: SolarWindPrediction | None = None,
            obs: pd.Series | None = None,
            quantile_levels: np.ndarray | None = None,
    ) -> pd.Series:
        """Compute summary calibration errors between nominal and empirical coverage.

        For each nominal confidence level q, the symmetric q-prediction
        interval is ``[Q((1-q)/2), Q((1+q)/2)]``.  The empirical coverage is
        the fraction of observations that fall inside this interval.  The
        signed error ``q - empirical_coverage`` measures miscalibration:
        positive values indicate under-coverage (intervals too narrow);
        negative values indicate over-coverage (intervals too wide).

        The signature mirrors :meth:`rmse` and :meth:`crps`: when both
        ``pred`` and ``obs`` are ``None``, the instance's stored prediction
        and observations are used.  Passing an explicit ``pred`` / ``obs``
        pair evaluates calibration on an arbitrary slice — for example, the
        HSS peak subset used by :meth:`evaluate`.

        Returns ``NaN`` for all metrics when called on a ``PointPrediction``,
        because point forecasts do not define a predictive distribution.

        Args:
            pred: Prediction to evaluate.  If ``None``, uses the stored
                prediction.
            obs: Observations to compare against.  If ``None``, uses the
                stored observations.
            quantile_levels: 1-D array of nominal coverage probabilities in
                (0, 1).  Defaults to 99 evenly spaced values from 0.01 to
                0.99.

        Returns:
            Series with index
            ``['mean_calib_error', 'mean_abs_calib_error', 'max_abs_calib_error']``.

        Raises:
            RuntimeError: If exactly one of ``pred`` / ``obs`` is provided.
        """
        if pred is None and obs is None:
            pred = self._prediction
            obs = self._observations
        elif (pred is None) != (obs is None):
            raise RuntimeError(
                "Either both 'pred' and 'obs' must be provided, "
                "or both must be None."
            )

        # PointPrediction has no predictive distribution — return NaN to keep
        # the metrics Series shape consistent with probabilistic predictions.
        if isinstance(pred, PointPrediction):
            return pd.Series(
                {
                    "mean_calib_error": np.nan,
                    "mean_abs_calib_error": np.nan,
                    "max_abs_calib_error": np.nan,
                }
            )

        if quantile_levels is None:
            quantile_levels = np.linspace(0.01, 0.99, 99)

        empirical_coverage = []
        for q in quantile_levels:
            lower = pred.get_quantiles((1 - q) / 2)
            upper = pred.get_quantiles((1 + q) / 2)
            covered = float(
                np.mean(
                    (obs.values >= lower.values) & (obs.values <= upper.values)
                )
            )
            empirical_coverage.append(covered)

        errors = quantile_levels - np.array(empirical_coverage)

        return pd.Series(
            {
                "mean_calib_error": float(np.mean(errors)),
                "mean_abs_calib_error": float(np.mean(np.abs(errors))),
                "max_abs_calib_error": float(np.max(np.abs(errors))),
            }
        )

    def evaluate(self) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
        """Compute all applicable metrics and return results.

        Always computes ``rmse`` and ``crps``. When an ``HSSFinder`` was
        provided, also computes ``peak_rmse``, ``peak_crps``, ``pod``,
        ``far``, ``ts``, ``bs``, ``ps_rmse``, and ``ps_crps``. For
        probabilistic predictions, also returns threshold Brier scores.

        Returns:
            For ``PointPrediction``: a ``pd.Series`` of metric values.
            For probabilistic predictions: a 2-tuple
            ``(metrics, brier_scores)`` where ``metrics`` is a ``pd.Series``
            and ``brier_scores`` is a ``pd.DataFrame`` indexed by speed
            threshold.
        """
        metrics: dict[str, float] = {
            "rmse": self.rmse(),
            "crps": self.crps(),
        }
        calib = self.calibration_errors()
        metrics.update(calib.to_dict())

        if self._hss_finder is not None:
            pred_peaks, obs_peaks = self._get_hss_peaks()
            pred_associated_peaks, obs_associated_peaks = self._matched_peaks(
                pred_peaks, obs_peaks
            )
            metrics["peak_rmse"] = self.rmse(pred_associated_peaks, obs_associated_peaks)
            metrics["peak_crps"] = self.crps(pred_associated_peaks, obs_associated_peaks)

            peak_calib = self.calibration_errors(
                pred_associated_peaks, obs_associated_peaks
            )
            metrics.update({f"peak_{k}": v for k, v in peak_calib.items()})

            event_metrics = self._hss_binary_event_metrics()
            metrics = metrics | event_metrics.to_dict()

            metrics["ps_rmse"] = self._prediction_score(
                metrics["rmse"], metrics["peak_rmse"]
            )
            metrics["ps_crps"] = self._prediction_score(
                metrics["crps"], metrics["peak_crps"]
            )


        if not isinstance(self._prediction, PointPrediction):
            brier_score_table = self._threshold_brier_scores([450, 500, 600, 700])
            return pd.Series(metrics), brier_score_table

        return pd.Series(metrics)

    # ------------------------------------------------------------------
    # Private metrics helpers
    # ------------------------------------------------------------------

    def _hss_binary_event_metrics(self) -> pd.Series:
        """Compute binary event verification metrics from cached HSS peaks.

        Probability of Detection (POD), False Alarm Ratio (FAR), Threat Score
        (TS/CSI), and Bias Score (BS) are derived from the hit/false-alarm/miss
        counts in the cached peak DataFrames. All metrics are NaN if the peaks
        are missing, empty, or contain non-finite values.

        Returns:
            Series with index ``['pod', 'far', 'ts', 'bs']``.

        Raises:
            RuntimeError: If HSS peaks have not been computed yet — call
                ``evaluate()`` or ``_get_hss_peaks()`` first.
        """
        if self._pred_peaks is None or self._obs_peaks is None:
            raise RuntimeError(
                "_hss_binary_event_metrics() requires HSS peaks to be "
                "computed first. Call evaluate() or _get_hss_peaks() before "
                "calling this method directly."
            )

        # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
        if self._event_metrics_computable(
            self._pred_peaks.loc[:, ["hit", "false_alarm"]],
            self._obs_peaks.loc[:, "miss"],
        ):
            TP = self._pred_peaks.loc[:, "hit"].sum()
            FP = self._pred_peaks.loc[:, "false_alarm"].sum()
            FN = self._obs_peaks.loc[:, "miss"].sum()

            pod = TP / (TP + FN)
            far = FP / (TP + FP)
            ts = TP / (TP + FP + FN)
            bs = (TP + FP) / (TP + FN)
        else:
            pod = np.nan
            far = np.nan
            ts = np.nan
            bs = np.nan

        return pd.Series([pod, far, ts, bs], index=["pod", "far", "ts", "bs"])

    @staticmethod
    def _prediction_score(
        timeline_error: float,
        peak_error: float,
        w_timeline: float = 2.0,
        w_peak: float = 1.0,
    ) -> float:
        """Compute a weighted harmonic mean of timeline and peak error metrics.

        Combines a full-series error and a peak-only error into a single
        scalar. Like RMSE and CRPS, lower values indicate better predictions.
        The harmonic mean penalises large values in either component more
        strongly than an arithmetic mean would.

        Args:
            timeline_error: Error over the full prediction period (e.g. RMSE
                or mean CRPS).
            peak_error: Error over matched HSS peak dates only.
            w_timeline: Weight for the timeline error. Default is 2.0.
            w_peak: Weight for the peak error. Default is 1.0.

        Returns:
            Weighted harmonic mean of the two error metrics.

        Raises:
            ValueError: If weights are negative or errors are non-positive.
        """
        if w_timeline < 0 or w_peak < 0:
            raise ValueError("Weights must not be negative.")
        if timeline_error <= 0 or peak_error <= 0:
            raise ValueError("Errors must be positive.")
        pred_score = (w_timeline + w_peak) / (
            (w_timeline / timeline_error) + (w_peak / peak_error)
        )
        return float(pred_score)

    @staticmethod
    def _event_metrics_computable(
        prediction: pd.DataFrame,
        observation: pd.Series,
    ) -> bool:
        """Return True if the inputs are non-None, non-empty, and finite.

        Args:
            prediction: DataFrame of predicted event flags (hit/false-alarm).
            observation: Series of observed event flags (miss).

        Returns:
            True if binary event metrics can be safely computed; False
            otherwise.
        """
        if prediction is None or observation is None:
            return False
        if len(prediction) == 0 or len(observation) == 0:
            return False
        if (
            np.any(np.isnan(prediction))
            or np.any(np.isnan(observation))
            or np.any(np.isinf(prediction))
            or np.any(np.isinf(observation))
        ):
            return False
        return True

    def _threshold_brier_scores(self, thresholds: list[float]) -> pd.DataFrame:
        """Compute Brier score and Brier skill score at each speed threshold.

        For each threshold, binary exceedance observations are constructed
        (1 if observed speed ≥ threshold) and compared with the predicted
        exceedance probability ``1 - CDF(threshold)`` from the predictive
        distribution.

        Args:
            thresholds: Solar wind speed thresholds (km/s) at which to
                evaluate binary event scores.

        Returns:
            DataFrame indexed by threshold with columns ``event_rate``,
            ``brier_score``, and ``brier_skill_score``.
        """
        results = []

        for thr in thresholds:
            # Binary observations: 1 if the event (speed >= threshold) occurred
            y = (self._observations >= thr).astype(int)

            # Predicted exceedance probability: P(X >= thr) = 1 - CDF(thr)
            cdf = self._prediction.get_cdfs(thr)
            p = 1.0 - cdf

            brier_score = np.mean((p - y) ** 2)

            # Climatological forecast: always predict the observed event rate
            p_clim = np.mean(y)
            climatology_brier_score = np.mean((p_clim - y) ** 2)

            if climatology_brier_score == 0:
                brier_skill_score = np.nan
            else:
                brier_skill_score = 1 - brier_score / climatology_brier_score

            results.append({
                "threshold": thr,
                "event_rate": p_clim,
                "brier_score": brier_score,
                "brier_skill_score": brier_skill_score,
            })

        return pd.DataFrame(results).set_index("threshold")

    # ------------------------------------------------------------------
    # Private: HSS peak helpers
    # ------------------------------------------------------------------

    def _get_hss_peaks(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return cached HSS peak DataFrames, computing them on first call.

        Returns:
            A 2-tuple ``(pred_peaks, obs_peaks)`` of peak DataFrames.

        Raises:
            RuntimeError: If no ``HSSFinder`` was provided at construction.
        """
        if self._hss_finder is None:
            raise RuntimeError(
                "HSS metrics require an HSSFinder instance. "
                "Pass hss_finder= at PredictionEvaluator construction."
            )
        if self._pred_peaks is None:
            # Pass the full SolarWindPrediction — HSSFinder extracts
            # expected values internally
            self._pred_peaks, self._obs_peaks = self._hss_finder.find(
                self._prediction, self._interpolation_dates
            )
        return self._pred_peaks, self._obs_peaks

    def _matched_peaks(
        self,
        pred_peaks: pd.DataFrame,
        obs_peaks: pd.DataFrame,
    ) -> tuple[SolarWindPrediction, pd.Series]:
        """Extract predicted and observed values at matched (hit) peak dates.

        Args:
            pred_peaks: Predicted peak DataFrame with ``hit`` and
                ``associated`` columns, indexed by peak timestamp.
            obs_peaks: Observed peak DataFrame with a ``peak_value`` column,
                indexed by peak timestamp.

        Returns:
            A 2-tuple ``(pred_slice, obs_values)`` where ``pred_slice`` is
            the stored prediction sliced to the hit timestamps, and
            ``obs_values`` is a Series of observed peak speeds at the
            associated dates.
        """
        hit_mask = pred_peaks.loc[:, "hit"].to_numpy()
        associated_dates = pred_peaks.loc[hit_mask, "associated"].to_numpy()
        pred_associated_peaks = self._prediction.slice(pred_peaks.index[hit_mask])
        obs_associated_peaks = obs_peaks.loc[associated_dates, "peak_value"]

        return pred_associated_peaks, obs_associated_peaks
