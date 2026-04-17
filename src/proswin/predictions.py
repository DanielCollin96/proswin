"""Core prediction container classes for solar wind forecasts.

Defines the abstract base :class:`SolarWindPrediction` and three concrete
subclasses — :class:`PointPrediction`, :class:`ProbabilisticPrediction`
(log-normal), and :class:`ProbabilisticNormalPrediction` (Gaussian) — that
are used throughout the evaluation, model, and utility subpackages.
"""
from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm
from scoringrules import crps_lognormal, crps_normal


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SolarWindPrediction(ABC):
    """Abstract base for all solar wind prediction containers.

    Subclasses store distribution parameters and expose a common interface
    for expected values, quantiles, CDFs, PDFs, CRPS, and log-likelihood.

    Args:
        params: Distribution parameters indexed by datetime. A ``pd.Series``
            is automatically promoted to a one-column ``pd.DataFrame``.

    Attributes:
        index: DatetimeIndex of prediction timestamps.
        data: Distribution parameter DataFrame (public alias for the internal
            parameter store).
    """

    def __init__(self, params: pd.Series | pd.DataFrame) -> None:
        if isinstance(params, pd.Series):
            params = params.to_frame()
        self._params: pd.DataFrame = params
        self.index: pd.Index = params.index
        self.data: pd.DataFrame = params

    # --- Slicing and combining ---------------------------------------------

    def slice(
        self, dates: pd.Index | slice | list[datetime.datetime]
    ) -> SolarWindPrediction:
        """Return a prediction subset restricted to ``dates``.

        Args:
            dates: Date selection passed directly to ``.loc``.

        Returns:
            Same subclass as the caller, containing only the selected dates.
        """
        return self.__class__(self._params.loc[dates])

    def union(self, other: SolarWindPrediction) -> SolarWindPrediction:
        """Concatenate two predictions and sort by index.

        Args:
            other: Must be the same subclass as the caller.

        Returns:
            Same subclass as the caller, with rows from both predictions
            merged and sorted by timestamp.
        """
        combined = pd.concat([self._params, other._params], axis=0).sort_index()
        return self.__class__(combined)

    # --- Abstract interface -------------------------------------------------

    @abstractmethod
    def get_expected_values(self) -> pd.Series:
        """Return the expected value of the predictive distribution.

        Returns:
            Expected values indexed by datetime.
        """

    @abstractmethod
    def get_quantiles(
        self, quantiles: float | list[float]
    ) -> pd.Series | pd.DataFrame:
        """Return predictive quantiles.

        Args:
            quantiles: Probability level(s) in (0, 1).

        Returns:
            A ``pd.Series`` for a single quantile, or a ``pd.DataFrame``
            with quantile levels as columns for multiple quantiles.
        """

    @abstractmethod
    def get_cdfs(self, x: float | np.ndarray) -> pd.Series:
        """Return the CDF evaluated at ``x``.

        Args:
            x: Evaluation point(s).

        Returns:
            CDF values as a Series.
        """

    @abstractmethod
    def get_pdfs(self, x: float | np.ndarray) -> pd.Series:
        """Return the PDF evaluated at ``x``.

        Args:
            x: Evaluation point(s).

        Returns:
            PDF values as a Series.
        """

    @abstractmethod
    def get_crps(self, obs: pd.Series) -> pd.Series:
        """Return the per-sample CRPS against observations.

        Args:
            obs: Observed values aligned with the prediction index.

        Returns:
            Per-sample CRPS values as a Series.
        """

    @abstractmethod
    def get_loglik(self, obs: pd.Series) -> float:
        """Return the mean negative log-likelihood against observations.

        Args:
            obs: Observed values aligned with the prediction index.

        Returns:
            Mean negative log-likelihood (scalar).
        """

    def to_frame(self) -> pd.DataFrame:
        """Return the distribution parameters as a DataFrame.

        Returns:
            A copy of the underlying parameter DataFrame (e.g. ``loc``/
            ``scale`` for Normal, ``shape``/``scale``/``loc`` for log-normal,
            or the point forecast mean), indexed by datetime.
        """
        return self._params.copy()


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class PointPrediction(SolarWindPrediction):
    """Container for deterministic (point) solar wind forecasts.

    Args:
        pred: Point predictions indexed by datetime. If a DataFrame is
            passed, the first column is used as the mean.

    Attributes:
        mean: The point forecast values as a Series.
    """

    def __init__(self, pred: pd.Series | pd.DataFrame) -> None:
        super().__init__(pred)
        # Store as a Series regardless of input shape for uniform access
        self.mean: pd.Series = (
            pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred.squeeze()
        )

    def get_expected_values(self) -> pd.Series:
        """Return point predictions as the expected value.

        Returns:
            The point forecast Series.
        """
        return self.mean

    def get_quantiles(
        self, quantiles: float | list[float]
    ) -> pd.Series | pd.DataFrame:
        """Not supported for point predictions.

        Raises:
            NotImplementedError: Always, since a point prediction has no
                predictive distribution.
        """
        raise NotImplementedError(
            "Quantiles are not defined for PointPrediction. "
            "Use a probabilistic prediction class instead."
        )

    def get_cdfs(self, x: float | np.ndarray) -> pd.Series:
        """Not supported for point predictions.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "CDF is not defined for PointPrediction."
        )

    def get_pdfs(self, x: float | np.ndarray) -> pd.Series:
        """Not supported for point predictions.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "PDF is not defined for PointPrediction."
        )

    def get_crps(self, obs: pd.Series) -> pd.Series:
        """Return NaN for all samples — CRPS requires a predictive distribution.

        Args:
            obs: Observed values (used only to determine the output index).

        Returns:
            Series of NaN values with the same index as ``obs``.
        """
        return pd.Series(np.nan, index=obs.index)

    def get_loglik(self, obs: pd.Series) -> float:
        """Not supported for point predictions.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Log-likelihood is not defined for PointPrediction."
        )


class ProbabilisticPrediction(SolarWindPrediction):
    """Container for log-normal probabilistic solar wind forecasts.

    Stores ``scipy.stats.lognorm`` parameters (shape, scale, loc) and
    exposes distribution methods for evaluation and scoring.

    Args:
        pred: DataFrame whose columns correspond to ``shape``, ``scale``,
            and ``loc``. If the columns are named exactly ``'shape'``,
            ``'scale'``, and ``'loc'``, they are accessed by name; otherwise
            the first three columns are used positionally.

    Attributes:
        shape: Log-normal shape parameter (s) — the standard deviation in
            log-space.
        scale: Log-normal scale parameter — ``exp(mu)`` where ``mu`` is the
            mean in log-space.
        loc: Additive location shift applied before the log transform.
    """

    def __init__(self, pred: pd.Series | pd.DataFrame) -> None:
        super().__init__(pred)
        try:
            # Preferred path: access columns by name when they are labelled
            self.shape: pd.Series = self._params['shape']
            self.scale: pd.Series = self._params['scale']
            self.loc: pd.Series = self._params['loc']
        except KeyError:
            # Fallback: assume shape, scale, loc are in the first 3 columns
            self.shape = self._params.iloc[:, 0]
            self.scale = self._params.iloc[:, 1]
            self.loc = self._params.iloc[:, 2]

    def get_expected_values(self) -> pd.Series:
        """Return the mean of the log-normal distribution.

        Returns:
            Expected values indexed by datetime.
        """
        return pd.Series(
            lognorm.mean(self.shape, loc=self.loc, scale=self.scale),
            index=self.index,
        )

    def get_quantiles(
        self, quantiles: float | list[float]
    ) -> pd.Series | pd.DataFrame:
        """Return log-normal predictive quantiles.

        Args:
            quantiles: Probability level(s) in (0, 1).

        Returns:
            A ``pd.Series`` for a single quantile, or a ``pd.DataFrame``
            with quantile levels as columns for multiple quantiles.
        """
        quantiles_arr = np.atleast_1d(np.asarray(quantiles))
        values = np.column_stack([
            lognorm.ppf(q, self.shape, loc=self.loc, scale=self.scale)
            for q in quantiles_arr
        ])
        result = pd.DataFrame(values, index=self.index, columns=quantiles_arr)
        return result.squeeze()

    def get_cdfs(self, x: float | np.ndarray) -> pd.Series:
        """Return the log-normal CDF evaluated at ``x``.

        Supports two calling modes:

        - **Scalar x, multiple time steps** (standard): evaluates the CDF of
          each per-timestep distribution at the single value ``x``.  Returns
          a Series indexed by datetime.
        - **Array x, single time step** (plotting): evaluates the CDF of the
          one stored distribution over the full array ``x``.  Returns a Series
          indexed by ``x``.

        Args:
            x: Scalar or 1-D array of evaluation point(s).

        Returns:
            CDF values as a Series.

        Raises:
            NotImplementedError: If ``x`` is array-valued and the prediction
                contains more than one time step.
        """
        if np.ndim(x) == 1:
            if len(self.index) == 1:
                val = lognorm.cdf(
                    x,
                    self.shape.squeeze(),
                    loc=self.loc.squeeze(),
                    scale=self.scale.squeeze(),
                )
                return pd.Series(val, index=x)

        if np.ndim(x) != 0:
            raise NotImplementedError(
                "Array-valued x is not supported for "
                "ProbabilisticPrediction.get_cdfs() if there is more than "
                "one predicted time point. Pass a scalar."
            )
        val = lognorm.cdf(x, self.shape, loc=self.loc, scale=self.scale)
        return pd.Series(val, index=self.index, name=x)

    def get_pdfs(self, x: float | np.ndarray) -> pd.Series:
        """Return the log-normal PDF evaluated at a scalar ``x``.

        Args:
            x: Scalar evaluation point.

        Returns:
            PDF values indexed by datetime.

        Raises:
            NotImplementedError: If ``x`` is array-valued.
        """
        if np.ndim(x) != 0:
            raise NotImplementedError(
                "Array-valued x is not supported for "
                "ProbabilisticPrediction.get_pdfs(). Pass a scalar."
            )
        val = lognorm.pdf(x, self.shape, loc=self.loc, scale=self.scale)
        return pd.Series(val, index=self.index, name=x)

    def get_crps(self, obs: pd.Series) -> pd.Series:
        """Return per-sample CRPS for the log-normal distribution.

        ``loc`` is subtracted from observations before scoring, consistent
        with the shifted log-normal parametrisation used by ``scoringrules``.

        Args:
            obs: Observed values aligned with the prediction index.

        Returns:
            Per-sample CRPS values as a Series.
        """
        obs_shifted = obs - self.loc.values

        crps = crps_lognormal(
            obs_shifted.values,
            mulog=np.log(self.scale.values),    # log-normal mu = log(scale)
            sigmalog=self.shape.values,          # log-normal sigma = shape
        )
        return pd.Series(crps, index=self.index)

    def get_loglik(self, obs: pd.Series) -> float:
        """Return the mean negative log-likelihood for the log-normal distribution.

        Args:
            obs: Observed values aligned with the prediction index.

        Returns:
            Mean negative log-likelihood (scalar).
        """
        prob = lognorm.pdf(obs, self.shape, loc=self.loc, scale=self.scale)
        return float(-np.mean(np.log(prob)))


class ProbabilisticNormalPrediction(SolarWindPrediction):
    """Container for Gaussian probabilistic solar wind forecasts.

    Args:
        pred: DataFrame whose first column is the Gaussian mean (``loc``)
            and second column is the standard deviation (``scale``), indexed
            by datetime.

    Attributes:
        loc: Gaussian mean (expected value).
        scale: Gaussian standard deviation.
    """

    def __init__(self, pred: pd.Series | pd.DataFrame) -> None:
        super().__init__(pred)
        self.loc: pd.Series = self._params.iloc[:, 0]
        self.scale: pd.Series = self._params.iloc[:, 1]

    def get_expected_values(self) -> pd.Series:
        """Return the Gaussian mean (loc parameter).

        Returns:
            Expected values indexed by datetime.
        """
        return self.loc

    def get_quantiles(
        self, quantiles: float | list[float]
    ) -> pd.Series | pd.DataFrame:
        """Return Gaussian predictive quantiles.

        Args:
            quantiles: Probability level(s) in (0, 1).

        Returns:
            A ``pd.Series`` for a single quantile, or a ``pd.DataFrame``
            with quantile levels as columns for multiple quantiles.
        """
        quantiles_arr = np.atleast_1d(np.asarray(quantiles))
        values = np.column_stack([
            norm.ppf(q, loc=self.loc, scale=self.scale)
            for q in quantiles_arr
        ])
        result = pd.DataFrame(values, index=self.index, columns=quantiles_arr)
        return result.squeeze()

    def get_cdfs(self, x: float | np.ndarray) -> pd.Series:
        """Return the Gaussian CDF evaluated at a scalar ``x``.

        Args:
            x: Scalar evaluation point.

        Returns:
            CDF values indexed by datetime.

        Raises:
            NotImplementedError: If ``x`` is array-valued.
        """
        if np.ndim(x) != 0:
            raise NotImplementedError(
                "Array-valued x is not supported for "
                "ProbabilisticNormalPrediction.get_cdfs(). Pass a scalar."
            )
        val = norm.cdf(x, loc=self.loc, scale=self.scale)
        return pd.Series(val, index=self.index, name=x)

    def get_pdfs(self, x: float | np.ndarray) -> pd.Series:
        """Return the Gaussian PDF evaluated at a scalar ``x``.

        Args:
            x: Scalar evaluation point.

        Returns:
            PDF values indexed by datetime.

        Raises:
            NotImplementedError: If ``x`` is array-valued.
        """
        if np.ndim(x) != 0:
            raise NotImplementedError(
                "Array-valued x is not supported for "
                "ProbabilisticNormalPrediction.get_pdfs(). Pass a scalar."
            )
        val = norm.pdf(x, loc=self.loc, scale=self.scale)
        return pd.Series(val, index=self.index, name=x)

    def get_crps(self, obs: pd.Series) -> pd.Series:
        """Return per-sample CRPS for the Gaussian distribution.

        Args:
            obs: Observed values aligned with the prediction index.

        Returns:
            Per-sample CRPS values as a Series.
        """
        crps = crps_normal(obs, self.loc.values, self.scale.values)
        return pd.Series(crps, index=self.index)

    def get_loglik(self, obs: pd.Series) -> float:
        """Return the mean negative log-likelihood for the Gaussian distribution.

        Args:
            obs: Observed values aligned with the prediction index.

        Returns:
            Mean negative log-likelihood (scalar).
        """
        prob = norm.pdf(obs, loc=self.loc, scale=self.scale)
        return float(-np.mean(np.log(prob)))
