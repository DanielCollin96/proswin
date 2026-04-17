from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from proswin.predictions import SolarWindPrediction

_FREQ = "60min"  # Fixed expected frequency of the time series


@dataclass(frozen=True)
class GapAnalysisResult:
    """Container for the output of :func:`get_interpolation_dates`.

    Attributes:
        missing_dates: DatetimeIndex of all timestamps that are absent
            from ``data_index`` but fall within a gap smaller than
            ``gap_len_threshold``. These are the dates that should be
            interpolated.
        gap_lengths: List of ``pd.Timedelta`` values, one per detected
            small gap, representing the span between the first and last
            missing timestamp in that gap. Use this to compute
            descriptive statistics (count, mean, median, min, max, …).
    """

    missing_dates: pd.DatetimeIndex
    gap_lengths: list[pd.Timedelta] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience statistics (computed on demand, not stored)
    # ------------------------------------------------------------------

    @property
    def n_gaps(self) -> int:
        """Number of small gaps found."""
        return len(self.gap_lengths)

    @property
    def total_missing(self) -> int:
        """Total number of missing timestamps across all small gaps."""
        return len(self.missing_dates)

    def summary(self) -> pd.Series:
        """Descriptive statistics for gap lengths.

        Returns a ``pd.Series`` with standard descriptive statistics
        (count, mean, std, min, quartiles, max) plus, for each threshold
        in ``[1, 2, 3, 4]`` days, the number of gaps at or above that
        threshold and their mean length.

        Returns:
            Series of gap-length statistics, or an empty Series when no
            gaps were found.
        """
        if not self.gap_lengths:
            return pd.Series(dtype="timedelta64[ns]", name="gap_lengths")

        s = pd.Series(self.gap_lengths, name="gap_lengths")
        stats = s.describe()

        for days in [1, 2, 3, 4]:
            threshold = pd.Timedelta(days=days)
            large = s[s >= threshold]
            stats[f"n_gaps_>={days}d"] = len(large)
            stats[f"mean_gap_>={days}d"] = large.mean() if not large.empty else pd.NaT

        return stats


def get_interpolation_dates(
    data_index: pd.DatetimeIndex,
    gap_len_threshold: pd.Timedelta = pd.Timedelta(days=5),
) -> GapAnalysisResult:
    """Return missing hourly timestamps and gap statistics for ``data_index``.

    Large gaps (duration strictly greater than ``gap_len_threshold``) are
    ignored — they are assumed to represent deliberate data absences such
    as sensor outages that should not be interpolated. For every small gap,
    the function collects the timestamps missing relative to a continuous
    60-minute grid and records the gap span for downstream statistics.

    The algorithm works in three steps:

    1. Identify positions of large gaps and split the index into
       contiguous chunks at those positions.
    2. For each chunk, compute the expected hourly grid and find missing
       timestamps.
    3. Within the missing timestamps of each chunk, identify individual
       gap runs and record their span (last missing − first missing
       timestamp). A gap containing a single missing hour therefore has a
       recorded span of zero.

    Args:
        data_index: Monotonically increasing ``DatetimeIndex`` of observed
            timestamps. Must contain at least two elements.
        gap_len_threshold: Gaps whose duration is strictly greater than
            this value are treated as large gaps and excluded from
            interpolation. Defaults to ``pd.Timedelta(days=5)``.

    Returns:
        A frozen :class:`GapAnalysisResult` with ``missing_dates`` (the
        timestamps to interpolate) and ``gap_lengths`` (one
        ``pd.Timedelta`` per small gap).

    Raises:
        TypeError: If ``data_index`` is not a ``pd.DatetimeIndex`` or
            ``gap_len_threshold`` is not a ``pd.Timedelta``.
        ValueError: If ``data_index`` has fewer than two elements or is
            not sorted in ascending order.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(data_index, pd.DatetimeIndex):
        raise TypeError(
            f"data_index must be a pd.DatetimeIndex, "
            f"got {type(data_index).__name__}."
        )
    if not isinstance(gap_len_threshold, pd.Timedelta):
        raise TypeError(
            f"gap_len_threshold must be a pd.Timedelta, "
            f"got {type(gap_len_threshold).__name__}."
        )
    if len(data_index) < 2:
        raise ValueError("data_index must contain at least two timestamps.")
    if not data_index.is_monotonic_increasing:
        raise ValueError("data_index must be sorted in ascending order.")

    # ------------------------------------------------------------------
    # Identify where large gaps begin
    # ------------------------------------------------------------------
    diffs = data_index.to_series().diff()  # first value is NaT → always False below
    large_gap_mask = diffs > gap_len_threshold
    large_gap_positions = np.flatnonzero(large_gap_mask)  # array of integer positions

    # Split the index into contiguous chunks separated by large gaps.
    # np.split on position p gives [..., p-1] and [p, ...] — exactly what we want.
    chunks = np.split(data_index, large_gap_positions)

    # ------------------------------------------------------------------
    # Collect missing timestamps and gap lengths from each chunk
    # ------------------------------------------------------------------
    all_missing: list[pd.DatetimeIndex] = []
    gap_lengths: list[pd.Timedelta] = []

    for chunk in chunks:
        if len(chunk) < 2:
            # A single-element chunk has no internal gaps to check.
            continue

        expected = pd.date_range(chunk[0], chunk[-1], freq=_FREQ)
        missing_in_chunk = expected.difference(chunk)
        if missing_in_chunk.empty:
            continue

        all_missing.append(missing_in_chunk)

        # Each contiguous run of missing timestamps is one gap. Identify
        # run boundaries by finding where consecutive missing timestamps
        # are more than one hour apart.
        diff_missing = missing_in_chunk.diff()
        gap_mask = diff_missing > pd.Timedelta(hours=1)
        gap_starts = np.concatenate([np.array([0]), np.flatnonzero(gap_mask)])

        for k, start in enumerate(gap_starts):
            end = (
                gap_starts[k + 1] - 1
                if k + 1 < len(gap_starts)
                else len(missing_in_chunk) - 1
            )
            # Gap span = last missing timestamp − first missing timestamp.
            # A gap with a single missing hour has a span of zero by this
            # definition (the actual data hole is 2 hours wide, but we
            # record the missing-timestamp span for consistency).
            gap_duration = missing_in_chunk[end] - missing_in_chunk[start]
            gap_lengths.append(gap_duration)

    missing_dates = (
        pd.DatetimeIndex(np.concatenate(all_missing))
        if all_missing
        else pd.DatetimeIndex([])
    )

    return GapAnalysisResult(missing_dates=missing_dates, gap_lengths=gap_lengths)


def interpolate_predictions(
    pred: SolarWindPrediction | pd.Series | pd.DataFrame,
    gap_len_threshold: pd.Timedelta = pd.Timedelta(days=5),
) -> SolarWindPrediction | pd.Series | pd.DataFrame:
    """Fill small temporal gaps in a prediction object via linear interpolation.

    Gaps strictly larger than ``gap_len_threshold`` are ignored (assumed to
    be deliberate data absences). All other missing hourly timestamps are
    filled with linearly interpolated values.

    The original type is preserved: a ``SolarWindPrediction`` subclass is
    reconstructed via its own constructor, a ``pd.Series`` is returned as a
    Series, and a ``pd.DataFrame`` is returned as a DataFrame.

    Args:
        pred: A prediction object of any supported type:
            ``SolarWindPrediction`` subclass (``ProbabilisticPrediction``,
            ``ProbabilisticNormalPrediction``, ``PointPrediction``),
            ``pd.Series``, or ``pd.DataFrame``.
        gap_len_threshold: Gaps whose duration is strictly greater than
            this value are not interpolated. Defaults to
            ``pd.Timedelta(days=5)``.

    Returns:
        Same type as the input, with small gaps filled by linear
        interpolation.

    Raises:
        TypeError: If ``pred`` is not a supported type.
    """
    # ------------------------------------------------------------------
    # Extract a DataFrame of columns to interpolate
    # ------------------------------------------------------------------
    if isinstance(pred, (SolarWindPrediction, pd.Series)):
        pred_cols = pred.to_frame()
    elif isinstance(pred, pd.DataFrame):
        pred_cols = pred.copy()
    else:
        raise TypeError(
            f"Unsupported prediction type: {type(pred).__name__}. "
            "Expected a SolarWindPrediction subclass, pd.Series, or pd.DataFrame."
        )

    # Detect gaps and insert NaN rows at missing timestamps
    gap_result = get_interpolation_dates(
        pred_cols.index, gap_len_threshold=gap_len_threshold
    )
    gaps_to_fill = gap_result.missing_dates

    if gaps_to_fill.empty:
        return pred  # nothing to do — return early, preserving the original object

    nan_frame = pd.DataFrame(
        np.nan,
        index=gaps_to_fill,
        columns=pred_cols.columns,
    )
    pred_cols = (
        pd.concat([pred_cols, nan_frame], axis=0)
        .sort_index()
        .interpolate(method="linear", axis=0)
    )

    # Reconstruct the original type from the interpolated DataFrame
    if isinstance(pred, SolarWindPrediction):
        # __class__ preserves the exact subclass without a long if/elif chain
        return pred.__class__(pred_cols)
    elif isinstance(pred, pd.Series):
        return pred_cols.squeeze()
    else:
        return pred_cols
