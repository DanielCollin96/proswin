"""HSS (High-Speed Stream) peak finder for solar wind speed time series."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from proswin.predictions import SolarWindPrediction


class HSSFinder:
    """Find and associate predicted HSS peaks with observed HSS peaks.

    Loads precomputed event lists (enhancements, CMEs, HSSs) from CSV files.
    The main entry point is :meth:`find`, which accepts a predicted time
    series, interpolates any gaps to ensure hourly continuity, finds predicted
    peaks, and associates them with observed peaks.

    Can be used standalone or via :class:`PredictionEvaluator`.

    Args:
        path_to_events: Directory containing ``cme_list.csv``,
            ``hss_list.csv``, and ``enhancement_list.csv``.

    Attributes:
        cme_list: DataFrame of coronal mass ejection events.
        hss_list: DataFrame of high-speed stream events.
        enhancement_list: DataFrame of solar wind speed enhancement events
            used as the reference observation catalogue.
    """

    # Peak detection hyperparameters — centralised for transparency
    _SMOOTH_WINDOW: int = 24
    _PEAK_DISTANCE: int = 4 * 24
    _THRESHOLD_SPEED: float = 390.0
    _PROMINENCE: float = 35.0
    _ASSOCIATION_WINDOW_DAYS: int = 3

    def __init__(self, path_to_events: Path) -> None:
        parse_cols = [1, 3, 4, 5]
        self.cme_list: pd.DataFrame = pd.read_csv(
            path_to_events / "cme_list.csv", index_col=0, parse_dates=parse_cols
        )
        self.hss_list: pd.DataFrame = pd.read_csv(
            path_to_events / "hss_list.csv", index_col=0, parse_dates=parse_cols
        )
        self.enhancement_list: pd.DataFrame = pd.read_csv(
            path_to_events / "enhancement_list.csv",
            index_col=0,
            parse_dates=parse_cols,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def find(
        self,
        predictions: pd.Series | SolarWindPrediction,
        interpolation_dates: pd.DatetimeIndex | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Find and associate predicted HSS peaks with observed HSS peaks.

        Accepts either a plain ``pd.Series`` or any ``SolarWindPrediction``
        subclass. For probabilistic predictions, the expected value series is
        used for peak detection.

        Gaps are filled by time-interpolation before peak detection so that
        the algorithm operates on a continuous hourly series. If
        ``interpolation_dates`` is not provided, gaps are detected
        automatically from the index.

        Args:
            predictions: Predicted solar wind speed. If a
                ``SolarWindPrediction`` is passed,
                ``get_expected_values()`` is called internally.
            interpolation_dates: Explicit set of missing datetimes to fill.
                If None, gaps are inferred automatically from the index.

        Returns:
            A 2-tuple ``(pred_peaks, obs_peaks)``:

            - ``pred_peaks``: DataFrame of predicted enhancement peaks with
              association columns (``associated``, ``hit``, ``false_alarm``).
            - ``obs_peaks``: DataFrame of observed enhancement peaks with
              association columns (``associated``, ``hit``, ``miss``).
        """
        # Unwrap SolarWindPrediction to a plain expected-value Series
        if not isinstance(predictions, pd.Series):
            predictions = predictions.get_expected_values()

        predictions = self._interpolate_gaps(predictions, interpolation_dates)
        return self._compute_predicted_event_list(predictions)

    def _interpolate_gaps(
        self,
        predictions: pd.Series,
        interpolation_dates: pd.DatetimeIndex | None,
    ) -> pd.Series:
        """Fill gaps in the prediction series with time-interpolated values.

        Args:
            predictions: Raw prediction time series, possibly with gaps.
            interpolation_dates: Explicit gap dates. If None, inferred from
                the index.

        Returns:
            Gap-filled, sorted time series.
        """
        if interpolation_dates is None:
            # Reconstruct the expected hourly index and find missing entries
            full_index = pd.date_range(
                start=predictions.index[0],
                end=predictions.index[-1],
                freq="1h",
            )
            interpolation_dates = full_index.difference(predictions.index)

        if len(interpolation_dates) == 0:
            return predictions.sort_index()

        gap_series = pd.Series(np.nan, index=interpolation_dates)
        return (
            pd.concat([predictions, gap_series], axis=0)
            .sort_index()
            .interpolate(method="time", limit_direction="both")
        )

    # ------------------------------------------------------------------
    # Private: chunk-level peak finding and association
    # ------------------------------------------------------------------

    def _compute_predicted_event_list(
        self, pred_ts: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the time series into continuous chunks and process each.

        Args:
            pred_ts: Gap-filled, continuous predicted time series.

        Returns:
            Concatenated predicted and observed enhancement DataFrames.
        """
        chunk_bounds = self._get_chunk_bounds(pred_ts)

        collect_pred, collect_obs = [], []
        for start, end in chunk_bounds:
            pred_chunk = pred_ts.loc[start:end]

            # Restrict observed enhancements to the current chunk's time range
            in_chunk = (
                (self.enhancement_list["start"] >= start)
                & (self.enhancement_list["end"] <= end)
            )
            obs_chunk = self.enhancement_list.loc[in_chunk].reset_index(drop=True)

            pred_enh = self._find_peaks_in_time_series(
                pred_chunk, exclude_unpredictable_cme_peaks=True
            )

            matching_pred, matching_obs = self._associate_pred_obs_peaks(
                pred_enh["smoothed_peak_date"].to_numpy(),
                obs_chunk["smoothed_peak_date"].to_numpy(),
            )

            pred_enh = pd.concat([pred_enh, matching_pred], axis=1)
            obs_chunk = pd.concat([obs_chunk, matching_obs], axis=1)

            pred_enh, obs_chunk = self._assign_actual_peaks(pred_enh, obs_chunk)

            # Use peak date as index for alignment with the observation series
            for df in (pred_enh, obs_chunk):
                df.index = df["peak_date"]
                df.drop(columns=["peak_date"], inplace=True)

            collect_pred.append(pred_enh)
            collect_obs.append(obs_chunk)

        pred_out = pd.concat(collect_pred, axis=0).sort_index()
        obs_out = pd.concat(collect_obs, axis=0).sort_index()

        return pred_out, obs_out

    @staticmethod
    def _get_chunk_bounds(
        pred_ts: pd.Series,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Return ``(start, end)`` pairs for each continuous chunk in the series.

        A new chunk begins wherever the gap between consecutive timestamps
        exceeds one hour (i.e. at a genuine data outage boundary).

        Args:
            pred_ts: Continuous (post-interpolation) predicted time series.

        Returns:
            List of ``(start, end)`` timestamp pairs, one per continuous chunk.
        """
        # A new chunk starts wherever the time gap exceeds one hour
        gap_mask = pred_ts.index.to_series().diff() > timedelta(hours=1)
        starts = [pred_ts.index[0]] + list(pred_ts.index[gap_mask.values])
        ends = list(pred_ts.index[gap_mask.shift(-1, fill_value=False).values]) + [
            pred_ts.index[-1]
        ]
        return list(zip(starts, ends))

    # ------------------------------------------------------------------
    # Private: peak detection
    # ------------------------------------------------------------------

    def _find_peaks_in_time_series(
        self,
        time_series: pd.Series,
        exclude_unpredictable_cme_peaks: bool = False,
    ) -> pd.DataFrame:
        """Detect HSS peaks and enhancement intervals in a time series.

        Applies Gaussian smoothing before peak detection to reduce noise, then
        uses ``scipy.signal.find_peaks`` with the class-level hyperparameters.
        Enhancement intervals (start/end of each event) are derived from the
        half-width of each smoothed peak at 60% relative height.

        Args:
            time_series: Predicted or observed solar wind speed (continuous,
                hourly).
            exclude_unpredictable_cme_peaks: If True, peaks overlapping CMEs
                or occurring one solar rotation (~27 days) after a CME are
                flagged in a ``cme_flag`` column.

        Returns:
            DataFrame with columns ``peak_date``, ``peak_value``, ``start``,
            ``end``, ``smoothed_peak_date``, and optionally ``cme_flag``.
        """
        ts_smoothed = pd.Series(
            gaussian_filter1d(time_series, self._SMOOTH_WINDOW),
            index=time_series.index,
        )

        peak_idx, _ = find_peaks(
            ts_smoothed,
            height=self._THRESHOLD_SPEED,
            distance=self._PEAK_DISTANCE,
            prominence=self._PROMINENCE,
        )
        smoothed_peaks = pd.to_datetime(pd.Series(time_series.index[peak_idx]))

        enhancement_intervals = self._compute_enhancement_intervals(
            ts_smoothed, peak_idx
        )

        peak_dates, peak_values, cme_flags = self._extract_peak_metadata(
            time_series, enhancement_intervals, exclude_unpredictable_cme_peaks
        )

        enhancement_list = pd.DataFrame({
            "peak_date": pd.to_datetime(pd.Series(peak_dates)),
            "peak_value": pd.Series(peak_values, dtype=float),
            "start": pd.to_datetime(
                enhancement_intervals[:, 0] if len(enhancement_intervals) else []
            ),
            "end": pd.to_datetime(
                enhancement_intervals[:, 1] if len(enhancement_intervals) else []
            ),
            "smoothed_peak_date": smoothed_peaks,
        })

        if exclude_unpredictable_cme_peaks:
            enhancement_list["cme_flag"] = cme_flags

        return enhancement_list

    @staticmethod
    def _compute_enhancement_intervals(
        ts_smoothed: pd.Series,
        peak_idx: np.ndarray,
    ) -> np.ndarray:
        """Compute start/end dates of each enhancement interval.

        Uses ``peak_widths`` at 60% relative height, then resolves any
        overlapping intervals by truncating the longer one.

        Args:
            ts_smoothed: Smoothed time series.
            peak_idx: Integer indices of detected peaks.

        Returns:
            Array of shape ``(n_peaks, 2)`` containing ``[start_date,
            end_date]`` pairs as timestamps. Returns an empty ``(0, 2)``
            array if no peaks were found.
        """
        if len(peak_idx) == 0:
            return np.empty((0, 2), dtype=object)

        widths = peak_widths(ts_smoothed.values, peak_idx, rel_height=0.6)
        intervals = np.column_stack([
            np.floor(widths[2]).astype(int),
            np.ceil(widths[3]).astype(int),
        ])

        # Resolve overlapping intervals by truncating the longer neighbour
        for i in range(len(intervals) - 1):
            if intervals[i, 1] > intervals[i + 1, 0]:
                len_left = intervals[i, 1] - intervals[i, 0]
                len_right = intervals[i + 1, 1] - intervals[i + 1, 0]
                if len_left > len_right:
                    intervals[i, 1] = intervals[i + 1, 0] - 1
                else:
                    intervals[i + 1, 0] = intervals[i, 1] + 1

        # Convert integer indices to timestamps
        dates = np.empty_like(intervals, dtype=object)
        dates[:, 0] = ts_smoothed.index[intervals[:, 0]]
        dates[:, 1] = ts_smoothed.index[intervals[:, 1]]
        return dates

    def _extract_peak_metadata(
        self,
        time_series: pd.Series,
        enhancement_intervals: np.ndarray,
        exclude_cme: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract peak date, value, and CME flag for each enhancement interval.

        Args:
            time_series: Original (unsmoothed) time series.
            enhancement_intervals: Array of shape ``(n, 2)`` containing
                ``[start_date, end_date]`` pairs.
            exclude_cme: Whether to compute CME flags.

        Returns:
            A 3-tuple ``(peak_dates, peak_values, cme_flags)``:

            - ``peak_dates``: object array of peak timestamps.
            - ``peak_values``: float array of peak wind speeds.
            - ``cme_flags``: bool array; True where the peak is CME-influenced.
        """
        n = len(enhancement_intervals)
        peak_dates = np.empty(n, dtype=object)
        peak_values = np.zeros(n, dtype=float)
        cme_flags = np.zeros(n, dtype=bool)

        for i, (start, end) in enumerate(enhancement_intervals):
            interval = time_series.loc[start:end]
            peak_values[i] = interval.max()
            peak_dates[i] = interval.index[interval.argmax()]

            if exclude_cme:
                cme_flags[i] = self._is_cme_influenced(peak_dates[i])

        return peak_dates, peak_values, cme_flags

    def _is_cme_influenced(self, peak_date: pd.Timestamp) -> bool:
        """Return True if a peak overlaps with or echoes a CME event.

        Two conditions are checked:

        - **Direct overlap**: a CME was active during the peak timestamp.
        - **Carrington echo**: a CME occurred approximately one solar rotation
          (~27 days) before the peak, making the peak a recurrence of
          CME-driven structure rather than a genuine HSS.

        Args:
            peak_date: Timestamp of the detected peak.

        Returns:
            True if the peak is CME-influenced by either condition.
        """
        # Direct CME overlap
        direct = (
            (self.cme_list["start"] <= peak_date)
            & (self.cme_list["end"] >= peak_date)
        )
        # One solar rotation echo (~27 days prior)
        echo = (
            (self.cme_list["start"] <= peak_date - timedelta(days=26))
            & (self.cme_list["end"] >= peak_date - timedelta(days=28))
        )
        return bool((direct | echo).any())

    # ------------------------------------------------------------------
    # Private: peak association
    # ------------------------------------------------------------------

    def _associate_pred_obs_peaks(
        self,
        pred_peaks: np.ndarray,
        obs_peaks: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Associate predicted and observed peaks by temporal proximity.

        Uses a mutual nearest-neighbour algorithm: a predicted and observed
        peak are only linked as a hit if each is the closest available
        candidate to the other within ``_ASSOCIATION_WINDOW_DAYS``.  The
        algorithm iterates until no new associations are formed.

        Args:
            pred_peaks: Predicted smoothed peak dates.
            obs_peaks: Observed smoothed peak dates.

        Returns:
            A 2-tuple ``(matching_pred, matching_obs)``:

            - ``matching_pred``: DataFrame with columns ``associated``,
              ``hit``, ``false_alarm``.
            - ``matching_obs``: DataFrame with columns ``associated``,
              ``hit``, ``miss``.
        """
        window = timedelta(days=self._ASSOCIATION_WINDOW_DAYS)

        obs_associated = np.empty(len(obs_peaks), dtype=object)
        pred_associated = np.empty(len(pred_peaks), dtype=object)
        hit_pred = np.zeros(len(pred_peaks), dtype=bool)
        hit_obs = np.zeros(len(obs_peaks), dtype=bool)

        new_associations = 1
        while new_associations > 0:
            new_associations = 0

            # For each unmatched predicted peak, find closest available obs peak
            for i, pp in enumerate(pred_peaks):
                if hit_pred[i]:
                    continue
                candidates = obs_peaks[
                    ~hit_obs
                    & (obs_peaks >= pd.Timestamp(pp) - window)
                    & (obs_peaks <= pd.Timestamp(pp) + window)
                ]
                pred_associated[i] = (
                    candidates[np.argmin(np.abs(candidates - pp))]
                    if len(candidates) else None
                )

            # For each unmatched obs peak, find closest available pred peak
            for i, op in enumerate(obs_peaks):
                if hit_obs[i]:
                    continue
                candidates = pred_peaks[
                    ~hit_pred
                    & (pred_peaks >= pd.Timestamp(op) - window)
                    & (pred_peaks <= pd.Timestamp(op) + window)
                ]
                obs_associated[i] = (
                    candidates[np.argmin(np.abs(candidates - op))]
                    if len(candidates) else None
                )

            # Confirm mutual nearest-neighbour pairs as hits
            for i, op in enumerate(obs_peaks):
                if hit_obs[i] or obs_associated[i] is None:
                    continue
                j = np.where(pred_peaks == obs_associated[i])[0][0]
                if pred_associated[j] == op:
                    hit_obs[i] = True
                    hit_pred[j] = True
                    new_associations += 1

        matching_pred = pd.DataFrame({
            "associated": pd.to_datetime(pd.Series(pred_associated)),
            "hit": pd.Series(hit_pred, dtype=bool),
            "false_alarm": pd.Series(~hit_pred, dtype=bool),
        })
        matching_obs = pd.DataFrame({
            "associated": pd.to_datetime(pd.Series(obs_associated)),
            "hit": pd.Series(hit_obs, dtype=bool),
            "miss": pd.Series(~hit_obs, dtype=bool),
        })
        return matching_pred, matching_obs

    @staticmethod
    def _assign_actual_peaks(
        pred_enhancements: pd.DataFrame,
        obs_enhancements: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Resolve smoothed associations to actual peak dates and remove CME peaks.

        After the mutual nearest-neighbour association step, the ``associated``
        columns contain smoothed peak dates. This method:

        1. Replaces smoothed dates with actual (unsmoothed) peak dates in both
           ``associated`` columns.
        2. Removes predicted peaks whose associated observed peak carries a
           CME flag (the event is CME-driven, not an HSS).
        3. Removes CME-flagged predicted peaks that have no association.
        4. Removes all CME-flagged observed peaks from the output catalogue.

        Args:
            pred_enhancements: DataFrame of predicted enhancement peaks,
                including ``associated``, ``peak_date``, and ``cme_flag``
                columns.
            obs_enhancements: DataFrame of observed enhancement peaks,
                including ``associated``, ``peak_date``, and ``cme_flag``
                columns.

        Returns:
            Filtered ``(pred_enhancements, obs_enhancements)`` DataFrames with
            CME-influenced peaks removed and indices reset.
        """
        not_cme_obs = ~obs_enhancements["cme_flag"]
        delete_pred_idx = []

        for i in range(len(pred_enhancements)):
            assoc_smoothed = pred_enhancements.at[i, "associated"]

            if not pd.isnull(assoc_smoothed):
                # Resolve smoothed → actual peak date for the associated obs peak
                obs_match = obs_enhancements.index[
                    obs_enhancements["smoothed_peak_date"] == assoc_smoothed
                ]
                if len(obs_match) == 0:
                    continue
                obs_idx = obs_match[0]
                actual_obs_peak = obs_enhancements.at[obs_idx, "peak_date"]

                # Overwrite smoothed associations with actual peak dates
                pred_enhancements.at[i, "associated"] = actual_obs_peak
                obs_enhancements.at[obs_idx, "associated"] = (
                    pred_enhancements.at[i, "peak_date"]
                )

                if obs_enhancements.at[obs_idx, "cme_flag"]:
                    delete_pred_idx.append(i)
            else:
                if pred_enhancements.at[i, "cme_flag"]:
                    delete_pred_idx.append(i)

        pred_enhancements = pred_enhancements.drop(index=delete_pred_idx)
        obs_enhancements = obs_enhancements.loc[not_cme_obs]

        pred_enhancements = pred_enhancements.reset_index(drop=True)
        obs_enhancements = obs_enhancements.reset_index(drop=True)

        return pred_enhancements, obs_enhancements
