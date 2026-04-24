"""Precompute observed solar wind enhancement, HSS, and CME event catalogues.
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from proswin.evaluation.hss_finder import HSSFinder


class EventCatalogueBuilder:
    """Build observed HSS and CME event catalogues from a solar wind speed series.

    Detects solar wind speed enhancements in the observed time series using
    the same peak-finding methodology as :class:`HSSFinder`, then classifies
    each enhancement as a CME or HSS based on its proximity to entries in the
    ICME list. CME entries from the ICME list that did not coincide with any
    detected enhancement are also appended to the CME catalogue (expanded by
    24 h before and 48 h after).

    Args:
        icme_list: DataFrame of observed ICMEs with columns ``'start'`` and
            ``'end'`` as datetime, indexed arbitrarily.
        solar_wind_speed: Observed hourly solar wind speed series used to
            detect enhancements.
        add_all_icme: If ``True`` (default), ICME entries from ``icme_list``
            that did not overlap any detected enhancement are added to the CME
            catalogue, expanded by 24 h before and 48 h after their listed
            times.

    Attributes:
        enhancement_list: DataFrame of all detected enhancements with a
            ``cme_flag`` column. Available after calling :meth:`build`.
        cme_list: DataFrame of CME events. Available after calling
            :meth:`build`.
        hss_list: DataFrame of HSS events. Available after calling
            :meth:`build`.
    """

    # Lead window used to associate an ICME with an enhancement
    _ICME_LEAD_HOURS: int = 48

    # Padding added to unmatched ICME entries when add_all_icme is True
    _ICME_PAD_BEFORE_HOURS: int = 24
    _ICME_PAD_AFTER_HOURS: int = 48

    def __init__(
        self,
        icme_list: pd.DataFrame,
        solar_wind_speed: pd.Series,
        add_all_icme: bool = True,
    ) -> None:
        self._icme_list = icme_list.copy().reset_index(drop=True)
        self._solar_wind_speed = solar_wind_speed
        self._add_all_icme = add_all_icme

        self.enhancement_list: pd.DataFrame | None = None
        self.cme_list: pd.DataFrame | None = None
        self.hss_list: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Detect enhancements and classify them as CMEs or HSSs.

        Populates :attr:`enhancement_list`, :attr:`cme_list`, and
        :attr:`hss_list`. Must be called before :meth:`save`.
        """
        # Restrict ICME list to the time interval of the solar wind series
        in_range = (
            (self._solar_wind_speed.index[0] <= self._icme_list["end"])
            & (self._solar_wind_speed.index[-1] >= self._icme_list["start"])
        )
        icme_list = self._icme_list.loc[in_range].reset_index(drop=True)

        # Use HSSFinder's peak detection on the observed series
        hss_finder = HSSFinder.__new__(HSSFinder)
        enhancement_list = hss_finder._find_peaks_in_time_series(
            self._solar_wind_speed, exclude_unpredictable_cme_peaks=False
        )

        self.enhancement_list, self.cme_list, self.hss_list = (
            self._assign_cme_hss(enhancement_list, icme_list)
        )

    def save(self, output_dir: Path) -> None:
        """Write the three catalogues to CSV files in ``output_dir``.

        Args:
            output_dir: Directory in which to write ``enhancement_list.csv``,
                ``cme_list.csv``, and ``hss_list.csv``. Created if it does
                not exist.

        Raises:
            RuntimeError: If :meth:`build` has not been called yet.
        """
        if self.enhancement_list is None:
            raise RuntimeError("Call build() before save().")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.enhancement_list.to_csv(output_dir / "enhancement_list.csv")
        self.cme_list.to_csv(output_dir / "cme_list.csv")
        self.hss_list.to_csv(output_dir / "hss_list.csv")

    # ------------------------------------------------------------------
    # Private: CME / HSS classification
    # ------------------------------------------------------------------

    def _assign_cme_hss(
        self,
        enhancement_list: pd.DataFrame,
        icme_list: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Classify enhancements as CMEs or HSSs and build event catalogues.

        An enhancement is labelled as a CME if any ICME in ``icme_list``
        overlaps it or starts within :attr:`_ICME_LEAD_HOURS` before it ends.
        For matched CME enhancements the enhancement interval is expanded to
        cover the full ICME interval, and adjacent enhancement boundaries are
        adjusted to prevent overlap.

        Unmatched ICME entries are appended to the CME catalogue (expanded by
        :attr:`_ICME_PAD_BEFORE_HOURS` / :attr:`_ICME_PAD_AFTER_HOURS`) when
        ``add_all_icme`` is ``True``.

        Args:
            enhancement_list: DataFrame of detected enhancements with at
                least ``'start'`` and ``'end'`` columns.
            icme_list: ICME catalogue restricted to the solar wind series
                time range.

        Returns:
            A 3-tuple ``(enhancement_list, cme_list, hss_list)`` where
            ``enhancement_list`` gains a ``'cme_flag'`` boolean column.
        """
        icme_list = icme_list.copy().reset_index(drop=True)
        cme_indices: list[int] = []
        hss_indices: list[int] = []
        cme_flags: list[bool] = []

        for j in range(len(enhancement_list)):
            enh_start = enhancement_list.at[j, "start"]
            enh_end = enhancement_list.at[j, "end"]

            # Match ICMEs that overlap or start within the lead window
            icme_match = (
                (enh_start - timedelta(hours=self._ICME_LEAD_HOURS) <= icme_list["end"])
                & (enh_end >= icme_list["start"])
            ).to_numpy()

            if icme_match.sum() > 0:
                cme_indices.append(j)
                cme_flags.append(True)
                icme_list = self._merge_icme_into_enhancement(
                    enhancement_list, icme_list, j, icme_match
                )
            else:
                hss_indices.append(j)
                cme_flags.append(False)

        enhancement_list = enhancement_list.copy()
        enhancement_list["cme_flag"] = cme_flags

        cme_list = enhancement_list.iloc[cme_indices].reset_index(drop=True)
        hss_list = enhancement_list.iloc[hss_indices].reset_index(drop=True)

        if self._add_all_icme:
            cme_list = self._append_unmatched_icmes(cme_list, icme_list)

        cme_list = self._merge_overlapping_cmes(cme_list)

        return enhancement_list, cme_list, hss_list

    def _merge_icme_into_enhancement(
        self,
        enhancement_list: pd.DataFrame,
        icme_list: pd.DataFrame,
        j: int,
        icme_match: np.ndarray,
    ) -> pd.DataFrame:
        """Expand enhancement ``j`` to cover matched ICME interval(s).

        Adjusts adjacent enhancement boundaries if the expanded interval
        would overlap them, then drops the matched ICMEs from ``icme_list``.

        Args:
            enhancement_list: Full enhancement DataFrame (modified in place).
            icme_list: Current ICME list (matched rows will be dropped).
            j: Row index of the enhancement being processed.
            icme_match: Boolean mask over ``icme_list`` rows.

        Returns:
            Updated ``icme_list`` with matched rows removed.
        """
        matched = icme_list.loc[icme_match]

        icme_start = matched["start"].iloc[0]
        icme_end = matched["end"].iloc[-1]

        # Expand enhancement start to earliest ICME start if earlier
        if icme_start < enhancement_list.at[j, "start"]:
            if j > 0 and enhancement_list.at[j - 1, "end"] >= icme_start:
                enhancement_list.at[j - 1, "end"] = (
                    icme_start - timedelta(hours=1)
                )
            enhancement_list.at[j, "start"] = icme_start

        # Expand enhancement end to latest ICME end if later
        if icme_end > enhancement_list.at[j, "end"]:
            if (
                j < len(enhancement_list) - 1
                and enhancement_list.at[j + 1, "start"] <= icme_end
            ):
                enhancement_list.at[j + 1, "start"] = (
                    icme_end + timedelta(hours=1)
                )
            enhancement_list.at[j, "end"] = icme_end

        # Drop all matched ICMEs and reset index for subsequent iterations
        icme_list = icme_list.drop(
            index=icme_list.index[icme_match]
        ).reset_index(drop=True)

        return icme_list

    def _append_unmatched_icmes(
        self,
        cme_list: pd.DataFrame,
        icme_list: pd.DataFrame,
    ) -> pd.DataFrame:
        """Append remaining (unmatched) ICME entries to the CME catalogue.

        Each unmatched ICME is expanded by :attr:`_ICME_PAD_BEFORE_HOURS`
        before its start and :attr:`_ICME_PAD_AFTER_HOURS` after its end
        before appending.

        Args:
            cme_list: CME catalogue built from matched enhancements.
            icme_list: Remaining (unmatched) ICMEs.

        Returns:
            Extended CME catalogue, sorted by ``'start'``.
        """
        if len(icme_list) == 0:
            return cme_list

        unmatched = icme_list.copy()
        unmatched["start"] -= timedelta(hours=self._ICME_PAD_BEFORE_HOURS)
        unmatched["end"] += timedelta(hours=self._ICME_PAD_AFTER_HOURS)

        return (
            pd.concat([cme_list, unmatched], axis=0)
            .sort_values("start")
            .reset_index(drop=True)
        )

    @staticmethod
    def _merge_overlapping_cmes(cme_list: pd.DataFrame) -> pd.DataFrame:
        """Merge overlapping or adjacent CME intervals in the catalogue.

        Iterates through the sorted CME list and merges any entry whose end
        overlaps the start of the next entry by extending the current end to
        the maximum of the two ends.

        Args:
            cme_list: CME catalogue sorted by ``'start'``.

        Returns:
            CME catalogue with overlapping intervals merged.
        """
        cme_list = cme_list.reset_index(drop=True)
        i = 0
        while i < len(cme_list) - 1:
            if cme_list.at[i, "end"] >= cme_list.at[i + 1, "start"]:
                # Extend current interval to cover both, then drop the next
                cme_list.at[i, "end"] = max(
                    cme_list.at[i, "end"], cme_list.at[i + 1, "end"]
                )
                cme_list = cme_list.drop(index=i + 1).reset_index(drop=True)
            else:
                i += 1
        return cme_list