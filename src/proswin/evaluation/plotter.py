"""Plotting utilities for probabilistic solar wind speed forecasts."""
import copy
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib.colors import ListedColormap
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MinMaxScaler

from proswin.evaluation.hss_finder import HSSFinder
from proswin.evaluation.solar_wind_prediction_evaluator import PredictionEvaluator
from proswin.predictions import SolarWindPrediction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default y-axis limits for all time-series plots (km/s).
_SWS_YLIM: tuple[int, int] = (250, 850)

# Default x-axis limits for the distribution histogram (km/s).
_HIST_XLIM: tuple[int, int] = (250, 900)

# DPI used for all saved figures — high enough for journal submission.
_FIGURE_DPI: int = 500


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def make_confidence_cmap_from_base(
    base_cmap_name: str = "gist_heat",
    n_steps: int = 200,
    jump_points: tuple[float, float] = (0.025, 0.975),
) -> ListedColormap:
    """Create a symmetric discrete colormap for confidence interval shading.

    The colormap is built by taking one half of ``base_cmap_name``, mirroring
    it to produce a symmetric palette, then replacing colours beyond
    ``jump_points`` with white.  The result has the most saturated colour at
    the centre (used for the innermost CI band) fading to white at the edges
    (used for the outermost band), giving an intuitive visual hierarchy.

    Args:
        base_cmap_name: Name of any Matplotlib colormap, e.g. ``"gist_heat"``
            or ``"viridis"``.
        n_steps: Total number of discrete colour steps.  Values above ~200
            are visually indistinguishable on screen.
        jump_points: ``(low, high)`` fractional positions at which the
            colormap transitions sharply to white.  The default ``(0.025,
            0.975)`` aligns with the outer edges of a 95 % prediction
            interval.

    Returns:
        A :class:`~matplotlib.colors.ListedColormap` named
        ``"{base_cmap_name}_conf95"``.
    """
    base_cmap = plt.get_cmap(base_cmap_name)
    half_steps = n_steps // 2
    positions = np.linspace(0, 1, half_steps + 1)
    colors_half = [base_cmap(p) for p in positions]

    # Replace colours beyond the jump threshold with white so the outermost
    # CI band fades out, keeping figures uncluttered.
    jump_idx = int(jump_points[0] * n_steps)
    if 0 < jump_idx < half_steps:
        for i in range(half_steps - jump_idx + 1, half_steps + 1):
            colors_half[i] = (1, 1, 1, 1)

    # Mirror to produce a palette symmetric around 0.5.
    colors = colors_half[::-1] + colors_half[1:]
    return ListedColormap(colors, name=f"{base_cmap_name}_conf95")


def plot_crps_illustration(
    prediction: SolarWindPrediction,
    observation: float,
    figure_dir: str | Path,
) -> None:
    """Plot a single-timestep illustration of how the CRPS is computed.

    Visualises the CRPS integral as the squared area between the predicted
    CDF F(z) and the Heaviside observation indicator 1[y <= z].  Intended as
    a didactic figure for the paper rather than a bulk evaluation plot.

    The x-axis is densely sampled between the 0.1th and 99.9th predicted
    quantiles, with extra resolution added in a +/-1 km/s window around the
    observation so the step discontinuity in the indicator function is
    rendered sharply.

    The figure is saved as ``crps_illustration.png`` in ``figure_dir``.

    Args:
        prediction: A single-timestep
            :class:`~proswin.predictions.SolarWindPrediction` slice,
            obtained via ``predictions.slice([date])``.
        observation: The observed solar wind speed at that timestep (km/s).
        figure_dir: Directory where the figure is saved.  Created
            automatically if it does not exist.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Build an x-grid that covers the full predicted distribution while
    # adding extra density around the observation so the step in H(z) is
    # rendered sharply rather than as a diagonal ramp.
    x_main = np.linspace(
        prediction.get_quantiles(0.001),
        prediction.get_quantiles(0.999),
        100,
    )
    x_around_obs = np.linspace(observation - 1, observation + 1, 100)
    x = np.sort(np.concatenate([x_main, x_around_obs]))

    # Predicted CDF and the Heaviside observation indicator H(z) = 1[z >= y].
    F = prediction.get_cdfs(x)
    H = (x >= observation).astype(float)

    fig, ax = plt.subplots()
    ax.yaxis.grid(True, linestyle="--")

    ax.plot(x, F, color="red", lw=1, label=r"Predicted CDF  $F_i(z)$", zorder=3)
    ax.plot(
        x,
        H,
        color="blue",
        lw=1.5,
        linestyle="--",
        label=r"Observed CDF  $\mathbf{1}[y_i \leq z]$",
        zorder=3,
    )

    # Shade the squared-difference area on both sides of the observation.
    # The mask is split at the observation to align with the step
    # discontinuity in H(z) and avoid a spurious diagonal fill at that point.
    left_mask = x < observation
    right_mask = x >= observation
    ax.fill_between(
        x[left_mask],
        F[left_mask],
        H[left_mask],
        color="red",
        alpha=0.3,
        label=r"$(F_i - \mathbf{1}[y_i \leq z])^2$ area",
        zorder=2,
    )
    ax.fill_between(
        x[right_mask],
        F[right_mask],
        H[right_mask],
        color="red",
        alpha=0.3,
        zorder=2,
    )

    # Label the observation on the x-axis.  Using get_xaxis_transform() fixes
    # the y-position in axes coordinates (0 = bottom) regardless of y-scaling.
    ax.text(
        observation,
        0.02,
        f"$y = {observation}$",
        ha="center",
        va="bottom",
        color="blue",
        fontsize=11,
        transform=ax.get_xaxis_transform(),
    )

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.1, 1.03)
    ax.set_ylabel("Cumulative probability", fontsize=15)
    ax.set_xlabel("Solar wind speed (km/s)", fontsize=15)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "crps_illustration.png",
        bbox_inches="tight",
        dpi=_FIGURE_DPI,
    )


def plot_cv_split(
    cv_split: dict[str, list[pd.DatetimeIndex]],
    figure_dir: str | Path,
) -> None:
    """Plot the cross-validation train/validation/test split as a timeline.

    Each CV fold occupies one horizontal band on the y-axis.  Within each
    band, the training, validation, and test periods are shaded in distinct
    colours.  Data gaps (consecutive timestamps more than one hour apart) are
    handled correctly: each contiguous segment is drawn as a separate
    ``axvspan`` so that gaps appear as empty space rather than being filled.

    The figure is saved as ``cross_validation_split.png`` in ``figure_dir``.

    Args:
        cv_split: Dictionary with keys ``"train"``, ``"val"``, and ``"test"``,
            each mapping to a list of :class:`pandas.DatetimeIndex` objects,
            one per fold.  This is the format produced by
            :class:`~proswin.data_management.cv_split.SequentialCVSplitGenerator`.
        figure_dir: Directory where the figure is saved.  Created
            automatically if it does not exist.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    n_folds = len(cv_split["train"])
    fig, ax = plt.subplots()

    # Colours and legend labels for the three split types.
    split_styles: list[tuple[str, str, str]] = [
        ("train", "tab:blue", "Training"),
        ("val", "tab:orange", "Validation"),
        ("test", "tab:green", "Test"),
    ]

    for fold_idx in range(n_folds):
        # Each fold occupies 1/n_folds of the y-axis height.  A padding of
        # 0.01 on each edge prevents adjacent fold bands from touching.
        ymin = 1 - (fold_idx + 1) / n_folds + 0.01
        ymax = 1 - fold_idx / n_folds - 0.01

        for split_key, color, legend_label in split_styles:
            dates = cv_split[split_key][fold_idx].unique()
            _draw_split_spans(
                ax=ax,
                dates=dates,
                ymin=ymin,
                ymax=ymax,
                color=color,
                # Only label the first fold to avoid duplicate legend entries.
                label=legend_label if fold_idx == 0 else None,
            )

    # Deduplicate legend entries and place above the plot.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        fancybox=True,
        ncol=3,
        fontsize=10,
    )

    # y-tick positions are at the midpoint of each fold band.
    tick_positions = [1 - (i + 0.5) / n_folds for i in range(n_folds)]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(range(1, n_folds + 1))

    ax.set_xlabel("Date", fontsize=15)
    ax.set_ylabel("CV split number", fontsize=15)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "cross_validation_split.png",
        bbox_inches="tight",
        dpi=_FIGURE_DPI,
    )


def _draw_split_spans(
    ax: plt.Axes,
    dates: pd.DatetimeIndex,
    ymin: float,
    ymax: float,
    color: str,
    label: str | None,
) -> None:
    """Draw one or more horizontal spans for a (possibly gapped) date sequence.

    Gaps larger than one hour in ``dates`` are left unshaded so that the
    visual representation reflects the actual data coverage rather than
    interpolating across missing periods.

    Args:
        ax: Axes to draw on.
        dates: Timestamps for this split segment (need not be pre-sorted).
        ymin: Bottom edge of the horizontal band in axes coordinates (0-1).
        ymax: Top edge of the horizontal band in axes coordinates (0-1).
        color: Fill colour for the span.
        label: Legend label, or ``None`` to suppress the legend entry.
    """
    dates = dates.sort_values()
    time_diffs = np.diff(dates).astype("timedelta64[ns]")
    gap_mask = time_diffs > np.timedelta64(1, "h")

    if not np.any(gap_mask):
        # No gaps — draw a single span covering the full range.
        ax.axvspan(
            dates[0],
            dates[-1],
            ymin=ymin,
            ymax=ymax,
            color=color,
            label=label,
        )
        return

    # Identify contiguous segments separated by gaps.
    # gap_indices marks positions just *before* each gap, so segment i runs
    # from segment_starts[i] to segment_ends[i] inclusive.
    gap_indices = np.where(gap_mask)[0]
    segment_starts = np.concatenate([[0], gap_indices + 1])
    segment_ends = np.concatenate([gap_indices, [len(dates) - 1]])

    for seg_idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        ax.axvspan(
            dates[start],
            dates[end],
            ymin=ymin,
            ymax=ymax,
            color=color,
            # Only the first segment carries the label to avoid duplicates.
            label=label if seg_idx == 0 else None,
        )


# ---------------------------------------------------------------------------
# Main plotter class
# ---------------------------------------------------------------------------


class SolarWindPredictionPlotter:
    """Produces publication-quality figures for PROSWIN solar wind forecasts.

    Each public ``plot_*`` method writes a PNG file to ``figure_dir`` and
    returns ``None``.  Figures are *not* shown automatically; call
    ``plt.show()`` after all ``plot_*`` calls if interactive display is
    desired, or pass ``show=True`` to :meth:`plot_all`.

    Args:
        predictions: Probabilistic SWS forecast object providing
            ``get_expected_values()``, ``get_quantiles()``, ``get_cdfs()``,
            ``get_pdfs()``, and ``slice()`` methods.
        observations: Hourly in-situ SWS time series (km/s) with a
            :class:`pandas.DatetimeIndex`, aligned to ``predictions``.
        figure_dir: Directory where PNG figures are saved.  Created
            automatically if it does not exist.
        solar_event_dir: Directory containing solar event catalogues (e.g.
            ``hss_list.csv``).  Required only by methods that annotate HSS
            events.
        physical_feature_dir: Directory containing physical feature CSV files
            (e.g. ``sunspot_number.csv``).  Required by
            :meth:`solar_cycle_pred_scores`.
        interpolation_dates: Timestamps of data gaps that were filled by
            interpolation.  When provided, the evaluator excludes these
            dates from HSS association.
        hss_finder: Pre-initialised :class:`HSSFinder` instance used to
            detect high-speed stream peaks.  Required by methods that
            evaluate or annotate HSS events.
    """

    def __init__(
        self,
        predictions: SolarWindPrediction,
        observations: pd.Series,
        figure_dir: str | Path,
        solar_event_dir: str | Path | None = None,
        physical_feature_dir: str | Path | None = None,
        interpolation_dates: pd.DatetimeIndex | None = None,
        hss_finder: HSSFinder | None = None,
    ) -> None:
        self.predictions = predictions
        self.observations = observations

        # Normalise to Path so that the / operator works regardless of whether
        # the caller passed a string or a Path object.
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.solar_event_dir = (
            Path(solar_event_dir) if solar_event_dir is not None else None
        )
        self.physical_feature_dir = (
            Path(physical_feature_dir) if physical_feature_dir is not None else None
        )
        self.interpolation_dates = interpolation_dates
        self.hss_finder = hss_finder

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot_all(self, show: bool = True) -> None:
        """Reproduce all figures used in the PROSWIN paper.

        Calls each ``plot_*`` method with the exact arguments used to generate
        the published figures and writes them to ``self.figure_dir``.

        Args:
            show: If ``True``, call ``plt.show()`` after generating all
                figures.  Set to ``False`` when running headlessly on
                HPC/SLURM where no display is available.
        """
        # Figure 1: example time series with calibrated confidence bands.
        self.plot_time_series(
            interval=[datetime(2016, 11, 7, 0), datetime(2016, 12, 18, 0)],
            overlay="confidence_intervals",
            ci_levels=[0.5, 0.8, 0.9, 0.95],
        )
        # Figure 2: probability of exceeding operationally relevant thresholds.
        self.plot_time_series(
            interval=[datetime(2019, 9, 21, 0), datetime(2019, 10, 9, 0)],
            overlay="probability_thresholds",
            thresholds=[500, 600, 700],
        )
        # Figure 3: HSS event probability over a full solar rotation.
        self.plot_time_series(
            interval=[datetime(2016, 9, 25), datetime(2017, 1, 24)],
            overlay="hss_probability",
            mark_above_50=False,
        )
        # Figure 4: confidence interval calibration for all data and HSS peaks.
        self.plot_calibration_curve_with_peaks()

        # Figure 5: threshold exceedance probability reliability diagram.
        self.plot_threshold_calibration_curve(thresholds=[450])

        # Figure 6: predicted vs. observed HSS peak scatter with CI bands.
        self.plot_hss_peak_scatter()

        # Figure 7: prediction score evolution over the solar cycle.
        self.solar_cycle_pred_scores()

        # Figure 8: aggregated predictive distribution vs. observed histogram.
        self.aggregated_pred_distribution_vs_obs_histogram()

        if show:
            plt.show()

    def plot_time_series(
        self,
        interval: list[datetime] | None = None,
        overlay: str = "mean",
        ci_levels: list[float] | None = None,
        thresholds: list[int] | None = None,
        mark_above_50: bool = False,
    ) -> None:
        """Plot a time-series window with one of four visualisation overlays.

        The ``overlay`` argument selects what is drawn on top of the base
        observed and expected-value lines:

        * ``"mean"`` — observed + expected value only.
        * ``"confidence_intervals"`` — shaded symmetric confidence bands
          around the expected value.  Pass band coverages via ``ci_levels``
          (e.g. ``[0.5, 0.8, 0.9, 0.95]``).
        * ``"probability_thresholds"`` — stacked probability-of-exceedance
          fills for one or more speed thresholds (km/s) via ``thresholds``
          (e.g. ``[500, 600, 700]``).
        * ``"hss_probability"`` — probability that SWS exceeds 450 km/s,
          used as a proxy for high-speed stream (HSS) occurrence.

        The filename encodes the interval dates and overlay mode so that
        repeated calls with different arguments never overwrite each other.

        Args:
            interval: ``[start, end]`` datetimes defining the x-axis window.
                Defaults to the full extent of ``self.observations``.
            overlay: One of ``"mean"``, ``"confidence_intervals"``,
                ``"probability_thresholds"``, or ``"hss_probability"``.
            ci_levels: Required when ``overlay="confidence_intervals"``.
                List of coverage probabilities in (0, 1).
            thresholds: Required when ``overlay="probability_thresholds"``.
                List of SWS thresholds in km/s.
            mark_above_50: Only used when ``overlay="hss_probability"``.
                If ``True``, shade merged intervals where P(SWS > 450) > 0.5.

        Raises:
            ValueError: If ``overlay`` is not one of the four recognised
                strings.
        """
        _VALID_OVERLAYS = {
            "mean",
            "confidence_intervals",
            "probability_thresholds",
            "hss_probability",
        }
        if overlay not in _VALID_OVERLAYS:
            raise ValueError(
                f"Unknown overlay '{overlay}'. "
                f"Must be one of: {sorted(_VALID_OVERLAYS)}"
            )

        if interval is None:
            interval = [
                self.observations.index[0],
                self.observations.index[-1],
            ]

        # Use a wider figure for long windows so the x-axis stays readable.
        long_interval = (interval[1] - interval[0]) > timedelta(days=60)
        figsize = (2 * 6.4, 4.8) if long_interval else (6.4, 4.8)
        fig, ax = plt.subplots(figsize=figsize)

        # The two probability overlays use a second y-axis (right, 0-1 scale)
        # so that the probability curve and the SWS time series share the same
        # x-axis without conflicting y-scales.
        uses_second_axis = overlay in {"probability_thresholds", "hss_probability"}
        ax2 = ax.twinx() if uses_second_axis else None
        if ax2 is not None:
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Probability", fontsize=15)
            # twinx() places ax2 on top of ax at the figure level, so ax2's
            # fills would cover ax's lines regardless of per-artist zorder.
            # Fix: lower ax2's figure-level zorder so ax renders last (on top),
            # then make ax's background transparent so ax2 is still visible.
            ax2.set_zorder(ax.get_zorder() - 1)
            ax.patch.set_visible(False)

        # Slice predictions and observations to the requested window.
        plot_index = self.predictions.data.loc[interval[0] : interval[1], :].index
        pred = self.predictions.slice(plot_index)
        exp_val = pred.get_expected_values()
        obs = self.observations.loc[interval[0] : interval[1]]

        # zorder=3 ensures these lines render in front of all overlay fills
        # (zorder=2) and overlay edge lines (zorder=2.5).
        ax.plot(obs, "blue", label="Observation", zorder=3)
        # The expected value line is omitted for the HSS probability overlay
        # because that panel focuses on the exceedance probability alone.
        if overlay != "hss_probability":
            ax.plot(
                exp_val,
                "black",
                linestyle="solid",
                label="Expected value",
                zorder=3,
            )

        # Dispatch to the appropriate private overlay method.
        if overlay == "confidence_intervals":
            self._add_confidence_interval_overlay(ax, pred, exp_val, ci_levels or [])
        elif overlay == "probability_thresholds":
            self._add_probability_threshold_overlay(ax2, pred, thresholds or [])
        elif overlay == "hss_probability":
            self._add_hss_probability_overlay(ax2, pred, mark_above_50)
        # overlay == "mean": nothing extra to draw.

        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlim(interval)
        ax.set_ylim(_SWS_YLIM)
        ax.set_xlabel("Date", fontsize=15)
        ax.set_ylabel("Solar wind speed (km/s)", fontsize=15)

        # Merge legend entries from both axes, deduplicating labels that
        # fill_between generates twice (one per fill_between call).
        handles, labels = ax.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
        by_label = dict(zip(labels, handles))
        legend_ax = ax2 if ax2 is not None else ax
        legend_ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=10,
            loc="upper right",
            ncol=1 if ax2 is not None else 3,
        )

        start_str = datetime.strftime(interval[0], "%Y-%m-%d")
        end_str = datetime.strftime(interval[1], "%Y-%m-%d")
        fig.tight_layout()
        fig.savefig(
            self.figure_dir / f"{start_str}_{end_str}_{overlay}.png",
            bbox_inches="tight",
            dpi=_FIGURE_DPI,
        )

    def solar_cycle_pred_scores(self) -> None:
        """Plot prediction score and RMSE evolution over the solar cycle.

        Computes yearly prediction scores (CRPS-based and RMSE-based) for
        consecutive 12-month windows spanning June 2010 to June 2024.  A
        smoothed international sunspot number (SSN) curve is overlaid on the
        same x-axis to give solar-cycle context: years of high solar activity
        tend to produce more high-speed streams and thus different forecast
        difficulty.

        The SSN is read from ``physical_feature_dir / "sunspot_number.csv"``
        and scaled to the CRPS y-range so both curves share the left axis.
        RMSE is plotted on a separate right axis so its absolute scale is
        preserved.

        Raises:
            ValueError: If ``self.physical_feature_dir`` is ``None``.
        """
        if self.physical_feature_dir is None:
            raise ValueError(
                "physical_feature_dir must be set to plot solar-cycle scores."
            )

        ssn = pd.read_csv(
            self.physical_feature_dir / "sunspot_number.csv",
            index_col=0,
            parse_dates=[0],
        )

        fig, ax = plt.subplots(figsize=(2 * 6.4, 4.8))

        # Iterate over 12-month windows from June 2010 to June 2024.
        # The first window is 13 months to avoid a short stub at dataset start.
        start = datetime(2010, 6, 1)
        end = datetime(2024, 6, 30, 23)
        current_year = copy.copy(start)

        all_midpoints: list[datetime] = []
        all_prediction_scores: list[list[float]] = []

        while current_year <= end:
            if current_year == start:
                yearly_date_range = pd.date_range(
                    current_year,
                    current_year + relativedelta(years=1, months=1, hours=-1),
                    freq="1h",
                )
            else:
                yearly_date_range = pd.date_range(
                    current_year,
                    current_year + relativedelta(years=1, hours=-1),
                    freq="1h",
                )

            idx = self.predictions.index.intersection(yearly_date_range)

            # Restrict interpolation dates to this window, if provided.
            if self.interpolation_dates is not None:
                yearly_interpolation_dates = self.interpolation_dates[
                    (self.interpolation_dates >= yearly_date_range[0])
                    & (self.interpolation_dates <= yearly_date_range[-1])
                ]
            else:
                yearly_interpolation_dates = None

            yearly_midpoint = yearly_date_range[len(yearly_date_range) // 2]
            yearly_pred = self.predictions.slice(idx)
            yearly_obs = self.observations.loc[idx]

            evaluator = PredictionEvaluator(
                yearly_pred,
                yearly_obs,
                self.hss_finder,
                yearly_interpolation_dates,
            )
            # evaluate() returns either (metrics, brier_table) or just metrics
            # depending on whether Brier-score computation is enabled.
            result = evaluator.evaluate()
            metrics = result[0] if isinstance(result, tuple) else result

            all_midpoints.append(yearly_midpoint)
            all_prediction_scores.append(
                [metrics.loc["ps_crps"], metrics.loc["ps_rmse"]]
            )
            # Advance to the first hour after this window ends.
            current_year = yearly_date_range[-1] + timedelta(hours=1)

        scores = np.array(all_prediction_scores)

        line1 = ax.plot(
            all_midpoints, scores[:, 0], "o-", color="tab:blue", label="CRPS"
        )

        # Scale SSN to the CRPS y-range so it fits on the left axis.
        # A 13-month rolling mean matches the 12-month evaluation windows.
        ssn_smoothed = ssn.rolling(window=13).mean()
        ssn_smoothed = ssn_smoothed.loc[
            start - relativedelta(months=12) : end + relativedelta(months=12)
        ]
        ax.set_xlim(
            [start - relativedelta(months=12), end + relativedelta(months=12)]
        )
        mms = MinMaxScaler(feature_range=(scores[:, 0].min(), scores[:, 0].max()))
        ssn_scaled = mms.fit_transform(ssn_smoothed.values.reshape(-1, 1))
        line3 = ax.plot(
            ssn_smoothed.index, ssn_scaled, "--", color="grey", label="SSN"
        )

        # RMSE on a separate right axis so its absolute scale is visible.
        ax2 = ax.twinx()
        line2 = ax2.plot(
            all_midpoints, scores[:, 1], "o-", color="tab:orange", label="RMSE"
        )

        ax.set_ylabel("Prediction score CRPS (km/s)", fontsize=15)
        ax.set_xlabel("Date", fontsize=15)
        ax2.set_ylabel("Prediction score RMSE (km/s)", fontsize=15)

        # Combine legend entries from both axes into one legend.
        lines = line1 + line2 + line3
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels)

        fig.tight_layout()
        fig.savefig(
            self.figure_dir / "solar_cycle.png",
            bbox_inches="tight",
            dpi=_FIGURE_DPI,
        )

    def aggregated_pred_distribution_vs_obs_histogram(self) -> None:
        """Plot the aggregated predictive distribution against observations.

        Shows three elements in one panel:

        * Blue histogram — empirical distribution of in-situ observations.
        * Orange histogram — empirical distribution of predicted expected
          values.
        * Red curve + fill — mean predictive PDF averaged over all timesteps,
          representing the marginal distribution the model assigns to SWS.

        This is a marginal calibration check: if the model is well calibrated,
        the red curve and the blue histogram should broadly agree.  It
        complements the conditional calibration checks in
        :meth:`plot_calibration_curve` and
        :meth:`plot_threshold_calibration_curve`.
        """
        fig, ax = plt.subplots()

        # Build a shared bin grid covering observations and the 1st-99th
        # percentile of the predicted distribution.  Rounding to the nearest
        # 10 km/s aligns bin edges with round numbers for cleaner tick labels.
        lb = 10 * (
            np.min(
                [
                    self.observations.min(),
                    self.predictions.get_quantiles(0.01).min(),
                ]
            )
            // 10
        )
        ub = 10 * (
            np.max(
                [
                    self.observations.max(),
                    self.predictions.get_quantiles(0.99).max(),
                ]
            )
            // 10
            + 1
        )
        bins = np.linspace(lb, ub, int((ub - lb) // 10 + 1))

        ax.hist(
            self.observations,
            bins=bins,
            color="blue",
            alpha=0.5,
            label="Observations",
            density=True,
        )
        ax.hist(
            self.predictions.get_expected_values(),
            bins=bins,
            color="orange",
            alpha=0.5,
            label="Expected values",
            density=True,
        )

        # Evaluate the mean predictive PDF on a fine grid.
        # get_pdfs(v) returns the PDF value at speed v for every timestep;
        # averaging across timesteps gives the marginal predictive PDF.
        val = np.linspace(lb, ub, 100)
        grid_pdf_values = [self.predictions.get_pdfs(v) for v in val]
        cum_dist = pd.concat(grid_pdf_values, axis=1)
        dist_mean = cum_dist.mean(axis=0)

        ax.plot(val, dist_mean, "red", label="Probabilistic predictions")
        ax.fill_between(val, dist_mean, np.zeros(len(val)), color="red", alpha=0.3)

        ax.legend()
        ax.set_xlabel("Solar wind speed (km/s)", fontsize=15)
        ax.set_ylabel("Probability density", fontsize=15)
        ax.set_xlim(_HIST_XLIM)

        fig.tight_layout()
        fig.savefig(
            self.figure_dir / "agg_dist_hist.png",
            bbox_inches="tight",
            dpi=_FIGURE_DPI,
        )

    def plot_calibration_curve(
        self,
        pred: SolarWindPrediction,
        obs: pd.Series,
        label: str = "timeline",
        second_pred: SolarWindPrediction | None = None,
        second_obs: pd.Series | None = None,
        second_label: str = "peaks",
        quantile_levels: np.ndarray | None = None,
    ) -> None:
        """Plot a reliability / calibration curve for probabilistic forecasts.

        A calibration curve compares nominal confidence interval coverage
        (x-axis) with empirically observed coverage (y-axis).  A perfectly
        calibrated model lies on the diagonal.  Deviation above the diagonal
        means intervals are too wide (over-dispersed); deviation below means
        they are too narrow (under-dispersed).

        Passing ``second_pred`` and ``second_obs`` adds a second curve to the
        same panel — typically used to compare timeline calibration against
        HSS-peak calibration.

        For an automatic version that computes the HSS peak slice internally
        from ``self.hss_finder``, use :meth:`plot_calibration_curve_with_peaks`.

        Args:
            pred: Primary probabilistic prediction object.
            obs: Observations aligned with ``pred``.
            label: Legend label and filename component for the primary curve.
            second_pred: Optional second prediction for a comparison curve.
                Must be provided together with ``second_obs``.
            second_obs: Observations aligned with ``second_pred``.
            second_label: Legend label for the second curve.
            quantile_levels: Nominal coverage levels to evaluate.  Defaults
                to 99 evenly spaced values from 0.01 to 0.99.

        Raises:
            ValueError: If exactly one of ``second_pred`` / ``second_obs``
                is provided.
        """
        if (second_pred is None) != (second_obs is None):
            raise ValueError(
                "second_pred and second_obs must be provided together or "
                "both omitted."
            )

        if quantile_levels is None:
            quantile_levels = np.linspace(0.01, 0.99, 99)

        fig, ax = plt.subplots()

        _, coverage = self._compute_empirical_coverage(pred, obs, quantile_levels)
        ax.plot(quantile_levels, coverage, label=label)

        if second_pred is not None:
            _, second_coverage = self._compute_empirical_coverage(
                second_pred, second_obs, quantile_levels
            )
            ax.plot(quantile_levels, second_coverage, label=second_label)

        ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="Ideal")

        ax.set_xlabel("Predicted confidence interval", fontsize=15)
        ax.set_ylabel("Observed fraction", fontsize=15)
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(
            self.figure_dir / "calibration_confidence_intervals.png",
            bbox_inches="tight",
            dpi=_FIGURE_DPI,
        )

    def plot_calibration_curve_with_peaks(
        self,
        quantile_levels: np.ndarray | None = None,
    ) -> None:
        """Plot timeline and HSS-peak calibration curves on a single panel.

        Convenience wrapper around :meth:`plot_calibration_curve` that
        computes the HSS peak slices automatically from ``self.hss_finder``.
        The peak slice is derived from hit pairs: predicted peaks matched to
        an observed peak within the association window.

        Args:
            quantile_levels: Passed through to :meth:`plot_calibration_curve`.

        Raises:
            ValueError: If ``self.hss_finder`` is ``None``.
        """
        if self.hss_finder is None:
            raise ValueError(
                "hss_finder must be set on the plotter to call "
                "plot_calibration_curve_with_peaks()."
            )

        pred_peaks, obs_peaks = self.hss_finder.find(self.predictions)

        hit_mask = pred_peaks.loc[:, "hit"].to_numpy()
        pred_peak_dates = pred_peaks.index[hit_mask]
        associated_obs_dates = pred_peaks.loc[hit_mask, "associated"].to_numpy()

        peak_pred = self.predictions.slice(pred_peak_dates)
        peak_obs = obs_peaks.loc[associated_obs_dates, "peak_value"]
        # Re-index to align peak_obs with peak_pred so that each row is a
        # matched (forecast, observation) pair at the same timestamp.
        peak_obs.index = pred_peak_dates

        self.plot_calibration_curve(
            pred=self.predictions,
            obs=self.observations,
            label="Timeline data points",
            second_pred=peak_pred,
            second_obs=peak_obs,
            second_label="HSS peak values",
            quantile_levels=quantile_levels,
        )

    def plot_threshold_calibration_curve(
        self,
        thresholds: list[int],
        n_bins: int = 10,
    ) -> None:
        """Plot a reliability diagram for predicted threshold exceedance probabilities.

        For each speed threshold T (km/s), the model predicts P(SWS > T) at
        every timestep.  This method uses scikit-learn's
        :func:`~sklearn.calibration.calibration_curve` to bin those predicted
        probabilities and compare them against the observed exceedance fraction
        within each bin.  A perfectly calibrated model lies on the diagonal.

        This is complementary to :meth:`plot_calibration_curve`: that method
        checks calibration of symmetric prediction intervals across all
        coverage levels, whereas this method checks calibration of exceedance
        probabilities at specific operationally relevant speed thresholds.

        Args:
            thresholds: SWS thresholds (km/s), e.g. ``[450, 500, 600, 700]``.
            n_bins: Number of equal-width probability bins.  More bins give
                finer resolution but noisier estimates for rare events.
                Defaults to 10.
        """
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], linestyle="--", color="red", label="Ideal")

        for threshold in thresholds:
            # P(SWS > threshold) = 1 - CDF(threshold).
            predicted_prob = 1 - self.predictions.get_cdfs(threshold)

            # Binary observations: 1 if the event occurred, 0 otherwise.
            observed_event = np.array(self.observations >= threshold, dtype=int)

            # calibration_curve() bins predicted_prob into n_bins equal-width
            # buckets and returns the mean observed fraction (prob_true) and
            # mean predicted probability (prob_pred) within each bucket.
            prob_true, prob_pred = calibration_curve(
                observed_event,
                predicted_prob,
                n_bins=n_bins,
                strategy="uniform",
            )
            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                label=f"Probability SWS > {threshold}",
            )

        ax.set_xlabel("Mean predicted probability", fontsize=15)
        ax.set_ylabel("Observed fraction", fontsize=15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            self.figure_dir / "calibration_threshold.png",
            bbox_inches="tight",
            dpi=_FIGURE_DPI,
        )

    def plot_hss_peak_scatter(
        self,
        ci_levels: list[float] | None = None,
        bin_size: float = 50.0,
        sws_range: tuple[float, float] = (400.0, 900.0),
    ) -> None:
        """Plot predicted vs. observed HSS peak speeds with binned CI bands.

        HSS peak pairs are detected automatically from ``self.hss_finder``.
        Each scatter point is one matched hit pair: x is the observed peak
        speed, y is the predicted expected value at that timestamp.  The
        dashed diagonal marks perfect agreement.

        On top of the scatter, predictions are grouped into equal-width
        observation-speed bins.  Within each bin the mean expected value and
        mean quantile bounds are computed to produce a smooth summary curve
        with shaded confidence bands.  This reveals whether the model is
        systematically biased at specific speed ranges and whether uncertainty
        is well-calibrated across the full speed range.

        Args:
            ci_levels: Coverage probabilities for the shaded bands.
                Defaults to ``[0.5, 0.8, 0.9, 0.95]``.
            bin_size: Width of each observation bin in km/s.  Defaults to 50.
            sws_range: ``(min, max)`` speed range for both axes and binning
                in km/s.  Defaults to ``(400, 900)``.

        Raises:
            ValueError: If ``self.hss_finder`` is ``None``.
        """
        if self.hss_finder is None:
            raise ValueError(
                "hss_finder must be set on the plotter to call "
                "plot_hss_peak_scatter()."
            )

        if ci_levels is None:
            ci_levels = [0.5, 0.8, 0.9, 0.95]

        # Detect HSS peaks and extract the hit pairs.
        pred_peaks, obs_peaks = self.hss_finder.find(self.predictions)
        hit_mask = pred_peaks.loc[:, "hit"].to_numpy()
        pred_peak_dates = pred_peaks.index[hit_mask]
        associated_obs_dates = pred_peaks.loc[hit_mask, "associated"].to_numpy()

        peak_pred = self.predictions.slice(pred_peak_dates)
        # Re-index peak_obs to pred_peak_dates so slicing operations inside
        # _compute_binned_ci use the prediction timestamps, not the
        # observation timestamps (which may differ by the association offset).
        peak_obs = obs_peaks.loc[associated_obs_dates, "peak_value"].copy()
        peak_obs.index = pred_peak_dates

        xmin, xmax = sws_range
        exp_vals, ci_array = self._compute_binned_ci(
            peak_pred, peak_obs, ci_levels, bin_size, xmin, xmax
        )

        cmap = make_confidence_cmap_from_base(base_cmap_name="gist_heat", n_steps=200)
        fig, ax = plt.subplots()

        # Draw scatter first at low zorder; re-drawn at zorder=5 after the CI
        # bands so individual points are always visible on top.
        ax.scatter(
            peak_obs,
            peak_pred.get_expected_values(),
            marker="o",
            s=10,
            color=cmap(0.5),
            label="Individual HSS predictions",
            zorder=2,
        )

        # Draw CI bands from innermost to outermost.
        for i, ci in enumerate(ci_levels):
            color = cmap(0.5 - ci / 2)
            # Innermost band fills from the expected-value curve outward;
            # subsequent bands fill from the previous band's edge outward.
            lower_ref = (
                exp_vals.values
                if i == 0
                else ci_array.loc[:, (ci_levels[i - 1], "lower")].values
            )
            upper_ref = (
                exp_vals.values
                if i == 0
                else ci_array.loc[:, (ci_levels[i - 1], "upper")].values
            )
            ax.fill_between(
                ci_array.index,
                lower_ref,
                ci_array.loc[:, (ci, "lower")].values,
                color=color,
                alpha=0.5,
                zorder=3,
            )
            ax.fill_between(
                ci_array.index,
                upper_ref,
                ci_array.loc[:, (ci, "upper")].values,
                color=color,
                alpha=0.5,
                zorder=3,
            )
            # Label only the lower edge to avoid duplicate legend entries.
            ax.plot(
                ci_array.loc[:, (ci, "lower")],
                color=color,
                label=f"{int(ci * 100)}% CI",
                zorder=4,
            )
            ax.plot(
                ci_array.loc[:, (ci, "upper")],
                color=cmap(0.5 + ci / 2),
                zorder=4,
            )

        ax.plot(exp_vals, color=cmap(0.5), label="Expected value", zorder=4)

        # Re-draw scatter on top of the CI bands.
        ax.scatter(
            peak_obs,
            peak_pred.get_expected_values(),
            marker="o",
            s=10,
            color=cmap(0.5),
            zorder=5,
        )

        ax.plot([xmin, xmax], [xmin, xmax], color="blue", linestyle="dashed", zorder=3)

        ax.set_xlabel("Observation (km/s)", fontsize=15)
        ax.set_ylabel("Prediction (km/s)", fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_aspect("equal")
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((xmin, xmax))
        ax.legend(fontsize=10, loc="upper left")

        fig.tight_layout()
        fig.savefig(
            self.figure_dir / "hss_peak_scatter.png",
            bbox_inches="tight",
            dpi=_FIGURE_DPI,
        )

    # ------------------------------------------------------------------
    # Private overlay methods — called exclusively by plot_time_series
    # ------------------------------------------------------------------

    def _add_confidence_interval_overlay(
        self,
        ax: plt.Axes,
        pred: SolarWindPrediction,
        exp_val: pd.Series,
        ci_levels: list[float],
    ) -> None:
        """Draw shaded confidence bands around the expected-value line.

        Bands are drawn from innermost to outermost so that darker shading
        appears near the expected value and lighter shading toward the tails.
        Each band is labelled once; the caller deduplicates legend entries.

        Args:
            ax: The primary matplotlib axes to draw on.
            pred: Prediction slice for the current time window.
            exp_val: Expected SWS values for the current window.
            ci_levels: Coverage probabilities, sorted internally before use.
        """
        if not ci_levels:
            return

        ci_levels = sorted(ci_levels)
        cmap = make_confidence_cmap_from_base(base_cmap_name="gist_heat", n_steps=200)

        # Pre-compute all quantile bounds before the drawing loop to avoid
        # calling the model multiple times for the same quantile level.
        ci_bounds: dict[float, dict[str, pd.Series]] = {
            ci: {
                "lower": pred.get_quantiles(0.5 - ci / 2),
                "upper": pred.get_quantiles(0.5 + ci / 2),
            }
            for ci in ci_levels
        }

        for i, ci in enumerate(ci_levels):
            # The innermost band fills from the expected value outward.
            # Each subsequent band fills from the previous band's edge outward
            # so the fills don't overlap and colours stay visually distinct.
            if i == 0:
                lower_ref = exp_val.values
                upper_ref = exp_val.values
            else:
                prev_ci = ci_levels[i - 1]
                lower_ref = ci_bounds[prev_ci]["lower"].values
                upper_ref = ci_bounds[prev_ci]["upper"].values

            color = cmap(0.5 - ci / 2)
            # zorder=2 keeps fills behind obs/expected lines (zorder=3).
            ax.fill_between(
                ci_bounds[ci]["lower"].index,
                lower_ref,
                ci_bounds[ci]["lower"].values,
                color=color,
                alpha=0.8,
                label=f"{int(ci * 100)}% CI",
                zorder=2,
            )
            ax.fill_between(
                ci_bounds[ci]["upper"].index,
                upper_ref,
                ci_bounds[ci]["upper"].values,
                color=color,
                alpha=0.8,
                zorder=2,
            )

        # Outermost edge lines at zorder=2.5: in front of fills, behind obs.
        outermost = ci_levels[-1]
        ax.plot(
            ci_bounds[outermost]["lower"],
            color=cmap(0.5 - outermost / 2),
            zorder=2.5,
        )
        ax.plot(
            ci_bounds[outermost]["upper"],
            color=cmap(0.5 + outermost / 2),
            zorder=2.5,
        )

    def _add_probability_threshold_overlay(
        self,
        ax2: plt.Axes,
        pred: SolarWindPrediction,
        thresholds: list[int],
    ) -> None:
        """Draw stacked probability-of-exceedance fills for speed thresholds.

        For each threshold T (km/s), the fill shows P(SWS > T | X).  Fills
        are stacked from the highest threshold downward: the highest threshold
        occupies the bottom layer and successive lower thresholds are stacked
        on top, forming a coherent layered chart.

        Args:
            ax2: The secondary (right) matplotlib axes to draw on.
            pred: Prediction slice for the current time window.
            thresholds: Speed thresholds in km/s, e.g. ``[500, 600, 700]``.
        """
        if not thresholds:
            return

        thresholds_asc = sorted(thresholds)
        cmap = make_confidence_cmap_from_base(base_cmap_name="gist_heat", n_steps=200)

        exceedance: dict[int, pd.Series] = {
            t: 1 - pred.get_cdfs(t) for t in thresholds_asc
        }

        # Use the paper's specific colour/alpha mapping for [500, 600, 700];
        # fall back to uniform spacing for any other threshold list.
        if thresholds_asc == [500, 600, 700]:
            cmap_steps = [0.75, 0.9, 0.95]
            alpha_steps = [1.0, 1.0, 1.0]
        else:
            cmap_steps = list(np.linspace(0.5, 0.95, len(thresholds_asc)))
            alpha_steps = [1.0] * len(thresholds_asc)

        thresholds_desc = sorted(thresholds_asc, reverse=True)
        for i, t in enumerate(thresholds_desc):
            bottom = (
                np.zeros(len(exceedance[t]))
                if i == 0
                else exceedance[thresholds_desc[i - 1]].values
            )
            ax2.fill_between(
                exceedance[t].index,
                bottom,
                exceedance[t].values,
                alpha=alpha_steps[i],
                color=cmap(cmap_steps[i]),
                label=f"Probability SWS > {t}",
            )

    def _add_hss_probability_overlay(
        self,
        ax2: plt.Axes,
        pred: SolarWindPrediction,
        mark_above_50: bool,
    ) -> None:
        """Draw the probability that SWS exceeds 450 km/s (HSS proxy).

        450 km/s is the conventional boundary between slow and fast solar
        wind.  A dashed reference line at 0.5 marks the decision boundary.
        When ``mark_above_50=True``, contiguous intervals where P > 0.5 are
        merged (with a 24 h gap tolerance to avoid fragmenting a single HSS
        prediction) and shaded green.

        Args:
            ax2: The secondary (right) matplotlib axes to draw on.
            pred: Prediction slice for the current time window.
            mark_above_50: If ``True``, shade merged above-threshold intervals.
        """
        prob_hss = 1 - pred.get_cdfs(450)
        cmap = make_confidence_cmap_from_base(base_cmap_name="gist_heat", n_steps=200)

        ax2.plot(
            prob_hss.index,
            prob_hss.values,
            alpha=1.0,
            color=cmap(0.75),
            label="Probability SWS > 450",
        )
        ax2.fill_between(
            prob_hss.index,
            prob_hss.values,
            np.zeros(len(prob_hss)),
            alpha=0.5,
            color=cmap(0.75),
        )
        ax2.plot(
            [prob_hss.index[0], prob_hss.index[-1]],
            [0.5, 0.5],
            color="grey",
            linestyle="--",
        )

        if not mark_above_50:
            return

        # Identify contiguous intervals where P > 0.5.
        # A 24 h gap tolerance prevents brief dips below the threshold from
        # splitting what is physically one HSS prediction into many fragments.
        gap_tolerance = pd.Timedelta("24h")
        above = prob_hss > 0.5

        # cumsum of the boolean transition mask produces a block ID that
        # increments each time the series crosses the threshold.
        is_transition = above != above.shift()
        change_points = above.index[is_transition.fillna(True)]
        blocks = pd.DataFrame(
            {"start": change_points, "above": above[change_points].values}
        ).assign(end=lambda df: df["start"].shift(-1, fill_value=prob_hss.index[-1]))
        blocks = blocks[blocks["above"]].reset_index(drop=True)

        if blocks.empty:
            return

        merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        current_start = blocks.loc[0, "start"]
        current_end = blocks.loc[0, "end"]

        for _, row in blocks.iloc[1:].iterrows():
            if row["start"] - current_end <= gap_tolerance:
                current_end = row["end"]
            else:
                merged.append((current_start, current_end))
                current_start, current_end = row["start"], row["end"]
        merged.append((current_start, current_end))

        for i, (seg_start, seg_end) in enumerate(merged):
            ax2.axvspan(
                seg_start,
                seg_end,
                color="green",
                alpha=0.5,
                label="Probability > 50%" if i == 0 else "_nolegend_",
            )

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_empirical_coverage(
        pred: SolarWindPrediction,
        obs: pd.Series,
        quantile_levels: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        """Compute empirical coverage at each nominal confidence level.

        For each level q, the symmetric q-prediction interval is
        ``[Q((1-q)/2), Q((1+q)/2)]``.  The empirical coverage is the
        fraction of observations that fall within this interval.

        Args:
            pred: Probabilistic prediction with a ``get_quantiles`` method.
            obs: Ground-truth observations aligned with ``pred``.
            quantile_levels: Nominal coverage probabilities in (0, 1).

        Returns:
            A 2-tuple ``(quantile_levels, empirical_coverage)`` where
            ``empirical_coverage`` is a list of floats of the same length.
        """
        empirical_coverage = []
        for q in quantile_levels:
            lower = pred.get_quantiles((1 - q) / 2)
            upper = pred.get_quantiles((1 + q) / 2)
            covered = np.mean(
                (obs.values >= lower.values) & (obs.values <= upper.values)
            )
            empirical_coverage.append(float(covered))
        return quantile_levels, empirical_coverage

    @staticmethod
    def _compute_binned_ci(
        pred: SolarWindPrediction,
        obs: pd.Series,
        ci_levels: list[float],
        bin_size: float,
        xmin: float,
        xmax: float,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Compute binned mean expected values and CI bounds.

        Groups predictions into equal-width observation-speed bins and
        computes the mean expected value and mean quantile bounds within each
        bin.  The arrays are extended to ``xmin`` / ``xmax`` so that shaded
        bands fill the full axis range without gaps at the edges.

        Args:
            pred: Prediction object covering all data points to be binned.
            obs: Observed values that determine bin assignment.
            ci_levels: Coverage probabilities for the confidence intervals.
            bin_size: Width of each bin in the same units as ``obs``.
            xmin: Left edge of the first bin and the axis minimum.
            xmax: Right edge of the last bin and the axis maximum.

        Returns:
            A 2-tuple ``(exp_vals, ci_array)`` where ``exp_vals`` is a Series
            of mean expected values and ``ci_array`` is a DataFrame with a
            MultiIndex ``(ci_level, "lower"/"upper")``, both indexed by
            x-position.
        """
        bins = np.arange(xmin, xmax + bin_size, bin_size)
        bin_midpoints = bins[:-1] + bin_size / 2

        multi_index = pd.MultiIndex.from_product(
            [ci_levels, ["lower", "upper"]], names=["ci", "q"]
        )
        ci_array = pd.DataFrame(
            np.zeros((len(bin_midpoints), len(multi_index)), dtype=float),
            index=bin_midpoints,
            columns=multi_index,
        )
        exp_vals = pd.Series(
            np.zeros(len(bin_midpoints), dtype=float), index=bin_midpoints
        )

        bin_start = xmin
        for mp in bin_midpoints:
            bin_end = bin_start + bin_size
            obs_in_bin = (obs >= bin_start) & (obs < bin_end)
            bin_pred = pred.slice(pred.index[obs_in_bin])

            exp_vals.loc[mp] = np.mean(bin_pred.get_expected_values())
            for ci in ci_levels:
                ci_array.loc[mp, (ci, "lower")] = np.mean(
                    bin_pred.get_quantiles(0.5 - ci / 2)
                )
                ci_array.loc[mp, (ci, "upper")] = np.mean(
                    bin_pred.get_quantiles(0.5 + ci / 2)
                )
            bin_start += bin_size

        # Extend arrays to axis limits so shading reaches the plot edges.
        edge_index = [xmin] + list(bin_midpoints) + [xmax]
        exp_vals = pd.concat([exp_vals.iloc[[0]], exp_vals, exp_vals.iloc[[-1]]])
        exp_vals.index = edge_index
        ci_array = pd.concat(
            [ci_array.iloc[[0], :], ci_array, ci_array.iloc[[-1], :]]
        )
        ci_array.index = edge_index

        return exp_vals, ci_array