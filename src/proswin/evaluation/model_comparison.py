import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Preset = Literal["image_only", "full_model", "literature", "everything", "all_proswin"]
ResultsType = pd.DataFrame | dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

# Each preset defines:
#   figsize       : figure size in inches
#   xlim / ylim   : axis limits (None = auto)
#   inset         : whether to show an inset zoom panel
#   inset_xlim/ylim: inset axis limits
#   contour_levels: (min, max) for contourf levels
#   legend_loc    : legend location string
#   grouped       : whether results is expected to be a dict of groups
#   group_colors  : list of colors for each group (grouped presets only)
#   label_fontsize: annotation fontsize

_PRESET_CFG: dict[str, dict] = {
    "image_only": dict(
        figsize=(6.4, 4.8),
        xlim=(40.5, 44.5),
        ylim=(42, 49),
        inset=True,
        contour_levels=(40, 46),
        legend_loc="lower right",
        grouped=False,
        label_fontsize=10,
        label_offset=(0.01, 0.01),  # as fractions of axis width/height
    ),
    "full_model": dict(
        figsize=(6.4, 4.8),
        xlim=(40.5, 44.5),
        ylim=(42, 49),
        inset=True,
        contour_levels=(40, 46),
        legend_loc="lower right",
        grouped=False,
        label_fontsize=10,
        label_offset=(0.01, 0.01),  # as fractions of axis width/height
    ),
    "literature": dict(
        figsize=(1.2 * 6.4, 4.8),
        xlim=(38, 60),
        ylim=(38, 80),
        inset=False,
        contour_levels=(35, 67),
        legend_loc="upper right",
        grouped=False,
        label_fontsize=10,
        label_offset=(0.01, 0.01),  # as fractions of axis width/height
    ),
    "everything": dict(
        figsize=(1.2 * 6.4, 4.8),
        xlim=(38, 60),
        ylim=(38, 80),
        inset=False,
        contour_levels=(35, 67),
        legend_loc="upper left",
        grouped=True,
        group_colors=["blue", "black", "green"],
        label_fontsize=7,
        label_offset=(0.01, 0.01),  # as fractions of axis width/height
    ),
    "all_proswin": dict(
        figsize=(6.4, 4.8),
        xlim=(40.5, 44.5),
        ylim=(42, 49),
        inset=False,
        contour_levels=(40, 46),
        legend_loc="lower right",
        grouped=True,
        group_colors=["blue", "black"],
        label_fontsize=7,
        label_offset=(0.01, 0.01),  # as fractions of axis width/height
    ),
    None: dict(
        figsize=(6.4, 4.8),
        xlim=None,
        ylim=None,
        inset=False,
        contour_levels=(35, 67),
        legend_loc="lower right",
        grouped=False,
        label_fontsize=10,
        label_offset=(0.01, 0.01),  # as fractions of axis width/height
    ),
}

# Per-preset, per-label annotation text overrides.
# Keys are (preset, group_or_None, label_text).
# Values are the display string to use instead of the raw label.
_LABEL_TEXT_OVERRIDES: dict[tuple, str] = {
    ("literature", None, "Bailey"): "Bailey et al. (Space Weather, 2021)",
    ("literature", None, "Brown"): "Brown et al. (Space Weather, 2022)",
    ("literature", None, "Collin PR"): "PR, Collin et al. (Space Weather, 2025)",
    ("literature", None, "Collin DT"): "PR+DT, Collin et al. (Space Weather, 2025)",
}

# Per-preset, per-label skip rules.
# If (preset, group, label) is in this set, the annotation is suppressed.
_LABEL_SKIP_RULES: set[tuple] = {
    ("everything", "image_only", None),   # skip all image_only except HMI
    ("everything", "full_model", None),   # skip all full_model except HMI
}

# Labels to always show despite group-level skip rules (preset, group, label).
_LABEL_FORCE_SHOW: set[tuple] = {
    ("everything", "image_only", "HMI"),
    ("everything", "full_model", "HMI"),
}

# Keys are (preset, group_or_None, label_text).
# Values are multiples of the default shift.
# Only entries that deviate from the preset default need to be listed.
_LABEL_OFFSET_OVERRIDES: dict[tuple, tuple[float, float]] = {
    ("full_model", None, "193-211-HMI"): (2, -3),
    ("full_model", None, "211"): (2, -3),
    ("full_model", None, "171-193-211"): (2, -4),
    ("full_model", None, "171-193-HMI"): (2, 0.5),
    ("full_model", None, "HMI"): (4, 2),
    ("image_only", None, "211"): (2, -3),
    ("image_only", None, "171-211"): (-4, 3),
    ("image_only", None, "HMI"): (4, 2),
    ("literature", None, "Persistence"): (-7, 1.5),
    ("literature", None, "CH baseline"): (2, -2),
    ("literature", None, "Collin DT"): (-7, 1.5),
    ("literature", None, "Constant sigma lognormal"): (2, -2),
    ("all_proswin", "image_only", "211"): (2, -1),
    ("all_proswin", "image_only", "193-211-HMI"): (2, -1),
    ("all_proswin", "full_model", "193-211-HMI"): (2, -1),
    ("all_proswin", "full_model", "171-193-HMI"): (2, -0.5),
    ("all_proswin", "full_model", "171-193-211"): (-5, -4),
    ("everything", "literature", "CH baseline"): (2, -1),
    ("everything", "literature", "Constant sigma lognormal"): (2, -1),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def weighted_harmonic_mean(
    x: np.ndarray,
    y: np.ndarray,
    w_x: float = 2.0,
    w_y: float = 1.0,
) -> np.ndarray:
    """Compute the weighted harmonic mean of two arrays.

    Args:
        x: First array of values.
        y: Second array of values, same shape as ``x``.
        w_x: Non-negative weight for ``x``. Defaults to ``2.0``.
        w_y: Non-negative weight for ``y``. Defaults to ``1.0``.

    Returns:
        Weighted harmonic mean, same shape as the inputs.
    """
    return (w_x + w_y) / ((w_x / x) + (w_y / y))


def prediction_score(
    timeline_error: float,
    peak_error: float,
    w_timeline: float = 2.0,
    w_peak: float = 1.0,
) -> float:
    """Compute the weighted harmonic mean prediction score.

    Args:
        timeline_error: Overall timeline error (must be positive).
        peak_error: Peak event error (must be positive).
        w_timeline: Non-negative weight for ``timeline_error``. Defaults
            give timeline twice the weight of peak.
        w_peak: Non-negative weight for ``peak_error``.

    Returns:
        Weighted harmonic mean of the two errors.

    Raises:
        ValueError: If weights are negative or errors are non-positive.
    """
    if w_timeline < 0 or w_peak < 0:
        raise ValueError("Weights must not be negative.")
    if timeline_error <= 0 or peak_error <= 0:
        raise ValueError("Errors must be positive.")
    return float(weighted_harmonic_mean(timeline_error, peak_error, w_timeline, w_peak))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ComparisonPlotter:
    """Scatter-plot comparison of prediction scores across models.

    Args:
        model_result_table: Either a single :class:`pd.DataFrame` (rows =
            metrics, columns = models) or a dict mapping group names to
            such DataFrames.
        figure_path: Directory where figures are saved.
        save_label: Optional suffix appended to saved filenames (preceded
            by ``_``).
    """

    def __init__(
        self,
        model_result_table: ResultsType,
        figure_path: str | Path,
        save_label: str | None = None,
    ) -> None:
        self.results = model_result_table
        self.figure_path = Path(figure_path)
        self.save_label = f"_{save_label}" if save_label else ""

    def plot(self, show: bool = True) -> None:
        """Render all comparison plots.

        Args:
            show: If ``True``, call ``plt.show()`` after plotting.
        """
        self.pred_score_comparison()
        if show:
            plt.show()

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_grouped(
        self, metric: str
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
        """Extract errors and pred_scores for dict-of-DataFrames results.

        Args:
            metric: Metric name used to retrieve ``ps_{metric}``,
                ``{metric}``, and ``peak_{metric}`` rows from each group
                DataFrame.

        Returns:
            A 2-tuple ``(errors, pred_scores)`` where ``errors`` maps each
            group name to an ``(n_models, 2)`` DataFrame and
            ``pred_scores`` maps each group name to a Series of valid
            prediction scores.
        """
        errors: dict[str, pd.DataFrame] = {}
        pred_scores: dict[str, pd.Series] = {}

        for group, df in self.results.items():
            ps = df.loc[f"ps_{metric}", :]
            valid = ~np.isnan(ps.values)
            pred_scores[group] = ps.iloc[valid]
            errors[group] = df.loc[[metric, f"peak_{metric}"], :].transpose().iloc[valid]

        return errors, pred_scores

    def _extract_single(
        self, metric: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract errors and pred_scores for a single DataFrame result.

        Args:
            metric: Metric name used to retrieve ``ps_{metric}``,
                ``{metric}``, and ``peak_{metric}`` rows.

        Returns:
            A 2-tuple ``(errors, pred_scores)`` where ``errors`` is an
            ``(n_models, 2)`` DataFrame and ``pred_scores`` is a Series
            of valid prediction scores.
        """
        ps = self.results.loc[f"ps_{metric}", :]
        valid = ~np.isnan(ps.values)
        pred_scores = ps.iloc[valid]
        errors = self.results.loc[[metric, f"peak_{metric}"], :].transpose().iloc[valid]
        return errors, pred_scores

    def _all_xy(
        self,
        errors: dict[str, pd.DataFrame] | pd.DataFrame,
        grouped: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return flat x/y arrays across all groups (for contour range).

        Args:
            errors: Either a dict of group DataFrames (grouped mode) or a
                single DataFrame (ungrouped mode).
            grouped: Whether ``errors`` is a dict of DataFrames.

        Returns:
            A 2-tuple ``(xs, ys)`` of 1-D NumPy arrays concatenated from
            all groups.
        """
        if grouped:
            xs = np.concatenate([e.iloc[:, 0].values for e in errors.values()])
            ys = np.concatenate([e.iloc[:, 1].values for e in errors.values()])
        else:
            xs = errors.iloc[:, 0].values
            ys = errors.iloc[:, 1].values
        return xs, ys

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    def _should_skip_label(
        self, preset: str | None, group: str | None, label: str
    ) -> bool:
        """Return True if this label should be suppressed.

        Args:
            preset: Current plot preset name.
            group: Group key if in grouped mode, else ``None``.
            label: Raw label string from the results index.

        Returns:
            ``True`` if the annotation should be omitted.
        """
        if (preset, group, label) in _LABEL_FORCE_SHOW:
            return False
        # Check group-level skip (None means "all labels in this group")
        return (preset, group, None) in _LABEL_SKIP_RULES

    def _resolve_label_text(
        self, preset: str | None, group: str | None, label: str
    ) -> str:
        """Return display text for a label, applying overrides if defined.

        Args:
            preset: Current plot preset name.
            group: Group key if in grouped mode, else ``None``.
            label: Raw label string from the results index.

        Returns:
            Override string from ``_LABEL_TEXT_OVERRIDES`` if present,
            otherwise ``label`` unchanged.
        """
        return _LABEL_TEXT_OVERRIDES.get((preset, group, label), label)

    def _collect_annotations(
        self,
        ax: plt.Axes,
        axins: plt.Axes | None,
        preset: str | None,
        errors: dict[str, pd.DataFrame] | pd.DataFrame,
        pred_scores: dict[str, pd.Series] | pd.Series,
        grouped: bool,
        fontsize: int,
    ) -> None:
        """Place annotations on ax using default or per-label offsets.

        Args:
            ax: Main axes to annotate.
            axins: Inset axes to annotate in parallel, or ``None``.
            preset: Current plot preset name (keys ``_PRESET_CFG``).
            errors: Per-group or flat DataFrame of x/y error values.
            pred_scores: Per-group or flat Series of prediction scores,
                used to iterate over labels.
            grouped: Whether ``errors`` and ``pred_scores`` are dicts.
            fontsize: Annotation font size in points.
        """
        cfg = _PRESET_CFG[preset]
        dx_default_fraction, dy_default_fraction = cfg["label_offset"]
        dx_default = (ax.get_xlim()[1] - ax.get_xlim()[0]) * dx_default_fraction
        dy_default = (ax.get_ylim()[1] - ax.get_ylim()[0]) * dy_default_fraction

        def _place(label: str, err: np.ndarray, group: str | None) -> None:
            if self._should_skip_label(preset, group, label):
                return
            display = self._resolve_label_text(preset, group, label)
            dx, dy = _LABEL_OFFSET_OVERRIDES.get(
                (preset, group, label),
                (2, 1),
            )
            ax.annotate(
                display,
                (err[0] + dx * dx_default, err[1] + dy * dy_default),
                fontsize=fontsize,
            )
            if axins is not None:
                axins.annotate(
                    display,
                    (err[0] + dx * dx_default, err[1] + dy * dy_default),
                    fontsize=fontsize,
                )

        if grouped:
            for group in self.results.keys():
                for label, err in zip(pred_scores[group].index, errors[group].values):
                    _place(label, err, group)
        else:
            for label, err in zip(pred_scores.index, errors.values):
                _place(label, err, None)

    # ------------------------------------------------------------------
    # Main plot method
    # ------------------------------------------------------------------

    def pred_score_comparison(
        self,
        metric: Literal["crps", "rmse"] = "crps",
        preset: Preset | None = None,
    ) -> None:
        """Render and save a prediction-score scatter comparison plot.

        Args:
            metric: Scoring metric to use. One of ``"crps"`` or
                ``"rmse"``.
            preset: Named layout preset. ``None`` uses automatic axis
                limits.
        """
        cfg = _PRESET_CFG[preset]
        grouped = cfg["grouped"]

        # --- Extract data -------------------------------------------------
        if grouped:
            if not isinstance(self.results, dict):
                raise TypeError(
                    f"Preset '{preset}' expects results to be a dict of DataFrames."
                )
            errors, pred_scores = self._extract_grouped(metric)
        else:
            if not isinstance(self.results, pd.DataFrame):
                raise TypeError(
                    f"Preset '{preset}' expects results to be a single DataFrame."
                )
            errors, pred_scores = self._extract_single(metric)

        errors_x, errors_y = self._all_xy(errors, grouped)

        # --- Figure and axes ---------------------------------------------
        fig, ax = plt.subplots(figsize=cfg["figsize"])

        if cfg["xlim"]:
            ax.set_xlim(*cfg["xlim"])
        if cfg["ylim"]:
            ax.set_ylim(*cfg["ylim"])
        ax.grid(True, alpha=0.5, zorder=0)

        # --- Scatter -----------------------------------------------------
        if grouped:
            for color, group in zip(cfg["group_colors"], self.results.keys()):
                label = f"{group} models" if preset == "all_proswin" else group
                ax.scatter(
                    errors[group].iloc[:, 0],
                    errors[group].iloc[:, 1],
                    c=color,
                    label=label,
                )
        else:
            ax.scatter(errors_x, errors_y, c="black", label="Models")

        # --- Inset panel -------------------------------------------------
        axins = None
        if cfg.get("inset"):
            outlier_x = [errors_x.max() - 1, errors_x.max() + 1]
            outlier_y = [errors_y.max() - 1, errors_y.max() + 1]
            axins = inset_axes(
                ax, width="20%", height="20%", loc="upper right", borderpad=0
            )
            axins.scatter(errors_x, errors_y, c="black")
            axins.set_xlim(*outlier_x)
            axins.set_ylim(*outlier_y)
            axins.grid(True, alpha=0.5)

        # --- Contour / isoline -------------------------------------------
        x_range = np.linspace(errors_x.min() - 20, errors_x.max() + 20, 100)
        y_range = np.linspace(errors_y.min() - 20, errors_y.max() + 20, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = weighted_harmonic_mean(X, Y)

        levels = np.linspace(*cfg["contour_levels"], 100)
        cs_filled = ax.contourf(X, Y, Z, levels=levels, cmap="plasma", zorder=0, alpha=0.7)

        cbar = plt.colorbar(cs_filled, ax=ax)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Prediction Score", fontsize=15)
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()

        if grouped:
            best_score = min(ps.min() for ps in pred_scores.values())
        else:
            best_score = pred_scores.min()

        cs = ax.contour(
            X, Y, Z, levels=[best_score], colors="black", linestyles="--", zorder=0
        )
        contour_artists = cs.legend_elements()[0]

        # --- Annotations --------------------
        self._collect_annotations(
            ax, axins, preset, errors, pred_scores,
            grouped=grouped,
            fontsize=cfg["label_fontsize"],
        )

        # --- Axis labels -------------------------------------------------
        _metric_labels = {
            "rmse": ("Timeline RMSE (km/s)", "HSS peak value RMSE (km/s)"),
            "crps": ("Timeline CRPS (km/s)", "HSS peak value CRPS (km/s)"),
        }
        if metric not in _metric_labels:
            raise NotImplementedError(f"Metric '{metric}' is not supported.")
        xlabel, ylabel = _metric_labels[metric]
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)

        # --- Legend ------------------------------------------------------
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        by_label["Best PS isoline"] = contour_artists[0]
        ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc=cfg["legend_loc"])

        # --- Save --------------------------------------------------------
        fig.tight_layout()
        fig.savefig(
            self.figure_path / f"ps_{metric}_comparison{self.save_label}.png",
            bbox_inches="tight",
            dpi=500,
        )
