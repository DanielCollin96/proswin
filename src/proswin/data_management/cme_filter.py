"""Utility for removing coronal mass ejection (CME) intervals from time series.

CMEs produce transient disturbances in the solar wind and are not represented
in the input data of the PROSWIN model. :func:`delete_cmes` masks
both the direct CME window and its recurrent Carrington echo (~27 days later)
from any time series or index.
"""
from datetime import timedelta

import numpy as np
import pandas as pd


def delete_cmes(
    ts: pd.Series | pd.DataFrame | pd.Index,
    cme_list: pd.DataFrame,
    return_mask: bool = False,
    delete_next_rotation: bool = True,
) -> (
    pd.Series
    | pd.DataFrame
    | pd.Index
    | tuple[pd.Series | pd.DataFrame | pd.Index, np.ndarray]
):
    """Remove dates affected by CMEs from a time series or index.

    For each CME in ``cme_list``, all timestamps falling within the CME
    start–end window are masked out. When ``delete_next_rotation=True``,
    the window shifted forward by one Carrington solar rotation (~27
    days, bracketed as 26–28 days to account for rotation-period
    uncertainty) is also masked. This removes the CME's influence
    on the persistence features appears roughly one rotation later.

    Args:
        ts: Input time series or index to filter. Accepts
            ``pd.Series``, ``pd.DataFrame``, or any array-like with a
            datetime index / values (e.g. ``pd.DatetimeIndex``).
        cme_list: DataFrame of CME events with at least ``"start"`` and
            ``"end"`` columns containing datetime values, integer-indexed
            (0-based).
        return_mask: If True, also return the boolean mask of
            non-CME dates (True = kept). Default ``False``.
        delete_next_rotation: If True, additionally mask the Carrington
            echo window (start + 26 days to end + 28 days) for each
            CME. Default ``True``.

    Returns:
        The filtered ``ts`` with CME-affected dates removed, in the
        same type as the input. If ``return_mask=True``, returns a
        2-tuple ``(filtered_ts, mask)`` where ``mask`` is a boolean
        ``np.ndarray`` (True = date retained).
    """
    # Resolve the datetime index from whichever input type was provided
    if isinstance(ts, (pd.Series, pd.DataFrame)):
        idx = ts.index
    else:
        idx = ts

    binary_cme = np.zeros(len(idx), dtype=bool)

    for j in range(cme_list.shape[0]):
        # Mark dates within the direct CME window
        binary_cme |= (
            (cme_list.loc[j, 'start'] <= idx) & (cme_list.loc[j, 'end'] >= idx)
        )
        if delete_next_rotation:
            # Mark the recurrent CME signature one Carrington rotation later.
            # The 26–28 day bracket (rather than a fixed 27.27 days) accounts
            # for variability in the solar rotation period.
            binary_cme |= (
                ((cme_list.loc[j, 'start'] + timedelta(days=26)) <= idx)
                & ((cme_list.loc[j, 'end'] + timedelta(days=28)) >= idx)
            )

    # Invert to get a mask of dates NOT affected by CMEs
    binary_no_cme = np.invert(binary_cme)

    if isinstance(ts, pd.Series):
        ts_without_cme_influence = ts.loc[binary_no_cme]
    elif isinstance(ts, pd.DataFrame):
        ts_without_cme_influence = ts.loc[binary_no_cme, :]
    else:
        ts_without_cme_influence = ts[binary_no_cme]

    if return_mask:
        return ts_without_cme_influence, binary_no_cme
    return ts_without_cme_influence
