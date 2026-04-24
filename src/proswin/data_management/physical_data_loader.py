from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# Solar cycle boundary dates based on NOAA SWPC.
# Each cycle is anchored at its minimum (value = 0.0) and the hour before
# the next minimum (value = 11.0), so that linear interpolation produces a
# continuous, gap-free sawtooth curve spanning 0–11 years per cycle.
# The one-hour offset at each transition (e.g. _CYCLE_24_END is 23:00 on
# the day before _CYCLE_25_START at 00:00) ensures the series goes
# …→ 11.0 → 0.0 →… without any NaN at the boundary.
_CYCLE_24_START: datetime = datetime(2008, 12, 15)      # Solar cycle 24 minimum
_CYCLE_24_END: datetime = datetime(2019, 12, 14, 23)    # Last hour of cycle 24
_CYCLE_25_START: datetime = datetime(2019, 12, 15)      # Solar cycle 25 minimum
_CYCLE_25_END: datetime = datetime(2030, 12, 14, 23)    # Last hour of cycle 25
_CYCLE_FEATURE_RANGE_END: datetime = datetime(2030, 12, 15)  # Upper bound of date range


def _resample_series(series: pd.Series, freq: str) -> pd.Series:
    """Sort, reindex to a complete date range, and linearly interpolate gaps.

    Args:
        series: Time series with a ``DatetimeIndex``.
        freq: Pandas offset alias for the target frequency (e.g. ``'60min'``
            for hourly, ``'MS'`` for month-start).

    Returns:
        Sorted, fully-indexed, and gap-interpolated series at the
        requested frequency.
    """
    series = series.sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    return series.reindex(full_index).interpolate()


def create_solar_cycle_feature() -> pd.Series:
    """Create a linearly interpolated solar cycle year feature.

    Covers solar cycles 24 and 25 (December 2008 – December 2030).
    The value represents elapsed years since the most recent cycle minimum,
    ranging from 0 (at solar minimum) to 11 (just before the next minimum).

    The feature is built by anchoring known cycle boundaries (see module-level
    ``_CYCLE_*`` constants) and interpolating linearly between them. The
    one-hour offset between the end-of-cycle anchor and the start-of-next-cycle
    anchor ensures a gap-free sawtooth transition with no NaN at any boundary.

    Returns:
        Hourly time series named ``'solar_cycle_year'``, covering
        ``_CYCLE_24_START`` to ``_CYCLE_FEATURE_RANGE_END``.
    """
    date_range = pd.date_range(_CYCLE_24_START, _CYCLE_FEATURE_RANGE_END, freq='h')
    solar_cycle_year = pd.Series(np.full(len(date_range), np.nan), index=date_range)

    # Anchor points define the sawtooth shape; interpolate() fills the rest
    solar_cycle_year.loc[_CYCLE_24_START] = 0.0
    solar_cycle_year.loc[_CYCLE_24_END] = 11.0
    solar_cycle_year.loc[_CYCLE_25_START] = 0.0
    solar_cycle_year.loc[_CYCLE_25_END] = 11.0

    solar_cycle_year = solar_cycle_year.interpolate()
    solar_cycle_year.name = 'solar_cycle_year'
    return solar_cycle_year


class SolarWindSpeedLoader:
    """Load and preprocess in-situ solar wind speed observations.

    Reads hourly solar wind speed measurements from a pre-formatted CSV file
    and returns a cleaned, gap-interpolated time series suitable for use as
    a model feature or prediction target. If the primary file is not found,
    falls back to parsing a raw OMNI data file.

    Args:
        data_folder: Directory containing ``solar_wind_speed.csv`` and,
            optionally, ``omni_sws.lst`` as a fallback data source.

    Attributes:
        data_file (Path): Full path to the primary solar wind speed CSV.
        fallback_data_file (Path): Full path to the OMNI fallback ``.lst`` file.
    """

    def __init__(self, data_folder: Path) -> None:
        self.data_file = data_folder / 'solar_wind_speed.csv'
        self.fallback_data_file = data_folder / 'omni_sws.lst'

    def load_data(self) -> pd.Series:
        """Load solar wind speed, resample to hourly, and interpolate short gaps.

        Reads from ``solar_wind_speed.csv`` if available, otherwise falls back
        to parsing the raw OMNI file via :meth:`read_omni`.

        Data gaps of up to 30 consecutive hours are filled by linear time
        interpolation; longer gaps remain as ``NaN`` to avoid introducing
        spurious values across major data outages.

        Returns:
            Hourly solar wind speed series named ``'solar_wind_speed'``.

        Raises:
            FileNotFoundError: If neither ``solar_wind_speed.csv`` nor
                ``omni_sws.lst`` is found in the data folder.
        """
        if self.data_file.exists():
            solar_wind_speed = pd.read_csv(
                self.data_file, index_col=0, parse_dates=[0]
            ).squeeze()
        else:
            if not self.fallback_data_file.exists():
                raise FileNotFoundError(
                    f"Neither '{self.data_file}' nor '{self.fallback_data_file}' "
                    f"were found. Provide at least one solar wind speed data file."
                )
            solar_wind_speed = self.read_omni()

        solar_wind_speed.sort_index(inplace=True)
        solar_wind_speed = solar_wind_speed.resample('60min').asfreq()
        # Interpolate gaps of up to 30 hours; longer outages stay as NaN
        solar_wind_speed = solar_wind_speed.interpolate(method='time', limit=30)
        solar_wind_speed.name = 'solar_wind_speed'
        return solar_wind_speed

    def read_omni(self) -> pd.Series:
        """Parse a raw OMNI solar wind speed file in fixed-width list format.

        Expects columns in the order: year, day-of-year, hour, speed [km/s].
        The OMNI fill value 9999.0 is replaced with ``NaN``.

        Returns:
            Solar wind speed series indexed by datetime, with missing values
            linearly interpolated.
        """
        raw = pd.read_table(
            self.fallback_data_file, header=None, delim_whitespace=True
        )

        years = raw.iloc[:, 0].astype(int)
        days = raw.iloc[:, 1].astype(int)
        hours = raw.iloc[:, 2].astype(int)
        sws = raw.iloc[:, 3].astype(float)

        # Replace OMNI fill value with NaN
        sws[sws == 9999.0] = np.nan

        # Reconstruct datetime index from year + day-of-year + hour
        index = [
            datetime(year, 1, 1, hour) + timedelta(days=int(day) - 1)
            for year, day, hour in zip(years, days, hours)
        ]

        return pd.Series(sws.values, index=index).interpolate()

class SunspotNumberLoader:
    """Load and preprocess monthly sunspot number data.

    Reads the monthly sunspot number from a pre-formatted CSV file
    and returns a monthly time series resampled to a complete date range.
    Falls back to parsing the raw SILSO file if the primary CSV is absent.

    Args:
        data_folder: Directory containing ``sunspot_number.csv`` and
            optionally ``silso_monthly_sunspot_number.csv`` as a fallback.

    Attributes:
        data_file (Path): Full path to the primary sunspot number CSV.
        fallback_data_file (Path): Full path to the SILSO raw data file.
    """

    def __init__(self, data_folder: Path) -> None:
        self.data_file = data_folder / 'sunspot_number.csv'
        self.fallback_data_file = data_folder / 'silso_monthly_sunspot_number.csv'

    def load_data(self) -> pd.Series:
        """Load monthly sunspot numbers, reindex to a complete range, and interpolate.

        Falls back to the raw SILSO file if the primary CSV is missing.

        Returns:
            Monthly sunspot number series named ``'sunspot_number'``.
        """
        try:
            sunspot_data = pd.read_csv(
                self.data_file, index_col=0, parse_dates=[0]
            ).squeeze()
        except FileNotFoundError:
            sunspot_data = self._read_silso()

        sunspot_series = _resample_series(sunspot_data, freq='MS')
        sunspot_series.name = 'sunspot_number'
        return sunspot_series

    def _read_silso(self) -> pd.Series:
        """Parse the raw SILSO semicolon-delimited monthly sunspot file.

        The SILSO format has no header; columns are accessed by position:
        col 0 = year, col 1 = month, col 3 = monthly mean sunspot number.

        Returns:
            Monthly sunspot numbers indexed by the first day of each month.
        """
        raw = pd.read_csv(self.fallback_data_file, sep=';', index_col=None)
        year = raw.iloc[:, 0].astype(int)
        month = raw.iloc[:, 1].astype(int)
        sunspot_number = raw.iloc[:, 3].astype(float)
        dates = pd.to_datetime({'year': year, 'month': month, 'day': 1})
        return pd.Series(sunspot_number.values, index=dates)


class HeliosphericLatitudeLoader:
    """Load and preprocess Earth's heliospheric latitude time series.

    Reads the heliocentric Earth ecliptic latitude from a pre-formatted CSV
    file and returns an hourly time series resampled to a complete date range.

    Args:
        data_folder: Directory containing ``heliospheric_latitude.csv``.

    Attributes:
        data_file (Path): Full path to the heliospheric latitude CSV.
    """

    def __init__(self, data_folder: Path) -> None:
        self.data_file = data_folder / 'heliospheric_latitude.csv'

    def load_data(self) -> pd.Series:
        """Load heliospheric latitude, reindex to hourly, and interpolate gaps.

        Returns:
            Hourly heliospheric latitude series named
            ``'heliospheric_latitude'``.
        """
        hl_data = pd.read_csv(
            self.data_file, parse_dates=[0], index_col=0
        ).squeeze()
        # Explicitly cast to DatetimeIndex to guard against pandas returning
        # an object-dtype Index in edge cases (e.g. mixed-type CSV columns)
        hl_data.index = pd.DatetimeIndex(hl_data.index)

        hl_data = _resample_series(hl_data, freq='60min')
        hl_data.name = 'heliospheric_latitude'
        return hl_data
