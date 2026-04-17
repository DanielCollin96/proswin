import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Subset

from proswin.data_management.distribution_transformer import DistributionTransformer
from proswin.data_management.physical_data_loader import (
    HeliosphericLatitudeLoader,
    SolarWindSpeedLoader,
    SunspotNumberLoader,
    create_solar_cycle_feature,
)


class SolarDataset(Dataset):
    """PyTorch Dataset for multi-modal solar wind speed forecasting.

    Combines SDO solar images with scalar physical time-series features
    (solar wind speed persistence, sunspot number, heliospheric latitude,
    solar cycle year) to form multi-modal input samples. On first use,
    availability and value tables are computed by scanning all data files;
    subsequent runs reload these tables from cache if
    ``precompute_and_reuse_dataset`` is enabled.

    Args:
        solar_image_dir: Root directory with one sub-folder per SDO channel
            (e.g. ``aia_171``, ``aia_193``, ``aia_211``, ``hmi``).
        solar_event_dir: Directory containing solar event CSVs
            (``cme_list.csv``, ``hss_list.csv``, ``enhancement_list.csv``).
        physical_features_dir: Directory containing physical feature CSVs.
        ml_data_dir: Directory used to cache pre-computed feature files.
        dataset_interval: ``[start, end]`` UTC datetimes defining the full
            dataset window. Defaults to June 2010 – June 2024.
        channels: SDO channel names to load (e.g. ``['aia_171', 'aia_193']``).
            Defaults to ``['aia_171', 'aia_211']``.
        physical_input_data: Physical feature names to include. Defaults to
            all four features.
        solar_wind_speed_persistence_interval: ``[start_days, end_days]``
            defining the persistence lag window. Defaults to ``[28, 26]``.
        solar_wind_speed_persistence_cadence: Hourly step between persistence
            lag samples. Defaults to ``1``.
        sunspot_number_months: Number of preceding months of sunspot data to
            include as features. Defaults to ``12``.
        precompute_and_reuse_dataset: If ``True``, save computed availability
            and feature tables to ``ml_data_dir`` and reload them on
            subsequent runs to avoid repeated file scanning. Defaults to
            ``True``.
        rank: Process rank for distributed training (used only for logging
            so that only rank-0 emits progress messages). Defaults to ``0``.
        logger: Optional logger for progress messages. If ``None``, no
            messages are emitted.

    Attributes:
        data (pd.DataFrame): Full assembled feature table; rows are valid
            timesteps, columns are target + image paths + physical features.
        target (pd.Series): Solar wind speed at each valid timestep (km/s).
        target_dates (pd.DatetimeIndex): Timestamps for all valid samples.
        target_values (np.ndarray): Target values as a NumPy array.
        physical_features (pd.DataFrame): Physical feature columns only.
        missing_dates (pd.DatetimeIndex): Timestamps within the dataset
            interval where at least one feature is unavailable.
        weights (pd.Series): Per-sample loss weights (all 1.0 by default).
        num_images (int): Number of image channels used.
        num_physical_features (int): Total number of physical feature columns.
    """

    def __init__(
        self,
        solar_image_dir: Path,
        solar_event_dir: Path,
        physical_features_dir: Path,
        ml_data_dir: Path,
        dataset_interval: list[datetime] | None = None,
        channels: list[str] | None = None,
        physical_input_data: list[str] | None = None,
        solar_wind_speed_persistence_interval: list[int] | None = None,
        solar_wind_speed_persistence_cadence: int = 1,
        sunspot_number_months: int = 12,
        precompute_and_reuse_dataset: bool = True,
        rank: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        if dataset_interval is None:
            dataset_interval = [datetime(2010, 6, 1), datetime(2024, 6, 30, 23)]
        if channels is None:
            channels = ['aia_171', 'aia_211']
        if physical_input_data is None:
            physical_input_data = [
                'solar_wind_speed_persistence',
                'sunspot_number',
                'solar_cycle_year',
                'heliospheric_latitude',
            ]
        if solar_wind_speed_persistence_interval is None:
            solar_wind_speed_persistence_interval = [28, 26]

        self.solar_image_dir = solar_image_dir
        self.solar_event_dir = solar_event_dir
        self.physical_features_dir = physical_features_dir
        self.ml_data_dir = ml_data_dir
        self.dataset_interval = dataset_interval
        self.channels = channels
        self.physical_input_data = physical_input_data
        self.solar_wind_speed_persistence_interval = solar_wind_speed_persistence_interval
        self.solar_wind_speed_persistence_cadence = solar_wind_speed_persistence_cadence
        self.sunspot_number_months = sunspot_number_months
        self.precompute_and_reuse_dataset = precompute_and_reuse_dataset
        self.logger = logger
        self.rank = rank

        self._check_paths()
        self._load_physical_features()

        # Initialise dataset containers (populated by _create_dataset below)
        self.data = pd.DataFrame(dtype='float32')
        self.target_dates = pd.DatetimeIndex([])
        self.target_values: list = []
        self.target = pd.Series(dtype='float32')
        self.missing_dates: list = []
        self.physical_features = pd.DataFrame(dtype='float32')
        self.weights = pd.Series(dtype='float32')

        if self.logger is not None and self.rank == 0:
            self.logger.info('Creating dataset...')
        self._create_dataset()
        # Initialise uniform per-sample weights (all 1.0); override with
        # set_loss_function_weights() before training to up-weight HSS events
        self.set_loss_function_weights()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _check_paths(self) -> None:
        """Verify that all required data files and directories exist.

        Raises:
            FileNotFoundError: If a required image channel directory or
                ``solar_wind_speed.csv`` is missing.
            NotImplementedError: If any solar event list CSV is missing
                (on-the-fly generation from raw data is not implemented).
        """
        for channel in self.channels:
            if not (self.solar_image_dir / channel).exists():
                raise FileNotFoundError(
                    f"Channel directory '{channel}' not found in "
                    f"'{self.solar_image_dir}'. Ensure folders named "
                    f"'aia_171', 'aia_193', 'aia_211', 'hmi' exist."
                )

        for file in ['cme_list.csv', 'hss_list.csv', 'enhancement_list.csv']:
            if not (self.solar_event_dir / file).exists():
                msg = (
                    f"'{file}' not found in '{self.solar_event_dir}'. "
                    f"Fallback to computing event lists from scratch."
                )
                if self.logger is not None:
                    self.logger.warning(msg)
                raise NotImplementedError(msg)

        # Solar wind speed is always required (it is also the forecast target)
        if not (self.physical_features_dir / 'solar_wind_speed.csv').exists():
            raise FileNotFoundError(
                f"'solar_wind_speed.csv' not found in "
                f"'{self.physical_features_dir}'."
            )

    def _load_physical_features(self) -> None:
        """Load physical feature time series from disk into instance attributes."""
        # Solar wind speed doubles as the forecast target, so always load it
        self.solar_wind_speed = SolarWindSpeedLoader(
            data_folder=self.physical_features_dir
        ).load_data()

        if 'heliospheric_latitude' in self.physical_input_data:
            self.heliospheric_latitude = HeliosphericLatitudeLoader(
                data_folder=self.physical_features_dir
            ).load_data()
        if 'sunspot_number' in self.physical_input_data:
            self.monthly_sunspot_number = SunspotNumberLoader(
                data_folder=self.physical_features_dir
            ).load_data()
        if 'solar_cycle_year' in self.physical_input_data:
            self.solar_cycle_year = create_solar_cycle_feature()

    # ------------------------------------------------------------------
    # Image normalisation
    # ------------------------------------------------------------------

    def _normalize_aia(self, image: np.ndarray, upper_limit: float) -> np.ndarray:
        """Normalise an AIA EUV image using the Jarolim et al. scheme.

        Applies min–max scaling to [0, 1] with hard clipping, an inverse
        hyperbolic sine (asinh) stretch, then linear rescaling to [-1, 1].
        The lower intensity limit is fixed at 0 for all AIA channels.

        Args:
            image: Raw AIA image array.
            upper_limit: Channel-specific upper intensity limit for
                min–max scaling.

        Returns:
            Normalised image in the range [-1, 1].
        """
        scaled = image / upper_limit  # lower_limit is 0 for all AIA channels
        clipped = np.clip(scaled, 0.0, 1.0)
        # Asinh stretch enhances faint structures; parameter 0.005 from Jarolim et al.
        stretched = np.asinh(clipped / 0.005) / np.asinh(1 / 0.005)
        return 2 * stretched - 1  # rescale [0, 1] → [-1, 1]

    def _normalize_hmi(self, image: np.ndarray) -> np.ndarray:
        """Normalise an HMI magnetogram using the Jarolim et al. scheme.

        Applies min–max scaling over the fixed range [-100, 100] Gauss with
        hard clipping, then rescales to [-1, 1].

        Args:
            image: Raw HMI line-of-sight magnetogram in Gauss.

        Returns:
            Normalised magnetogram in the range [-1, 1].
        """
        lower_limit, upper_limit = -100.0, 100.0
        scaled = (image - lower_limit) / (upper_limit - lower_limit)
        clipped = np.clip(scaled, 0.0, 1.0)
        return 2 * clipped - 1

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample by integer index.

        Loads the image file(s) for the indexed timestep, applies channel-
        specific normalisation, stacks them into a 3-channel tensor (padding
        by channel duplication when fewer than 3 channels are used), and
        returns all inputs and targets as a dictionary of tensors.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Dictionary with the following keys:

            - ``'images'`` (torch.Tensor, float32, shape ``[3, H, W]``):
              Stacked and normalised image channels.
            - ``'target'`` (torch.Tensor, float32, shape ``[1]``): Solar wind
              speed target value.
            - ``'target_date'`` (torch.Tensor, float64, shape ``[1]``): Target
              timestamp as seconds since the Unix epoch.
            - ``'loss_function_weights'`` (torch.Tensor, float32, shape
              ``[1]``): Per-sample loss weight.
            - ``'physical_features'`` (torch.Tensor, float32, shape ``[F]``):
              Physical feature vector of length F.

        Raises:
            ValueError: If an unknown SDO channel is requested or more than
                3 channels are specified.
        """
        item = self.data.iloc[idx, :]
        target_date = self.data.index[idx]

        # Maps channel name → normalisation function with channel-specific
        # upper-limit constants from Jarolim et al.
        channel_map: dict[str, Any] = {
            'aia_171': lambda img: self._normalize_aia(img, 6457.50),
            'aia_193': lambda img: self._normalize_aia(img, 7757.31),
            'aia_211': lambda img: self._normalize_aia(img, 6539.0),
            'hmi': self._normalize_hmi,
        }

        loaded_channels = []
        for channel in self.channels:
            if channel not in channel_map:
                raise ValueError(f"Unknown SDO channel: '{channel}'.")
            norm_fn = channel_map[channel]
            img = np.load(item.loc[channel])
            loaded_channels.append(norm_fn(img))

        # The Swin Transformer encoder expects a 3-channel (RGB-like) input.
        # Duplicate channels when fewer than 3 are provided so the architecture
        # always receives a fixed-size tensor.
        if len(loaded_channels) == 1:
            image = np.stack([loaded_channels[0]] * 3, axis=0)
        elif len(loaded_channels) == 2:
            # First channel fills slot 0; second channel fills slots 1 and 2
            image = np.stack(
                [loaded_channels[0], loaded_channels[1], loaded_channels[1]], axis=0
            )
        elif len(loaded_channels) == 3:
            image = np.stack(loaded_channels, axis=0)
        else:
            raise ValueError('Select between 1 and 3 SDO channels.')

        image_tensor = torch.from_numpy(image).float()

        target = np.expand_dims(item.loc['solar_wind_speed'], axis=0)
        # Column layout in self.data: [solar_wind_speed | channels... | physical_features...]
        # Skip col 0 (target) and cols 1..num_images (image paths) to reach features
        physical_features = item.iloc[self.num_images + 1:].astype(float)
        loss_function_weights = np.expand_dims(self.weights.iloc[idx], axis=0)
        target_timestamp = target_date.timestamp()  # seconds since Unix epoch

        return {
            'images': image_tensor,
            'target': torch.tensor(target, dtype=torch.float32),
            'target_date': torch.tensor(target_timestamp, dtype=torch.float64),
            'loss_function_weights': torch.tensor(
                loss_function_weights, dtype=torch.float32
            ),
            'physical_features': torch.tensor(
                physical_features.values, dtype=torch.float32
            ),
        }

    # ------------------------------------------------------------------
    # Data availability scanning
    # ------------------------------------------------------------------

    def _check_images(self, folder: Path, key: str) -> None:
        """Scan a directory for ``.npy`` image files and register their timestamps.

        For each valid ``.npy`` file the filename is parsed to extract the
        observation timestamp, which is then marked as available in
        ``self.available_datapoints`` and stored as a file path string in
        ``self.datapoint_values``. Files that cannot be parsed are silently
        skipped.

        Args:
            folder: Directory to scan for ``.npy`` files.
            key: Column name in ``available_datapoints`` and
                ``datapoint_values`` under which to register results.
        """
        if os.path.isdir(folder):
            for file_name in os.listdir(folder):
                if not file_name.endswith('.npy'):
                    continue
                try:
                    _, date_str, time_str = file_name.split('_')
                    timestamp = datetime.strptime(
                        f"{date_str}_{time_str[:4]}", '%Y-%m-%d_%H:%M'
                    )
                    self.available_datapoints.loc[timestamp, key] = True
                    self.datapoint_values.loc[timestamp, key] = str(
                        folder / file_name
                    )
                except Exception:
                    continue

    def _check_sunspot_number(self, current_month: datetime) -> None:
        """Register sunspot number availability for all hours in a given month.

        If the monthly sunspot number is not NaN, marks every hourly
        timestep in the month as available and fills in the constant monthly
        value in ``self.datapoint_values``.

        Args:
            current_month: First day of the month to process.
        """
        if not np.isnan(self.monthly_sunspot_number.loc[current_month]).squeeze():
            idx = pd.date_range(
                start=current_month,
                end=current_month + relativedelta(months=1) - timedelta(hours=1),
                freq='60min',
            )
            self.available_datapoints.loc[idx, 'sunspot_number'] = True
            self.datapoint_values.loc[idx, 'sunspot_number'] = (
                self.monthly_sunspot_number.loc[current_month]
            )

    def _check_available_data(self) -> None:
        """Scan image directories and physical feature time series for data availability.

        Iterates over the dataset interval month by month. For each month,
        scans the corresponding image subdirectories and registers sunspot
        availability. Scalar features (solar wind speed, heliospheric
        latitude, solar cycle year) are registered directly from the loaded
        time series. Results are stored in ``self.available_datapoints``
        (boolean flags) and ``self.datapoint_values`` (numeric or path values).

        The scan window starts ``sunspot_number_months`` months before
        ``dataset_interval[0]`` so that sunspot lag features are available
        for the very first target timestep.

        If ``precompute_and_reuse_dataset`` is ``True``, the resulting tables
        are written to ``ml_data_dir`` as ``data_availability.csv`` and
        ``data_values.csv``.
        """
        physical_data_cols = ['solar_wind_speed'] + list(self.physical_input_data)
        # solar_cycle_year and solar_wind_speed_persistence are derived at
        # runtime from preloaded series; they don't need file-based scanning
        for derived in ('solar_cycle_year', 'solar_wind_speed_persistence'):
            if derived in physical_data_cols:
                physical_data_cols.remove(derived)

        cols = self.channels + physical_data_cols
        dtypes = [str] * len(self.channels) + [float] * len(physical_data_cols)

        # Extend the scan window back so that sunspot lag features are
        # populated for the very first target timestep
        start = (
            self.dataset_interval[0]
            - relativedelta(months=self.sunspot_number_months)
        )
        start = datetime(start.year, start.month, 1)
        end = self.dataset_interval[1]

        date_index = pd.date_range(start, end, freq='60min')
        self.available_datapoints = pd.DataFrame(
            np.zeros((len(date_index), len(cols))),
            index=date_index,
            columns=cols,
            dtype=bool,
        )
        self.datapoint_values = pd.DataFrame(
            np.zeros((len(date_index), len(cols))),
            index=date_index,
            columns=cols,
        ).astype(dtype=dict(zip(cols, dtypes)))

        # Register solar wind speed availability from the loaded time series
        idx_intersection = self.solar_wind_speed.index.intersection(
            self.available_datapoints.index
        )
        self.available_datapoints.loc[idx_intersection, 'solar_wind_speed'] = True
        self.datapoint_values.loc[idx_intersection, 'solar_wind_speed'] = (
            self.solar_wind_speed.loc[idx_intersection]
        )

        # Guard with hasattr: these attributes only exist when the
        # corresponding feature is included in physical_input_data
        if hasattr(self, 'heliospheric_latitude'):
            idx_intersection = self.heliospheric_latitude.index.intersection(
                self.available_datapoints.index
            )
            self.available_datapoints.loc[
                idx_intersection, 'heliospheric_latitude'
            ] = True
            self.datapoint_values.loc[
                idx_intersection, 'heliospheric_latitude'
            ] = self.heliospheric_latitude.loc[idx_intersection]

        if hasattr(self, 'solar_cycle_year'):
            idx_intersection = self.solar_cycle_year.index.intersection(
                self.available_datapoints.index
            )
            self.available_datapoints.loc[
                idx_intersection, 'solar_cycle_year'
            ] = True
            self.datapoint_values.loc[
                idx_intersection, 'solar_cycle_year'
            ] = self.solar_cycle_year.loc[idx_intersection]

        current_month = start
        printed_years: set[str] = set()
        while current_month <= end:
            year_str = current_month.strftime('%Y')
            month_str = current_month.strftime('%m')
            if (
                self.logger is not None
                and self.rank == 0
                and year_str not in printed_years
            ):
                self.logger.info(f'Checking files for {year_str}.')
                printed_years.add(year_str)
            for channel in self.channels:
                self._check_images(
                    self.solar_image_dir / channel / year_str / month_str, channel
                )
            self._check_sunspot_number(current_month)
            current_month += relativedelta(months=1)

        if self.precompute_and_reuse_dataset:
            self.available_datapoints.to_csv(
                self.ml_data_dir / 'data_availability.csv', index=True
            )
            self.datapoint_values.to_csv(
                self.ml_data_dir / 'data_values.csv', index=True
            )

    # ------------------------------------------------------------------
    # Feature assembly
    # ------------------------------------------------------------------

    def _assign_features(
        self,
        persistence_td: pd.TimedeltaIndex,
        ssn_monthly_idx: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute availability flags and values for all physical and image features.

        For each requested feature, looks up the corresponding entry in
        ``self.available_datapoints`` and ``self.datapoint_values`` at the
        appropriate lag relative to each target date, and returns the results
        as parallel DataFrames.

        Args:
            persistence_td: Time deltas for each solar wind speed
                persistence lag.
            ssn_monthly_idx: Month offsets (negative integers) for
                sunspot number lags.

        Returns:
            Tuple of four DataFrames, all indexed by ``self.target_dates``:

            - ``avail_image_features``: Boolean availability flags for images.
            - ``avail_physical_features``: Boolean availability flags for
              physical features.
            - ``data_image_features``: File paths (strings) for image channels.
            - ``data_physical_features``: Numeric values for physical features.
        """
        avail_physical: list[np.ndarray] = []
        data_physical: list[np.ndarray] = []
        physical_names: list[str] = []

        # Solar wind speed persistence lags
        for td in persistence_td:
            shifted_idx = self.target_dates + td
            avail_physical.append(
                self.available_datapoints.loc[shifted_idx, 'solar_wind_speed'].values
            )
            data_physical.append(
                self.datapoint_values.loc[shifted_idx, 'solar_wind_speed'].values
            )
            lag_hours = -(td.days * 24 + td.seconds // 3600)
            physical_names.append(f"solar_wind_speed_persistence_{lag_hours}")

        # Monthly sunspot number lags
        for idx in ssn_monthly_idx:
            shifted_idx = self.target_dates + pd.DateOffset(months=idx)
            avail_physical.append(
                self.available_datapoints.loc[shifted_idx, 'sunspot_number'].values
            )
            data_physical.append(
                self.datapoint_values.loc[shifted_idx, 'sunspot_number'].values
            )
            physical_names.append(f"sunspot_number_{-idx}")

        # Heliospheric latitude (4-day lag matches the solar wind travel time)
        shifted_idx = self.target_dates - timedelta(days=4)
        avail_physical.append(
            self.available_datapoints.loc[shifted_idx, 'heliospheric_latitude'].values
        )
        data_physical.append(
            self.datapoint_values.loc[shifted_idx, 'heliospheric_latitude'].values
        )
        physical_names.append('heliospheric_latitude')

        # Solar cycle year (no lag needed)
        avail_physical.append(
            self.available_datapoints.loc[self.target_dates, 'solar_cycle_year'].values
        )
        data_physical.append(
            self.datapoint_values.loc[self.target_dates, 'solar_cycle_year'].values
        )
        physical_names.append('solar_cycle_year')

        avail_physical_features = pd.DataFrame(
            np.stack(avail_physical, axis=1),
            index=self.target_dates,
            columns=physical_names,
        )
        data_physical_features = pd.DataFrame(
            np.stack(data_physical, axis=1),
            index=self.target_dates,
            columns=physical_names,
        )

        # Image features (4-day lag for solar wind travel time)
        avail_images: list[np.ndarray] = []
        data_images: list[np.ndarray] = []
        shifted_idx = self.target_dates - timedelta(days=4)
        for key in self.channels:
            avail_images.append(
                self.available_datapoints.loc[shifted_idx, key].values
            )
            data_images.append(
                self.datapoint_values.loc[shifted_idx, key].values
            )

        avail_image_features = pd.DataFrame(
            np.stack(avail_images, axis=1),
            index=self.target_dates,
            columns=self.channels,
        )
        data_image_features = pd.DataFrame(
            np.stack(data_images, axis=1),
            index=self.target_dates,
            columns=self.channels,
        )

        return (
            avail_image_features,
            avail_physical_features,
            data_image_features,
            data_physical_features,
        )

    # ------------------------------------------------------------------
    # Loss function weighting
    # ------------------------------------------------------------------

    def set_loss_function_weights(
        self, weights_alpha: float = 0.0, weights_gamma: float = 0.0
    ) -> pd.Series:
        """Compute per-sample loss weights that up-weight high-speed stream events.

        Within each HSS interval from ``hss_list.csv``, weights are scaled by
        the min–max normalised solar wind speed:
        ``weight = 1 + alpha * (speed_norm ** gamma)``. Samples outside HSS
        intervals retain a weight of 1.0.

        Calling with ``weights_alpha = 0.0`` (the default) resets all weights
        to 1.0 (equal weighting). The result is stored in ``self.weights``
        and also returned.

        Args:
            weights_alpha: Maximum weight increment above the baseline of 1.
                Set to 0.0 to disable weighting (all samples equal weight).
            weights_gamma: Exponent controlling the sharpness of the weight
                profile within each HSS interval.

        Returns:
            Per-sample weights aligned with ``self.target``.
        """
        self.weights = pd.Series(
            np.ones(self.target.shape[0], dtype=float), index=self.target.index
        )
        if weights_alpha == 0.0:
            return self.weights

        hss_list = pd.read_csv(
            self.solar_event_dir / 'hss_list.csv',
            index_col=0,
            parse_dates=[1, 3, 4, 5],
        )
        for i in range(hss_list.shape[0]):
            hss_start = hss_list.loc[i, 'start']
            hss_end = hss_list.loc[i, 'end']
            hss_values = self.target.loc[hss_start:hss_end]
            hss_min, hss_max = hss_values.min(), hss_values.max()
            scaled = (hss_values - hss_min) / (hss_max - hss_min + 1e-6)
            hss_weights = 1 + weights_alpha * (scaled ** weights_gamma)
            self.weights.loc[hss_start:hss_end] = hss_weights

        return self.weights

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_feature_config(
        self,
    ) -> tuple[pd.TimedeltaIndex, np.ndarray, list[str]]:
        """Derive feature layout from ``self.physical_input_data``.

        Translates the high-level feature names in ``physical_input_data``
        into the concrete time deltas, month offsets, and column names needed
        by ``_assign_features``. Also sets ``self.num_physical_features`` as
        a side effect.

        Returns:
            Tuple of three items:

            - ``persistence_td`` (pd.TimedeltaIndex): Time deltas for
              solar wind speed persistence lags.
            - ``ssn_monthly_idx`` (np.ndarray): Month offsets (negative
              integers) for sunspot number lags.
            - ``data_col_names`` (list[str]): Ordered list of all feature
              column names.
        """
        data_col_names: list[str] = ['solar_wind_speed'] + self.channels
        self.num_physical_features = 0

        # Default to empty; populated only when the feature is requested
        persistence_td: pd.TimedeltaIndex = pd.TimedeltaIndex([])
        ssn_monthly_idx: np.ndarray = np.array([], dtype=int)

        if 'solar_wind_speed_persistence' in self.physical_input_data:
            days_start, days_end = self.solar_wind_speed_persistence_interval
            persistence_td = pd.timedelta_range(
                timedelta(days=-days_start),
                timedelta(days=-days_end),
                freq=timedelta(hours=self.solar_wind_speed_persistence_cadence),
            )
            self.num_physical_features += len(persistence_td)
            col_hour_offsets = np.arange(
                -days_start * 24,
                -days_end * 24 + 1,
                self.solar_wind_speed_persistence_cadence,
            )
            data_col_names.extend(
                [f"solar_wind_speed_persistence_{-i}" for i in col_hour_offsets]
            )

        if 'sunspot_number' in self.physical_input_data:
            ssn_monthly_idx = np.arange(-self.sunspot_number_months, 0)
            self.num_physical_features += len(ssn_monthly_idx)
            data_col_names.extend(
                [f"sunspot_number_{-i}" for i in ssn_monthly_idx]
            )

        if 'heliospheric_latitude' in self.physical_input_data:
            self.num_physical_features += 1
            data_col_names.append('heliospheric_latitude')

        if 'solar_cycle_year' in self.physical_input_data:
            self.num_physical_features += 1
            data_col_names.append('solar_cycle_year')

        return persistence_td, ssn_monthly_idx, data_col_names

    def _load_or_compute_features(
        self,
        data_col_names: list[str],
        persistence_td: pd.TimedeltaIndex,
        ssn_monthly_idx: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load cached feature arrays, or compute them from raw data if unavailable.

        Attempts to read ``feature_availability.csv`` and
        ``feature_values.csv`` from ``ml_data_dir``. If the cache is absent,
        unreadable, or does not match the current feature configuration, falls
        back to calling ``_check_available_data`` and ``_assign_features`` to
        rebuild the tables from scratch. The recomputed tables are written
        back to the cache if ``precompute_and_reuse_dataset`` is ``True``.

        Args:
            data_col_names: Expected column names, used to validate that
                cached files match the current feature configuration.
            persistence_td: Time deltas for solar wind speed persistence lags.
            ssn_monthly_idx: Month offsets for sunspot number lags.

        Returns:
            Tuple of two DataFrames:

            - ``avail_features``: Boolean availability flags for each feature
              and timestep.
            - ``vals_features``: Feature values corresponding to
              ``avail_features``.
        """
        avail_features = None
        vals_features = None

        if self.precompute_and_reuse_dataset:
            try:
                avail_features = pd.read_csv(
                    self.ml_data_dir / 'feature_availability.csv',
                    index_col=0,
                    parse_dates=[0],
                )
                vals_features = pd.read_csv(
                    self.ml_data_dir / 'feature_values.csv',
                    index_col=0,
                    parse_dates=[0],
                )
                expected_cols = pd.Index(data_col_names)
                if (
                    not avail_features.columns.equals(expected_cols)
                    or not vals_features.columns.equals(expected_cols)
                ):
                    if self.logger is not None:
                        self.logger.warning(
                            'Saved features do not coincide with current feature '
                            'configuration. Features are recomputed.'
                        )
                    avail_features = None
                    vals_features = None
            except (OSError, pd.errors.ParserError, ValueError):
                avail_features = None
                vals_features = None

        if avail_features is None or vals_features is None:
            self._check_available_data()

            self.target_dates = pd.date_range(
                self.dataset_interval[0], self.dataset_interval[1], freq='60min'
            )
            target_availability = self.available_datapoints.loc[
                self.target_dates, 'solar_wind_speed'
            ]
            target_values = self.datapoint_values.loc[
                self.target_dates, 'solar_wind_speed'
            ]

            (
                avail_image_features,
                avail_physical_features,
                vals_image_features,
                vals_physical_features,
            ) = self._assign_features(persistence_td, ssn_monthly_idx)

            avail_features = pd.concat(
                [target_availability, avail_image_features, avail_physical_features],
                axis=1,
            )
            vals_features = pd.concat(
                [target_values, vals_image_features, vals_physical_features], axis=1
            )

            if self.precompute_and_reuse_dataset:
                avail_features.to_csv(
                    self.ml_data_dir / 'feature_availability.csv', index=True
                )
                vals_features.to_csv(
                    self.ml_data_dir / 'feature_values.csv', index=True
                )

        return avail_features, vals_features

    def _create_dataset(self) -> None:
        """Assemble the final dataset by combining features and filtering to complete samples.

        Orchestrates the full dataset construction pipeline:

        1. Derives feature layout via ``_build_feature_config``.
        2. Loads or computes feature availability and value tables via
           ``_load_or_compute_features``.
        3. Retains only timesteps where every feature is available.
        4. Populates all public dataset attributes (``data``, ``target``,
           ``target_dates``, ``physical_features``, ``missing_dates``).
        """
        self.num_images = len(self.channels)
        persistence_td, ssn_monthly_idx, data_col_names = self._build_feature_config()
        avail_features, vals_features = self._load_or_compute_features(
            data_col_names, persistence_td, ssn_monthly_idx
        )

        # Keep only timesteps where every feature is available
        all_available = np.all(avail_features.values, axis=1)
        data = vals_features.loc[all_available, :]

        self.target = data.iloc[:, 0]
        self.physical_features = data.iloc[:, -self.num_physical_features:]
        self.target_dates = self.target.index
        self.target_values = self.target.values
        self.data = data
        self.features = data.iloc[:, 1:]
        self.image_features = data.iloc[:, 1:1 + self.num_images]

        self.missing_dates = pd.date_range(
            self.dataset_interval[0], self.dataset_interval[1], freq='h'
        ).difference(self.target_dates)

        if self.logger is not None and self.rank == 0:
            self.logger.info(
                f"Found {self.data.shape[0]} valid samples in the dataset. "
                f"{len(self.missing_dates)} samples missing."
            )


class TransformedDataset(Dataset):
    """Wrapper dataset that applies optional scalers to physical inputs and targets.

    Wraps a ``SolarDataset`` (or a ``Subset`` thereof) and applies
    scikit-learn-compatible scalers to the physical feature vector and/or
    the target value at sample retrieval time. Image tensors and all other
    fields are passed through unchanged.

    Args:
        base_dataset: The underlying dataset to wrap.
        input_scaler: Fitted scaler for physical feature vectors. Must
            implement ``transform``. If ``None``, features are returned
            unchanged.
        output_scaler: Fitted scaler for target values. Must implement
            ``transform``. If ``None``, targets are returned unchanged.

    Attributes:
        base_dataset (SolarDataset | Subset): The wrapped dataset.
        input_scaler (StandardScaler | None): Physical feature scaler.
        output_scaler (DistributionTransformer | None): Target scaler.
    """

    def __init__(
        self,
        base_dataset: SolarDataset | Subset,
        input_scaler: StandardScaler | None = None,
        output_scaler: DistributionTransformer | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

    def __len__(self) -> int:
        """Return the number of samples in the wrapped dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single scaled sample by integer index.

        Retrieves the sample from the underlying dataset, then applies
        any configured scalers to the physical features and/or target.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Sample dictionary with the same keys as
            ``SolarDataset.__getitem__``, with ``'physical_features'``
            and/or ``'target'`` scaled if the corresponding scaler is set.
        """
        item = self.base_dataset[idx]

        if self.input_scaler is not None:
            item['physical_features'] = torch.tensor(
                self.input_scaler.transform(
                    item['physical_features'].numpy().reshape(1, -1)
                ).reshape(-1),
                dtype=torch.float32,
            )
        if self.output_scaler is not None:
            item['target'] = torch.tensor(
                self.output_scaler.transform(
                    item['target'].numpy().reshape(1, -1)
                ).reshape(-1),
                dtype=torch.float32,
            )
        return item
