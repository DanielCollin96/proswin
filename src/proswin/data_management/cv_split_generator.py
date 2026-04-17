import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


class SequentialCVSplitGenerator:
    """Generate sequential cross-validation splits for time-series data.

    Divides the observation window into equally sized chunks and assigns them
    to train, validation, and test folds in a round-robin fashion. A
    configurable gap is excluded around each held-out fold to prevent data
    leakage from the persistence feature (Carrington rotation, ~27 days).

    Args:
        dataset_dates: Timestamps of samples that are present in the dataset.
        gaps_in_dataset: Timestamps that fall within the observation window
            but have no corresponding sample (missing data).

    Attributes:
        existing_dates: DatetimeIndex of all available sample timestamps.
        missing_dates: DatetimeIndex of all gap timestamps.
        dates: Union of existing_dates and missing_dates; covers the full
            observation window without holes.
    """

    def __init__(
        self,
        dataset_dates: pd.DatetimeIndex,
        gaps_in_dataset: pd.DatetimeIndex,
    ) -> None:
        self.existing_dates = pd.DatetimeIndex(dataset_dates)
        self.missing_dates = pd.DatetimeIndex(gaps_in_dataset)
        self.dates = self.existing_dates.union(self.missing_dates)

    def compute_split(
        self,
        num_folds: int = 5,
        n_test_chunks: int = 4,
        gap_days: int = 90,
        validation: bool = True,
        save_split: bool = False,
        ml_data_dir: Path | None = None,
    ) -> tuple[dict[str, list[pd.DatetimeIndex]], dict[str, list[pd.DatetimeIndex]]]:
        """Compute cross-validation splits and optionally save them to disk.

        Splits the time series into chunks, assigns them to folds, excludes
        gap periods around each held-out fold, and records which dates were
        missing in each subset.

        Args:
            num_folds: Number of cross-validation folds.
            n_test_chunks: Number of equally sized chunks assigned to each
                test fold.
            gap_days: Days excluded on each side of a test or validation fold
                to prevent data leakage (should exceed the Carrington rotation
                period of ~27 days; default 90).
            validation: If True, include a separate validation fold in each
                split. If False, validation dates are merged into training.
            save_split: If True, serialise the split to
                ``ml_data_dir/cross_validation_split.pkl``.
            ml_data_dir: Directory in which to save the split pickle. Required
                when ``save_split`` is True; a warning is emitted if absent.

        Returns:
            A 2-tuple ``(cv_split_dates, missing_dates)``:

            - ``cv_split_dates``: dict with keys ``'train'``, ``'test'``, and
              (when ``validation=True``) ``'val'``; each value is a list of
              ``num_folds`` DatetimeIndex objects containing the timestamps
              present in the dataset for that fold.
            - ``missing_dates``: same structure, but each DatetimeIndex
              contains the gap timestamps that fall within the fold's date
              range.
        """
        date_intervals, extended_date_intervals = self._split_in_chunks(
            num_folds=num_folds,
            n_test_chunks=n_test_chunks,
            gap_days=gap_days,
        )
        train_chunk_ids, valid_chunk_ids, test_chunk_ids = (
            self._assign_chunk_ids_to_subsets(
                num_folds=num_folds,
                n_test_chunks=n_test_chunks,
                validation=validation,
            )
        )

        cv_split_dates: dict[str, list[pd.DatetimeIndex]] = {}
        cv_split_dates["train"], cv_split_dates["val"], cv_split_dates["test"] = (
            self._create_folds(
                date_intervals,
                extended_date_intervals,
                train_chunk_ids,
                valid_chunk_ids,
                test_chunk_ids,
                num_folds=num_folds,
                validation=validation,
            )
        )

        # Record which dates within each fold are missing from the dataset
        missing_dates: dict[str, list[pd.DatetimeIndex]] = {
            "train": [], "val": [], "test": []
        }
        for i in range(num_folds):
            for k in cv_split_dates:
                missing_dates[k].append(
                    cv_split_dates[k][i].intersection(self.missing_dates)
                )
                cv_split_dates[k][i] = cv_split_dates[k][i].intersection(
                    self.existing_dates
                )

        if not validation:
            cv_split_dates.pop("val")
            missing_dates.pop("val")

        if save_split:
            if ml_data_dir is None:
                warnings.warn(
                    "No directory specified for saving cross-validation splits. "
                    "Saving is skipped."
                )
            else:
                cv_info = {"split": cv_split_dates, "missing": missing_dates}
                with open(ml_data_dir / "cross_validation_split.pkl", "wb") as f:
                    pickle.dump(cv_info, f)

        return cv_split_dates, missing_dates

    def _split_in_chunks(
        self,
        num_folds: int = 5,
        n_test_chunks: int = 4,
        gap_days: int = 90,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Divide the full date range into equally sized chunks.

        The time series is divided into ``num_folds * n_test_chunks`` chunks
        of equal length. Any remainder samples are appended to the last chunk.
        For each chunk, an extended interval is computed by padding
        ``gap_days`` on each side; this buffer is later excluded from adjacent
        folds to enforce the leakage gap.

        Args:
            num_folds: Number of cross-validation folds.
            n_test_chunks: Number of chunks per test fold.
            gap_days: Days to pad around each chunk boundary for the leakage
                exclusion buffer.

        Returns:
            A 2-tuple ``(date_intervals, extended_date_intervals)``, each a
            NumPy array of shape ``(num_folds * n_test_chunks, 2)`` whose
            columns are the chunk start and end timestamps. The extended
            intervals additionally pad ``gap_days`` on each side.
        """
        chunk_len = len(self.dates) // (num_folds * n_test_chunks)
        chunk_remainder = len(self.dates) % (num_folds * n_test_chunks)

        date_intervals = []
        extended_date_intervals = []
        end_date_chunk = datetime(1900, 1, 1)
        for i in range(num_folds * n_test_chunks):
            idx_start_date = i * chunk_len
            start_date_chunk = self.dates[idx_start_date]

            # Guard against duplicate timestamps (e.g. from CME alert lists)
            while start_date_chunk == end_date_chunk:
                idx_start_date += 1
                start_date_chunk = self.dates[idx_start_date]
            end_date_chunk = self.dates[(i + 1) * chunk_len - 1]

            date_intervals.append([start_date_chunk, end_date_chunk])
            extended_date_intervals.append([
                start_date_chunk - timedelta(days=gap_days),
                end_date_chunk + timedelta(days=gap_days),
            ])

        date_intervals = np.array(date_intervals)
        extended_date_intervals = np.array(extended_date_intervals)

        # Absorb any remainder samples into the last chunk
        if chunk_remainder > 0:
            date_intervals[-1, 1] = self.dates[-1]

        return date_intervals, extended_date_intervals

    @staticmethod
    def _assign_chunk_ids_to_subsets(
        num_folds: int = 5,
        n_test_chunks: int = 4,
        validation: bool = True,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        """Assign chunk indices to train, validation, and test subsets.

        Uses a round-robin scheme: for fold ``i``, chunks whose index modulo
        ``num_folds`` equals ``i`` are assigned to validation and chunks with
        index modulo ``num_folds`` equal to ``(i + 1) % num_folds`` are
        assigned to test. All remaining chunks go to training. When
        ``validation=False``, the validation chunks are merged into training.

        Args:
            num_folds: Number of cross-validation folds.
            n_test_chunks: Number of chunks per test fold.
            validation: If True, keep validation chunks separate. If False,
                merge them into the training set.

        Returns:
            A 3-tuple ``(train_chunk_ids, valid_chunk_ids, test_chunk_ids)``,
            each a list of ``num_folds`` lists of integer chunk indices.
        """
        train_chunk_ids: list[list[int]] = []
        valid_chunk_ids: list[list[int]] = []
        test_chunk_ids: list[list[int]] = []

        for i in range(num_folds):
            id_train: list[int] = []
            id_valid: list[int] = []
            id_test: list[int] = []
            start_idx = i
            for j in range(num_folds * n_test_chunks):
                if j % num_folds == start_idx:
                    id_valid.append(j)
                elif j % num_folds == (start_idx + 1) % num_folds:
                    id_test.append(j)
                else:
                    id_train.append(j)
            train_chunk_ids.append(id_train)
            if validation:
                valid_chunk_ids.append(id_valid)
            else:
                train_chunk_ids[-1].extend(id_valid)
            test_chunk_ids.append(id_test)

        return train_chunk_ids, valid_chunk_ids, test_chunk_ids

    def _create_folds(
        self,
        date_intervals: np.ndarray,
        extended_date_intervals: np.ndarray,
        train_chunk_ids: list[list[int]],
        valid_chunk_ids: list[list[int]],
        test_chunk_ids: list[list[int]],
        num_folds: int = 5,
        validation: bool = True,
    ) -> tuple[list[pd.DatetimeIndex], list[pd.DatetimeIndex], list[pd.DatetimeIndex]]:
        """Assemble per-fold DatetimeIndex objects from chunk assignments.

        For each fold the test (and optionally validation) chunks are
        collected first. Their extended intervals define blocked date ranges
        that are then removed from the training (and validation) dates to
        enforce the leakage gap.

        Args:
            date_intervals: Array of shape ``(n_chunks, 2)`` with start and
                end timestamps for each chunk.
            extended_date_intervals: Array of shape ``(n_chunks, 2)`` with
                gap-padded start and end timestamps for each chunk.
            train_chunk_ids: Per-fold lists of chunk indices for training.
            valid_chunk_ids: Per-fold lists of chunk indices for validation.
            test_chunk_ids: Per-fold lists of chunk indices for testing.
            num_folds: Number of cross-validation folds.
            validation: If True, build a separate validation DatetimeIndex
                and exclude its blocked dates from training.

        Returns:
            A 3-tuple ``(cv_train, cv_valid, cv_test)``, each a list of
            ``num_folds`` DatetimeIndex objects.
        """
        cv_train: list[pd.DatetimeIndex] = []
        cv_test: list[pd.DatetimeIndex] = []
        cv_valid: list[pd.DatetimeIndex] = []

        for i in range(num_folds):
            test_chunks = test_chunk_ids[i]
            test_data = self.dates[
                (self.dates >= date_intervals[test_chunks[0], 0])
                & (self.dates <= date_intervals[test_chunks[0], 1])
            ]
            dates_blocked_by_test_set = pd.date_range(
                extended_date_intervals[test_chunks[0], 0],
                extended_date_intervals[test_chunks[0], 1],
                freq="1h",
            )
            for j in test_chunks[1:]:
                test_data = test_data.union(
                    self.dates[
                        (self.dates >= date_intervals[j, 0])
                        & (self.dates <= date_intervals[j, 1])
                    ]
                )
                dates_blocked_by_test_set = dates_blocked_by_test_set.union(
                    pd.date_range(
                        extended_date_intervals[j, 0],
                        extended_date_intervals[j, 1],
                        freq="1h",
                    )
                )

            if validation:
                val_chunks = valid_chunk_ids[i]
                valid_data = self.dates[
                    (self.dates >= date_intervals[val_chunks[0], 0])
                    & (self.dates <= date_intervals[val_chunks[0], 1])
                ]
                dates_blocked_by_valid_set = pd.date_range(
                    extended_date_intervals[val_chunks[0], 0],
                    extended_date_intervals[val_chunks[0], 1],
                    freq="1h",
                )
                for j in val_chunks[1:]:
                    valid_data = valid_data.union(
                        self.dates[
                            (self.dates >= date_intervals[j, 0])
                            & (self.dates <= date_intervals[j, 1])
                        ]
                    )
                    dates_blocked_by_valid_set = dates_blocked_by_valid_set.union(
                        pd.date_range(
                            extended_date_intervals[j, 0],
                            extended_date_intervals[j, 1],
                            freq="1h",
                        )
                    )
                # Exclude the leakage buffer around the test fold from validation
                valid_data = valid_data.difference(dates_blocked_by_test_set)
            else:
                valid_data = pd.Index([])

            train_chunks = train_chunk_ids[i]
            train_data = self.dates[
                (self.dates >= date_intervals[train_chunks[0], 0])
                & (self.dates <= date_intervals[train_chunks[0], 1])
            ]
            for j in train_chunks[1:]:
                train_data = train_data.union(
                    self.dates[
                        (self.dates >= date_intervals[j, 0])
                        & (self.dates <= date_intervals[j, 1])
                    ]
                )
            train_data = train_data.difference(dates_blocked_by_test_set)
            if validation:
                train_data = train_data.difference(dates_blocked_by_valid_set)

            cv_train.append(train_data.unique())
            cv_test.append(test_data.unique())
            cv_valid.append(valid_data.unique())

        return cv_train, cv_valid, cv_test
