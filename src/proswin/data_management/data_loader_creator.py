import os

import pandas as pd
import torch
import torch.distributed
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from proswin.data_management.distribution_transformer import DistributionTransformer
from proswin.data_management.solar_dataset import SolarDataset, TransformedDataset


class DataLoaderCreator:
    """Create fitted, scaled DataLoaders for training, validation, and testing.

    Scalers are fitted exclusively on training data to prevent data leakage,
    then applied consistently across all splits via ``TransformedDataset``
    wrappers.

    When ``use_multiple_gpus=True``, rank and world size are read automatically
    from the active ``torch.distributed`` process group, so they do not need to
    be passed manually.

    Args:
        dataset: The full dataset containing all dates.
        train_dates: Dates to include in the training split.
        validation_dates: Dates to include in the validation split.
        test_dates: Dates to include in the test split.
        use_multiple_gpus: If True, attach a ``DistributedSampler`` to the
            training loader for multi-GPU (DDP) training. Default is False.

    Attributes:
        input_scaler: ``StandardScaler`` fitted on training physical features,
            or ``None`` in image-only mode.
        output_scaler: ``DistributionTransformer`` fitted on training targets.
    """

    def __init__(
        self,
        dataset: SolarDataset,
        train_dates: pd.DatetimeIndex,
        validation_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        use_multiple_gpus: bool = False,
    ) -> None:
        self._dataset = dataset
        self._train_dates = train_dates
        self._validation_dates = validation_dates
        self._test_dates = test_dates
        self._use_multiple_gpus = use_multiple_gpus

        # Fit scalers on training data only to prevent data leakage
        self.input_scaler, self.output_scaler = _fit_scalers(dataset, train_dates)

    def get_loaders(
        self,
        batch_size: int = 32,
        num_workers: int | None = None,
        drop_last: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Build and return DataLoaders for training, validation, and testing.

        The training loader is shuffled; validation and test loaders preserve
        insertion order.

        Args:
            batch_size: Number of samples per batch. Default is 32.
            num_workers: Subprocesses for data loading. Defaults to half the
                number of available CPU cores.
            drop_last: If True, drop the final incomplete batch from the
                training loader only. Default is True.

        Returns:
            A 3-tuple ``(train_loader, val_loader, test_loader)``.
        """
        num_workers = _resolve_num_workers(num_workers)

        # Map each target timestamp to its integer position in the dataset
        date_to_index = pd.Series(
            range(len(self._dataset)),
            index=self._dataset.target_dates,
        )

        splits = {
            "train": self._train_dates,
            "val": self._validation_dates,
            "test": self._test_dates,
        }

        loaders = {}
        for split_name, dates in splits.items():
            indices = list(date_to_index.loc[dates])
            subset = Subset(self._dataset, indices)
            transformed = TransformedDataset(
                subset, self.input_scaler, self.output_scaler
            )
            shuffle = split_name == "train"  # Only shuffle training data
            # Drop the incomplete final batch for training; keep all for eval
            drop_last_this_split = drop_last and (split_name == "train")
            loaders[split_name] = self._build_loader(
                transformed,
                batch_size,
                num_workers,
                shuffle=shuffle,
                drop_last=drop_last_this_split,
            )

        return loaders["train"], loaders["val"], loaders["test"]

    def _build_loader(
        self,
        dataset: TransformedDataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        drop_last: bool = False,
    ) -> DataLoader:
        """Construct a single DataLoader, with an optional DDP sampler.

        Args:
            dataset: The (sub)dataset to load from.
            batch_size: Number of samples per batch.
            num_workers: Number of data-loading subprocesses.
            shuffle: Whether to shuffle data. Ignored when using DDP, as the
                ``DistributedSampler`` controls the sample order.
            drop_last: Whether to drop the last incomplete batch. Default is
                False.

        Returns:
            Configured ``DataLoader`` instance.
        """
        # Workers that stay alive across epochs avoid the per-epoch spawn cost
        persistent_workers = num_workers > 0

        if self._use_multiple_gpus:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,  # per-GPU batch size; caller's responsibility
                sampler=sampler,        # sampler and shuffle are mutually exclusive
                num_workers=num_workers,
                pin_memory=True,        # speeds up CPU → GPU transfers
                persistent_workers=persistent_workers,
                drop_last=drop_last,
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
        )


def get_prediction_loader(
    dataset: SolarDataset,
    input_scaler: StandardScaler,
    output_scaler: DistributionTransformer,
    dates: pd.DatetimeIndex | None = None,
    batch_size: int = 32,
    num_workers: int | None = None,
) -> DataLoader:
    """Build a single DataLoader for inference using pre-fitted scalers.

    No scaler fitting occurs here; scalers must be obtained from a previously
    fitted ``DataLoaderCreator`` instance to avoid data leakage.

    Args:
        dataset: The dataset to load from.
        input_scaler: Already-fitted scaler for physical input features.
        output_scaler: Already-fitted scaler for target values.
        dates: Subset of dates to load. If None, the entire dataset is used.
        batch_size: Number of samples per batch. Default is 32.
        num_workers: Subprocesses for data loading. Defaults to half the
            number of available CPU cores.

    Returns:
        Non-shuffled ``DataLoader`` ready for prediction; sample order is
        preserved so predictions can be matched back to their timestamps.

    Raises:
        KeyError: If any date in ``dates`` is not found in the dataset.
    """
    num_workers = _resolve_num_workers(num_workers)

    if dates is not None:
        date_to_index = pd.Series(
            range(len(dataset)), index=dataset.target_dates
        )
        indices = list(date_to_index.loc[dates])
        source = Subset(dataset, indices)
    else:
        source = dataset  # Use entire dataset when no date filter is given

    transformed = TransformedDataset(source, input_scaler, output_scaler)
    persistent_workers = num_workers > 0

    return DataLoader(
        transformed,
        batch_size=batch_size,
        shuffle=False,          # Never shuffle for prediction; order must be preserved
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )


def _fit_scalers(
    dataset: SolarDataset,
    train_dates: pd.DatetimeIndex,
) -> tuple[StandardScaler | None, DistributionTransformer]:
    """Fit input and output scalers on training data only.

    Args:
        dataset: Full dataset containing physical features and targets.
        train_dates: Dates corresponding to the training split.

    Returns:
        A 2-tuple ``(input_scaler, output_scaler)``:

        - ``input_scaler``: ``StandardScaler`` fitted on the physical feature
          columns of the training split.
        - ``output_scaler``: ``DistributionTransformer`` fitted on the target
          values of the training split.
    """
    y_train = dataset.target.loc[train_dates].values

    # When no physical features are configured (images-only mode) there is
    # nothing to scale; leave input_scaler as None so TransformedDataset
    # passes the empty feature tensor through unchanged.
    if dataset.physical_features.shape[1] > 0:
        X_train = dataset.physical_features.loc[train_dates].values
        input_scaler: StandardScaler | None = StandardScaler()
        input_scaler.fit(X_train)
    else:
        input_scaler = None

    output_scaler = DistributionTransformer()
    output_scaler.fit(y_train)

    return input_scaler, output_scaler


def _resolve_num_workers(num_workers: int | None) -> int:
    """Return the worker count for data loading, computing a default if needed.

    The default is half the number of available CPU cores, with a minimum of 1.

    Args:
        num_workers: Explicit worker count, or None to use the default.

    Returns:
        Resolved worker count as a positive integer.
    """
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count // 2)
    return num_workers
