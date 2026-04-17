import logging
import os
import random
from datetime import datetime, timedelta
from logging import LoggerAdapter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.distributed import destroy_process_group, init_process_group


class ConfigLoader:
    """Loads, validates, and exposes the PROSWIN experiment configuration.

    Reads ``configs/config.yaml`` relative to the project root, resolves
    all file-system paths to absolute ``Path`` objects, optionally
    initialises PyTorch Distributed Data Parallel (DDP), sets up a shared
    logger, and seeds all random-number generators for reproducibility.

    Attributes:
        project_root (Path): Absolute path to the project root directory.
        content (dict): Raw parsed YAML content.
        dataset_config (dict): Dataset section of the configuration.
        model_config (dict): Model architecture section.
        train_config (dict): Training hyperparameters section.
        paths (dict[str, Path]): Resolved absolute paths for all
            directories (inputs and outputs).
        rank (int): Global DDP rank of this process (0 for single-GPU runs).
        local_rank (int): Local DDP rank within the node (0 for single-GPU).
        world_size (int): Total number of DDP processes (1 for single-GPU).
        logger (LoggerAdapter): Shared logger that injects the DDP rank
            into every log message.
    """

    def __init__(
        self,
        project_root: Path | str,
        enable_logging: bool = True,
        initialize_ddp: bool = True,
    ) -> None:
        """Initialise the configuration loader.

        Reads ``configs/config.yaml``, resolves paths, optionally sets up
        DDP and logging, and seeds all RNGs.

        Args:
            project_root: Path to the project root directory. Both
                ``pathlib.Path`` and plain strings are accepted.
            enable_logging: If ``True``, configure a file and stream logger
                under ``<project_root>/logs/``. Set to ``False`` when the
                logger is not needed (e.g., evaluation-only scripts).
            initialize_ddp: If ``True`` *and* ``training.use_multiple_gpus``
                is set in the config, initialise the PyTorch DDP process
                group. Set to ``False`` for scripts that do not train.

        Raises:
            FileNotFoundError: If ``configs/config.yaml`` does not exist
                under ``project_root``.
            KeyError: If a required top-level YAML section is missing.
            ValueError: If ``dataset_interval`` cannot be parsed as ISO 8601.
            RuntimeError: If DDP is requested but the required environment
                variables (RANK, WORLD_SIZE, LOCAL_RANK) are not set.
        """
        # Normalise to Path immediately so all downstream path operations
        # work correctly regardless of whether the caller passes a str or Path
        self.project_root = Path(project_root)
        config_path = self.project_root / "configs" / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        self.content = self._read_yaml(config_path)

        self.dataset_config = self.content["dataset"]
        self.model_config = self.content["model"]
        self.train_config = self.content["training"]

        self._process_dataset_interval()

        self.paths = self.content["paths"]
        self._setup_paths()

        if initialize_ddp and self.train_config["use_multiple_gpus"]:
            self._setup_ddp()
        else:
            # Single-GPU or CPU run: set DDP attributes to their default values
            self.rank: int = 0
            self.local_rank: int = 0
            self.world_size: int = 1

        if enable_logging:
            self._setup_logger()

        seed = self.content.get("experiment", {}).get("seed", 42)
        self._set_seed(seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_yaml(config_path: Path) -> dict[str, Any]:
        """Parse a YAML file and return its content as a dictionary.

        Args:
            config_path: Absolute path to the YAML configuration file.

        Returns:
            Parsed YAML content as a nested dictionary.

        Raises:
            ValueError: If the YAML file is empty.
        """
        with open(config_path, "r", encoding="utf-8") as file:
            content = yaml.safe_load(file)

        if content is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

        return content

    def _process_dataset_interval(self) -> None:
        """Parse ISO 8601 date strings in ``dataset_interval`` to datetimes.

        Converts the two-element list of ISO 8601 strings stored under
        ``dataset.dataset_interval`` in the YAML into Python ``datetime``
        objects and writes them back to ``self.dataset_config`` in-place.

        Raises:
            KeyError: If ``dataset_interval`` is absent from the dataset
                config section.
            ValueError: If ``dataset_interval`` is not a two-element list,
                or if either string cannot be parsed as ISO 8601.
        """
        if "dataset_interval" not in self.dataset_config:
            raise KeyError(
                "Configuration must contain 'dataset_interval'"
            )

        interval = self.dataset_config["dataset_interval"]

        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(
                "dataset_interval must be a list of two ISO 8601 date strings."
            )

        try:
            self.dataset_config["dataset_interval"] = [
                datetime.fromisoformat(interval[0]),
                datetime.fromisoformat(interval[1]),
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid date format in dataset_interval. Expected ISO 8601 "
                f"format (e.g., '2010-06-01T00:00:00'). Error: {exc}"
            ) from exc

    def _setup_paths(self) -> None:
        """Resolve all config paths to absolute paths and create output directories.

        Converts every entry in ``self.paths`` from a relative string to an
        absolute ``Path`` by prepending the resolved project root. Then
        creates the main ``results/`` directory and its standard
        subdirectories (``epoch_losses``, ``predictions``, ``evaluation``,
        ``figures``, ``models``), registering each subdirectory path in
        ``self.paths`` for convenient downstream access.
        """
        # Resolve to an absolute path so all downstream code is location-independent
        base_path = self.project_root.resolve()
        for key in self.paths:
            self.paths[key] = base_path / self.paths[key]

        # Create the top-level results directory
        self.paths["results"].mkdir(parents=True, exist_ok=True)

        # Create standard output subdirectories and register them in self.paths
        for subdir in ["epoch_losses", "predictions", "evaluation", "figures", "models"]:
            subdir_path = self.paths["results"] / subdir
            self.paths[subdir] = subdir_path
            subdir_path.mkdir(exist_ok=True)

    def _setup_logger(self) -> None:
        """Configure a file and stream logger shared across all DDP ranks.

        Creates a timestamped log file under ``<project_root>/logs/`` and
        attaches both a ``FileHandler`` and a ``StreamHandler`` to a named
        logger (``"solar_logger"``). The logger is wrapped in a
        ``LoggerAdapter`` that injects the DDP rank into every message, so
        log lines from different processes can be distinguished.

        In multi-GPU runs a ``dist.barrier()`` is called first to
        synchronise all ranks before the log file is created, which avoids
        race conditions on shared filesystems (common on HPC/SLURM).

        After this method returns, ``self.logger`` is ready for use.
        """
        log_path = self.project_root / "logs"
        log_path.mkdir(exist_ok=True)
        log_file_path = log_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Synchronise all ranks before file creation to avoid race conditions
        if self.train_config["use_multiple_gpus"]:
            dist.barrier()

        logger = logging.getLogger("solar_logger")
        logger.setLevel(logging.INFO)

        # Guard against duplicate handlers if ConfigLoader is instantiated
        # more than once in the same process (e.g., in unit tests)
        if not logger.handlers:
            formatter = logging.Formatter(
                fmt="[%(asctime)s][Rank %(rank)d] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            file_handler = logging.FileHandler(log_file_path, "a")
            file_handler.setFormatter(formatter)

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

        # Wrap in LoggerAdapter so %(rank)d is always available in the format string
        self.logger = LoggerAdapter(logger, extra={"rank": self.rank})

        self.logger.info(
            f"Logger initialised — rank {self.rank}, local rank {self.local_rank}"
        )

    def _setup_ddp(self) -> None:
        """Initialise the PyTorch Distributed Data Parallel process group.

        Reads the distributed environment variables set by ``torchrun``
        (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``), assigns the appropriate
        CUDA device to this process, and initialises the NCCL process group.

        Raises:
            RuntimeError: If any of ``RANK``, ``WORLD_SIZE``, or
                ``LOCAL_RANK`` are absent from the environment, indicating
                that the script was not launched with ``torchrun``.
        """
        # Validate that torchrun has set the required environment variables
        required_env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
        missing_vars = [var for var in required_env_vars if var not in os.environ]

        if missing_vars:
            raise RuntimeError(
                f"Missing environment variables for DDP: {missing_vars}. "
                f"Ensure the script is launched with torchrun or equivalent."
            )

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        # Bind this process to its designated GPU
        torch.cuda.set_device(self.local_rank)

        # NCCL is the recommended backend for GPU-to-GPU communication
        init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=60),
        )

    def _set_seed(self, seed: int = 42) -> None:
        """Seed all random-number generators for reproducibility.

        Seeds Python's ``random`` module, NumPy, and both PyTorch CPU and
        CUDA RNGs. In multi-GPU runs each rank receives a unique but
        deterministic seed (``seed + rank``) to avoid identical data
        augmentation across processes while keeping runs reproducible.

        Args:
            seed: Base random seed, read from ``experiment.seed`` in
                ``config.yaml`` (default 42). Each DDP rank uses
                ``seed + rank``.
        """
        random.seed(seed + self.rank)
        np.random.seed(seed + self.rank)
        torch.manual_seed(seed + self.rank)
        torch.cuda.manual_seed_all(seed + self.rank)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def terminate_ddp(self) -> None:
        """Destroy the DDP process group when training is complete.

        Must be called after training finishes in multi-GPU runs to cleanly
        shut down inter-process communication. Safe to call in single-GPU
        or CPU runs — it is a no-op when ``use_multiple_gpus`` is false.
        """
        if self.train_config["use_multiple_gpus"]:
            destroy_process_group()
