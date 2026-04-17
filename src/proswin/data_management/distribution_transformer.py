import numpy as np
import pandas as pd
import torch
from scipy.stats import lognorm
from sklearn.preprocessing import StandardScaler


class DistributionTransformer:
    """Transform solar wind speed targets to and from a standardised log-space.

    The model predicts a mean ``mu`` and standard deviation ``sigma`` in a
    standardised log-space.  This class encapsulates the two-step forward
    transform applied to training targets:

    1. **Log-shift**: subtract the ``loc`` parameter of a fitted lognormal
       distribution, then take the natural logarithm.
    2. **Standardisation**: apply a ``StandardScaler`` (zero mean, unit
       variance) fitted on the log-shifted values.

    The inverse transforms convert network outputs back to the original solar
    wind speed domain: ``inverse_transform_target`` reverses the pipeline for
    scalar target values, while ``inverse_transform_prediction`` maps predicted
    ``(mu, sigma)`` pairs to the ``(shape, scale, loc)`` parametrisation
    expected by ``scipy.stats.lognorm``.

    Attributes:
        loc_shift: Location shift subtracted before the log transform; set by
            ``fit`` to the ``loc`` parameter of a fitted lognormal distribution.
        target_scaler: ``StandardScaler`` fitted on log-shifted target values.
    """

    def __init__(self) -> None:
        self.loc_shift: float = 0.0
        self.target_scaler: StandardScaler = StandardScaler()

    def fit(self, y: np.ndarray) -> None:
        """Fit the transformer to a training target array.

        Estimates the lognormal location shift from ``y`` via MLE, then fits
        a ``StandardScaler`` on the log-shifted values.

        Args:
            y: 1-D array of positive training target values (solar wind
                speeds in km/s).
        """
        log_fit_params = lognorm.fit(y)
        self.loc_shift = log_fit_params[1]

        y_shifted = y - self.loc_shift
        y_log_transformed = np.log(y_shifted)
        self.target_scaler.fit(np.expand_dims(y_log_transformed, axis=1))

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply the log-shift and standardisation to target values.

        Args:
            y: Array of positive target values (scalar, 1-D, or 2-D).

        Returns:
            Standardised log-shifted values with the same leading shape as
            ``y``, squeezed to remove the trailing singleton dimension added
            for sklearn compatibility.
        """
        y_shifted = y - self.loc_shift
        y_log_transformed = np.log(y_shifted)
        return self.target_scaler.transform(
            self._to_sklearn_input(y_log_transformed)
        ).squeeze()

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit the transformer and return the transformed values.

        Equivalent to calling ``fit(y)`` followed by ``transform(y)``.

        Args:
            y: 1-D array of positive training target values (solar wind
                speeds in km/s).

        Returns:
            Standardised log-shifted values; see ``transform``.
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform_target(
        self, target: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """Reverse the standardisation and log-shift for scalar target values.

        Accepts PyTorch tensors (with or without gradients) as well as NumPy
        arrays; always returns a NumPy array.

        Args:
            target: Predicted or true target values in standardised log-space.
                May be a PyTorch tensor or a NumPy array of any shape.

        Returns:
            Target values in the original solar wind speed domain (km/s),
            always at least 1-D.
        """
        if torch.is_tensor(target):
            # detach() is a no-op for tensors that do not require gradients
            target = target.detach().cpu().numpy()

        target_rescaled = self.target_scaler.inverse_transform(
            self._to_sklearn_input(target)
        ).squeeze()

        # Ensure at least 1-D so callers can always index the result
        if np.ndim(target_rescaled) == 0:
            target_rescaled = np.expand_dims(target_rescaled, axis=0)

        target_inverse_log_trans = np.exp(target_rescaled)
        target_reshifted = target_inverse_log_trans + self.loc_shift
        return target_reshifted

    def inverse_transform_prediction(self, pred: pd.DataFrame) -> pd.DataFrame:
        """Convert normalised (mu, sigma) predictions to lognormal parameters.

        The model outputs ``mu`` and ``sigma`` in standardised log-space.
        This method maps them to the ``(shape, scale, loc)`` parametrisation
        used by ``scipy.stats.lognorm`` and ``scoringrules.crps_lognormal``:

        - ``shape`` (s): standard deviation in log-space —
          ``sigma * scaler.scale_``.
        - ``scale``: ``exp(mu_log)`` where ``mu_log`` is the unstandardised
          log-space mean — ``exp(mu * scaler.scale_ + scaler.mean_)``.
        - ``loc``: the additive location shift fitted during ``fit``
          (constant across all predictions).

        Args:
            pred: DataFrame with columns ``'mu'`` and ``'sigma'`` containing
                the model's normalised predictions.

        Returns:
            DataFrame with columns ``'shape'``, ``'scale'``, ``'loc'`` and
            the same index as ``pred``, ready for use with
            ``scipy.stats.lognorm`` or ``scoringrules.crps_lognormal``.
        """
        # Invert the StandardScaler: y_scaled = (y_log - mean_) / scale_,
        # so y_log = y_scaled * scale_ + mean_; for sigma (a std), only
        # the scale factor applies — no mean shift.
        mu_underlying_norm = (
            pred.loc[:, "mu"] * self.target_scaler.scale_
            + self.target_scaler.mean_
        )
        sigma_underlying_norm = pred.loc[:, "sigma"] * self.target_scaler.scale_

        shape = sigma_underlying_norm.values       # s in scipy.stats.lognorm
        scale = np.exp(mu_underlying_norm.values)  # exp(mu) in log-space
        loc = np.ones(len(shape)) * self.loc_shift # additive location shift

        return pd.DataFrame(
            np.stack([shape, scale, loc], axis=1),
            index=pred.index,
            columns=["shape", "scale", "loc"],
        )

    # --- Private helpers ---

    @staticmethod
    def _to_sklearn_input(arr: np.ndarray) -> np.ndarray:
        """Reshape an array to a 2-D column vector for sklearn compatibility.

        ``StandardScaler.transform`` and ``.inverse_transform`` require a 2-D
        input of shape ``(n_samples, n_features)``.  This helper squeezes any
        length-1 dimensions and then promotes the result to 2-D.

        Args:
            arr: Array of any shape (scalar, 1-D, or already 2-D).

        Returns:
            2-D array of shape ``(n, 1)`` for 1-D input, ``(1, 1)`` for a
            scalar, or the original array unchanged if it is already 2-D.
        """
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            return np.expand_dims(arr, axis=1)
        if arr.ndim == 0:
            return np.expand_dims(arr, axis=[0, 1])
        return arr  # already 2-D
