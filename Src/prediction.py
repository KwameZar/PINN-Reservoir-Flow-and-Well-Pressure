import numpy as np
from .model import model


def predict_drawdown(x, y, time_days):
    """
    Predict normalized reservoir drawdown using the trained PINN.

    Parameters
    ----------
    x : array-like
        Normalized x-coordinate, 0 to 1.
    y : array-like
        Normalized y-coordinate, 0 to 1.
    time_days : array-like
        Time in days, 0 to 30.

    Returns
    -------
    numpy.ndarray
        Predicted normalized drawdown.
    """

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    time_days = np.asarray(time_days, dtype=np.float32)

    time_normalized = time_days / 30.0

    X = np.column_stack([
        x,
        y,
        time_normalized
    ])

    return model(X, training=False).numpy().flatten()

