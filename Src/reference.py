import numpy as np
from pathlib import Path


REFERENCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "Data"
    / "reference_well_driven_reservoir.npz"
)


def load_reference():
    """
    Load the numerical reference solution for the
    well-driven heterogeneous reservoir.

    Returns
    -------
    time : numpy.ndarray
        Simulation times in seconds.
    pressure : numpy.ndarray
        Reference pressure field.
    """

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference solution not found: {REFERENCE_PATH}"
        )

    data = np.load(REFERENCE_PATH)

    return data["time"], data["pressure"]


def get_reference_at_time(time_days):
    """
    Return the numerical reference pressure field
    at the requested simulation time.
    """

    time, pressure = load_reference()

    target_seconds = time_days * 24.0 * 3600.0

    time_index = np.argmin(
        np.abs(time - target_seconds)
    )

    return pressure[time_index]


