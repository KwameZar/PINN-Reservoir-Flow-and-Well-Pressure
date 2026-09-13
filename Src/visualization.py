import numpy as np

from .reference import get_reference_at_time
from .prediction import predict_drawdown


LX = 1000.0
LY = 1000.0
TOTAL_DAYS = 30.0
DX = 20.0
DY = 20.0

NX = int(LX / DX) + 1
NY = int(LY / DY) + 1

WELL_X = 500.0
WELL_Y = 500.0

WELL_I = int(WELL_X / DX)
WELL_J = int(WELL_Y / DY)


def create_evaluation_grid():
    x = np.linspace(0.0, LX, NX)
    y = np.linspace(0.0, LY, NY)

    X, Y = np.meshgrid(x, y, indexing="ij")

    return X, Y


def predict_reservoir_drawdown(time_days):
    """
    Generate PINN and numerical-reference drawdown fields
    for a given simulation time.
    """

    X, Y = create_evaluation_grid()

    x_normalized = X / LX
    y_normalized = Y / LY

    time_array = np.full(
        X.size,
        time_days,
        dtype=np.float32
    )

    drawdown = predict_drawdown(
        x_normalized.ravel(),
        y_normalized.ravel(),
        time_array
    )

    drawdown = drawdown.reshape(X.shape)

    reference_pressure = get_reference_at_time(time_days)

    reference_drawdown = 1.0 - reference_pressure

    return (
        X,
        Y,
        drawdown,
        reference_drawdown,
        WELL_X,
        WELL_Y
    )


def get_well_drawdown_history():
    """
    Generate PINN and numerical-reference drawdown
    at the production well from 0 to 30 days.
    """

    time_days = np.arange(
        0.0,
        TOTAL_DAYS + 1.0,
        1.0
    )

    well_x_normalized = WELL_X / LX
    well_y_normalized = WELL_Y / LY

    x_array = np.full(
        len(time_days),
        well_x_normalized,
        dtype=np.float32
    )

    y_array = np.full(
        len(time_days),
        well_y_normalized,
        dtype=np.float32
    )

    pinn_drawdown = predict_drawdown(
        x_array,
        y_array,
        time_days
    )

    reference_drawdown = []

    for day in time_days:
        reference_pressure = get_reference_at_time(day)

        well_pressure = reference_pressure[
    WELL_I,
    WELL_J
     ]   

        reference_drawdown.append(
            1.0 - well_pressure
        )

    reference_drawdown = np.array(
        reference_drawdown
    )

    return (
        time_days,
        pinn_drawdown,
        reference_drawdown
    )