import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "pinn_well_ic10_mass25.keras"
)

REFERENCE_PATH = (
    PROJECT_ROOT
    / "Data"
    / "reference_well_driven_reservoir.npz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "Results"
    / "well_ic10_mass25_final_evaluation.csv"
)


# --------------------------------------------------
# Load model and reference
# --------------------------------------------------

print("Loading final PINN model...")
model = tf.keras.models.load_model(MODEL_PATH)

test_input = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
test_prediction = model(test_input, training=False).numpy()[0, 0]

print("MODEL PATH:", MODEL_PATH)
print("DIRECT MODEL TEST:", test_prediction)

print("Loading numerical reference...")
reference_data = np.load(REFERENCE_PATH)

time = reference_data["time"]
pressure = reference_data["pressure"]


# --------------------------------------------------
# Reservoir grid
# --------------------------------------------------

NX = 51
NY = 51

WELL_I = 25
WELL_J = 25

CELL_AREA_STAR = 0.02 * 0.02

x = np.linspace(0.0, 1.0, NX)
y = np.linspace(0.0, 1.0, NY)

X, Y = np.meshgrid(
    x,
    y,
    indexing="ij"
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

evaluation_days = [1, 5, 10, 15, 20, 25, 30]

results = []


for day in evaluation_days:

    # ------------------------------
    # Reference
    # ------------------------------

    target_seconds = day * 24.0 * 3600.0

    time_index = np.argmin(
        np.abs(time - target_seconds)
    )

    reference_pressure = pressure[time_index]

    reference_drawdown = (
        1.0 - reference_pressure
    )


    # ------------------------------
    # PINN prediction
    # ------------------------------

    X_input = np.column_stack([
        X.ravel(),
        Y.ravel(),
        np.full(
            X.size,
            day / 30.0,
            dtype=np.float32
        )
    ]).astype(np.float32)

    pinn_drawdown = (
        model(X_input, training=False)
        .numpy()
        .reshape(X.shape)
    )


    # ------------------------------
    # Error metrics
    # ------------------------------

    absolute_error = np.abs(
        pinn_drawdown -
        reference_drawdown
    )

    mae = np.mean(absolute_error)

    rmse = np.sqrt(
        np.mean(
            (
                pinn_drawdown -
                reference_drawdown
            ) ** 2
        )
    )

    max_error = np.max(absolute_error)


    # ------------------------------
    # Well drawdown
    # ------------------------------

    reference_well = reference_drawdown[
        WELL_I,
        WELL_J
    ]

    pinn_well = pinn_drawdown[
        WELL_I,
        WELL_J
    ]


    # ------------------------------
    # Integrated drawdown
    # ------------------------------

    reference_integrated = (
        np.sum(reference_drawdown)
        * CELL_AREA_STAR
    )

    pinn_integrated = (
        np.sum(pinn_drawdown)
        * CELL_AREA_STAR
    )


    # ------------------------------
    # Diagnostic output
    # ------------------------------

    print(
        f"Day {day:2d} | "
        f"PINN well = {pinn_well:.6f} | "
        f"Reference well = {reference_well:.6f} | "
        f"MAE = {mae:.6f}"
    )


    results.append([
        day,
        mae,
        rmse,
        max_error,
        reference_well,
        pinn_well,
        reference_integrated,
        pinn_integrated
    ])


# --------------------------------------------------
# Save results
# --------------------------------------------------

final_evaluation = pd.DataFrame(
    results,
    columns=[
        "Day",
        "MAE",
        "RMSE",
        "Max_Absolute_Error",
        "Reference_Well_Drawdown",
        "PINN_Well_Drawdown",
        "Reference_Integrated_Drawdown",
        "PINN_Integrated_Drawdown"
    ]
)

final_evaluation.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("Final evaluation complete.")
print()
print(final_evaluation)
print()
print(f"Saved to: {OUTPUT_PATH}")