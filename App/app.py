import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from Src.visualization import (
    predict_reservoir_drawdown,
    get_well_drawdown_history
)

st.set_page_config(
    page_title="PINN Reservoir Flow",
    page_icon="🌐",
    layout="wide"
)

st.title("PINN Reservoir Flow Simulator")

st.write(
    "Physics-Informed Neural Network for well-driven reservoir "
    "drawdown prediction."
)

st.info(
    "The model represents a 1,000 m × 1,000 m heterogeneous reservoir "
    "with a production well located at the center (500 m, 500 m)."
)

time_days = st.slider(
    "Simulation time (days)",
    min_value=0.0,
    max_value=30.0,
    value=15.0,
    step=1.0
)

st.write(f"Selected time: **{time_days:.0f} days**")

X, Y, drawdown, reference_drawdown, well_x, well_y = predict_reservoir_drawdown(time_days)

mae = np.mean(
    np.abs(drawdown - reference_drawdown)
)

rmse = np.sqrt(
    np.mean(
        (drawdown - reference_drawdown) ** 2
    )
)

max_reference = reference_drawdown.max()
max_predicted = drawdown.max()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Max PINN Drawdown (normalized)",
    f"{max_predicted:.4f}"
)

col2.metric(
    "Max Reference Drawdown (normalized)",
    f"{max_reference:.4f}"
)

col3.metric(
    "MAE",
    f"{mae:.4f}"
)

col4.metric(
    "RMSE",
    f"{rmse:.4f}"
)

st.caption(
    "Drawdown represents the normalized reduction in reservoir pressure "
    "relative to the initial pressure field. Higher values indicate "
    "greater pressure depletion."
)

st.info(
    "Time is entered in physical days. The PINN internally normalizes "
    "time for model prediction. Reported drawdown values are normalized "
    "drawdown, not pressure units such as psi or MPa."
)

st.subheader("Reservoir Drawdown Comparison")

st.write(
    "Comparison between the PINN prediction and the numerical "
    f"reference solution at {time_days:.0f} days."
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

# PINN prediction
image1 = axes[0].imshow(
    drawdown.T,
    extent=[0, 1000, 0, 1000],
    origin="lower",
    aspect="equal"
)

axes[0].scatter(
    well_x,
    well_y,
    marker="*",
    s=180,
    label="Production well"
)

axes[0].set_xlabel("X distance (m)")
axes[0].set_ylabel("Y distance (m)")
axes[0].set_title("PINN Prediction")
axes[0].legend()

fig.colorbar(
    image1,
    ax=axes[0],
    label="Normalized drawdown"
)


# Numerical reference
image2 = axes[1].imshow(
    reference_drawdown.T,
    extent=[0, 1000, 0, 1000],
    origin="lower",
    aspect="equal"
)

axes[1].scatter(
    well_x,
    well_y,
    marker="*",
    s=180,
    label="Production well"
)

axes[1].set_xlabel("X distance (m)")
axes[1].set_ylabel("Y distance (m)")
axes[1].set_title("Numerical Reference")
axes[1].legend()

fig.colorbar(
    image2,
    ax=axes[1],
    label="Normalized drawdown"
)

st.pyplot(fig)

plt.close(fig)

st.success("PINN prediction generated successfully.")

st.subheader("Production Well Drawdown History")

time_history, pinn_history, reference_history = (
    get_well_drawdown_history()
)

fig_history, ax_history = plt.subplots(
    figsize=(10, 5)
)

ax_history.plot(
    time_history,
    pinn_history,
    label="PINN"
)

ax_history.plot(
    time_history,
    reference_history,
    label="Numerical Reference"
)

ax_history.set_xlabel("Time (days)")
ax_history.set_ylabel("Normalized drawdown")

ax_history.set_title(
    "Drawdown at Production Well"
)

ax_history.legend()

ax_history.grid(True)

st.pyplot(fig_history)

plt.close(fig_history)