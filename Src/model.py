import tensorflow as tf
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "Models"
    / "pinn_well_ic10_mass25.keras"
)

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained PINN model not found: {MODEL_PATH}"
    )

model = load_model()

