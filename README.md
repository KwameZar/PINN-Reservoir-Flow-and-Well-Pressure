# PINN-Reservoir-Flow-and-Well-Pressure

Physics-Informed Neural Network for surrogate modeling of well-driven flow and pressure drawdown in a heterogeneous reservoir.

## Project Overview

This project develops a Physics-Informed Neural Network (PINN) surrogate model for predicting pressure drawdown in a heterogeneous reservoir with production from a localized well.

The project combines reservoir-flow physics, numerical simulation, and deep learning to investigate whether a neural network can reproduce the behavior of a conventional numerical reservoir-flow solution while incorporating the governing physical laws directly into model training.

The final model represents a 1,000 m × 1,000 m heterogeneous reservoir containing a production well at the center of the domain. Spatial heterogeneity is represented through a varying permeability field, while the production well is represented using a smooth distributed Gaussian source term.

A finite-difference numerical simulation is used to generate the reference solution. The PINN is then trained using the governing flow equation together with the initial condition, no-flow boundary conditions, and a global mass-conservation constraint.

The trained model is evaluated by comparing its predicted drawdown field against the numerical reference solution at multiple stages of a 30-day simulation.

---

## Research Motivation

Conventional numerical reservoir simulation provides physically meaningful solutions but can become computationally expensive when simulations must be repeated for forecasting, optimization, uncertainty analysis, or parameter studies.

Physics-Informed Neural Networks provide an alternative approach in which physical governing equations are incorporated directly into the learning objective.

The objective of this project is therefore to investigate a physics-informed surrogate modeling workflow for heterogeneous reservoir flow, with particular emphasis on:

- Incorporating governing physics into neural-network training
- Representing heterogeneous reservoir properties
- Modeling localized production effects
- Enforcing initial and boundary conditions
- Incorporating global mass conservation
- Evaluating PINN predictions against a numerical reference solution
- Understanding the limitations of PINNs for transient reservoir-flow problems

---

## Physical Problem

The reservoir is represented as a two-dimensional square domain:

- Reservoir length: 1,000 m
- Reservoir width: 1,000 m
- Spatial grid spacing: 20 m
- Grid dimensions: 51 × 51
- Simulation duration: 30 days
- Production well location: (500 m, 500 m)

The reservoir permeability is spatially heterogeneous.

The heterogeneous permeability field is represented using a spatial variation around a nominal permeability of approximately 100 mD.

The production well is located at the center of the reservoir and produces continuously throughout the simulation.

Rather than representing the well as a single mathematical point source or imposing a pressure directly at the well, the final formulation represents production using a smooth Gaussian source distribution around the well location.

This provides a numerically smooth representation of the localized production process and is suitable for incorporation into the PINN residual.

---

## Governing Physics

The model is formulated using a pressure-diffusion equation with spatially varying permeability.

For the heterogeneous reservoir, the normalized drawdown formulation is written as:

$$
s_{t^*}
-
\alpha^*
\left[
\frac{\partial}{\partial x^*}
\left(
k_r
\frac{\partial s}{\partial x^*}
\right)
+
\frac{\partial}{\partial y^*}
\left(
k_r
\frac{\partial s}{\partial y^*}
\right)
\right]
-
S
=
0
$$

where:

- $s$ is normalized pressure drawdown
- $x^*$ and $y^*$ are normalized spatial coordinates
- $t^*$ is normalized time
- $k_r$ is the normalized heterogeneous permeability field
- $\alpha^*$ is the normalized diffusivity
- $S$ is the distributed production source term

The drawdown is defined relative to the initial pressure field:

$$
s = 1-p
$$

where $p$ represents normalized pressure.

The formulation therefore allows the production-induced pressure depletion to be modeled directly.

---

## Initial and Boundary Conditions

The reservoir initially has a uniform normalized pressure:

$$
p(x,y,0)=1
$$

or equivalently:

$$
s(x,y,0)=0
$$

The outer reservoir boundaries are treated as no-flow boundaries.

For drawdown, this corresponds to zero normal pressure-gradient conditions:

$$
\frac{\partial s}{\partial n}=0
$$

The PINN incorporates these conditions into its training objective.

---

## Production Well Representation

The production well is located at:

$$
(x_w,y_w)=(0.5,0.5)
$$

in normalized coordinates.

The well source is represented using a Gaussian distribution:

$$
S(x,y)=A
\exp
\left[
-\frac{(x-x_w)^2+(y-y_w)^2}
{2\sigma_w^2}
\right]
$$

with a characteristic well-region width:

$$
\sigma_w=0.02
$$

The source amplitude is normalized so that the integrated source strength over the computational grid corresponds to a total normalized drawdown contribution of 0.5 over the 30-day simulation.

This formulation avoids the numerical singularity associated with an idealized point source while maintaining a localized production effect.

---

## Numerical Reference Solution

A finite-difference solver is used to generate the reference solution against which the PINN is evaluated.

The heterogeneous diffusion operator is implemented in conservative flux form using face-centered permeability values.

The numerical solution uses:

- 51 × 51 spatial grid
- 20 m spatial resolution
- 300 s time step
- 30-day simulation period
- Spatially heterogeneous permeability
- Gaussian production source
- No-flow outer boundaries

The time step was selected based on the stability characteristics of the heterogeneous diffusion problem.

The reference solution provides the numerical field used for PINN validation.

A global mass-conservation check was also performed on the numerical reference solution.

The integrated drawdown follows the expected cumulative source behavior:

| Time | Integrated Drawdown |
|---:|---:|
| 1 day | 0.016667 |
| 5 days | 0.083333 |
| 10 days | 0.166667 |
| 15 days | 0.250000 |
| 20 days | 0.333333 |
| 25 days | 0.416667 |
| 30 days | 0.500000 |

The numerical mass-balance error was approximately at machine precision.

---

## PINN Methodology

The PINN approximates the reservoir drawdown field using a neural network:

$$
s_\theta(x,y,t)
$$

where $\theta$ represents the trainable neural-network parameters.

The network receives normalized spatial and temporal coordinates and predicts normalized drawdown.

### Network Architecture

The final PINN consists of:

- Input layer: 3 variables $(x^*,y^*,t^*)$
- 4 fully connected hidden layers
- 50 neurons per hidden layer
- Hyperbolic tangent activation functions
- Single output neuron
- Approximately 7,901 trainable parameters

### Physics-Informed Loss

The final training objective combines several components:

$$
L =
L_{physics}
+
10L_{IC}
+
L_{BC}
+
25L_{mass}
$$

where:

- $L_{physics}$ enforces the governing PDE
- $L_{IC}$ enforces the initial condition
- $L_{BC}$ enforces the no-flow boundary conditions
- $L_{mass}$ enforces global mass conservation

The physics residual is evaluated using automatic differentiation.

The mass-conservation term compares the spatially integrated PINN drawdown against the expected cumulative source contribution.

---

## Training Strategy

The final model was trained using the Adam optimizer.

Training configuration:

- Optimizer: Adam
- Initial learning rate: 0.001
- Training iterations: 5,000 epochs
- Physics collocation points: 12,000
- Dedicated well-region points: 2,000
- Initial-condition weight: 10
- Mass-conservation weight: 25

The physics collocation set combines randomly distributed points with additional points concentrated around the production well to improve representation of the localized source region.

The best model weights were restored after training and saved as the final model.

---

## Model Development and Experimentation

The project was developed progressively rather than training only a single final model.

### Experiment 1 — 1D Reservoir Flow

A one-dimensional pressure-diffusion problem was first implemented to establish the numerical and PINN workflow.

This provided a controlled environment for testing:

- PDE formulation
- Numerical solution generation
- Neural-network training
- Automatic differentiation
- PINN validation

### Experiment 2 — 2D Homogeneous Reservoir

The problem was extended to a two-dimensional homogeneous reservoir.

The baseline PINN reproduced the numerical reference solution with approximately:

- MAE: 0.00124
- RMSE: 0.00176

This established the baseline capability of the PINN framework.

### Experiment 3 — Physical Reservoir Parameters

The model was then reformulated using physical reservoir parameters, including permeability, porosity, viscosity, and total compressibility.

Physical parameters were converted into normalized quantities for stable neural-network training.

### Experiment 4 — Heterogeneous Reservoir

Spatially varying permeability was introduced.

The model was trained using a conservative heterogeneous diffusion formulation.

Boundary-loss weighting was investigated because the initial unweighted model produced larger errors in parts of the domain.

### Experiment 5 — Well-Driven Reservoir

The final experiment introduced a localized production well.

Several formulations were investigated, including:

- Direct pressure boundary/source representations
- Discrete multi-cell source representations
- Smooth distributed Gaussian source representation
- Different conservation-loss weights
- Initial-condition weighting
- Hard versus soft initial-condition enforcement

The final Gaussian-source formulation was selected because it provided a consistent localized source representation suitable for both the numerical reference solver and PINN.

---

## Model Selection

The final model uses:

$$
L =
L_{physics}
+
10L_{IC}
+
L_{BC}
+
25L_{mass}
$$

The mass-conservation weight was selected after comparing alternative conservation-weight configurations.

The final model was selected based on its ability to reproduce the numerical reference solution while maintaining physically meaningful behavior and improved global mass conservation.

The best-performing model weights were restored after training before being saved as the final model.

---

## Validation

The PINN is evaluated against the numerical reference solution at multiple simulation times.

The primary evaluation metrics are:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Maximum Absolute Error
- Production-well drawdown
- Integrated drawdown

The evaluation considers the complete 51 × 51 spatial field rather than only the production-well location.

### Final Evaluation Results

The saved final model was evaluated at 1, 5, 10, 15, 20, 25, and 30 days.

| Day | MAE | RMSE | Max. Absolute Error | Reference Well | PINN Well |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.052761 | 0.059253 | 0.110312 | 0.411816 | 0.508716 |
| 5 | 0.035007 | 0.041379 | 0.075092 | 0.564699 | 0.564822 |
| 10 | 0.017469 | 0.022571 | 0.047870 | 0.654143 | 0.635927 |
| 15 | 0.012105 | 0.014176 | 0.037817 | 0.735392 | 0.707452 |
| 20 | 0.028135 | 0.030007 | 0.075245 | 0.815727 | 0.778665 |
| 25 | 0.050309 | 0.052063 | 0.114280 | 0.895888 | 0.848895 |
| 30 | 0.073897 | 0.076188 | 0.154425 | 0.976003 | 0.917589 |

### Day-15 Performance

At 15 days, the final saved model produces:

- Maximum PINN drawdown: 0.7075
- Maximum reference drawdown: 0.7354
- MAE: 0.0121
- RMSE: 0.0142
- Maximum absolute error: 0.0378
- PINN well drawdown: 0.7075
- Reference well drawdown: 0.7354

The Day-15 result represents the lowest full-field MAE among the evaluated time points.

### Integrated Drawdown

Global integrated drawdown provides an additional measure of how well the PINN captures the cumulative production effect.

| Day | Reference | PINN |
|---:|---:|---:|
| 1 | 0.016667 | 0.071046 |
| 5 | 0.083333 | 0.119694 |
| 10 | 0.166667 | 0.181167 |
| 15 | 0.250000 | 0.242822 |
| 20 | 0.333333 | 0.304061 |
| 25 | 0.416667 | 0.364325 |
| 30 | 0.500000 | 0.423118 |

The PINN reproduces the overall drawdown evolution but does not maintain exact global mass conservation throughout the full simulation.

This is an important limitation of the final model and demonstrates the trade-off between local field accuracy and global conservation in the trained PINN.

---

## Results

The repository contains visual comparisons between the PINN and numerical reference solutions, including:

- Day-15 drawdown fields
- Day-30 drawdown fields
- Day-15 absolute-error field
- Day-30 absolute-error field
- PINN training loss
- Production-well drawdown history

The final evaluation results are stored in:

`Results/well_ic10_mass25_final_evaluation.csv`

The trained model is stored in:

`Models/pinn_well_ic10_mass25.keras`

---

## Normalization and Units

The neural network operates using normalized coordinates and variables.

Spatial coordinates are normalized as:

$$
x^*=\frac{x}{L_x}
$$

$$
y^*=\frac{y}{L_y}
$$

Time is normalized using the 30-day simulation period:

$$
t^*=\frac{t}{T}
$$

where:

$$
T=30\text{ days}
$$

The application accepts time in physical days and performs the normalization internally before passing the inputs to the trained PINN.

The model output is **normalized pressure drawdown**, not pressure in psi, MPa, or another dimensional pressure unit.

Therefore:

> Higher normalized drawdown indicates greater pressure depletion relative to the initial normalized pressure field.

---

## Interactive Application

A Streamlit application is included to provide an interactive demonstration of the trained PINN.

The application allows the user to select the simulation time and view:

1. PINN-predicted reservoir drawdown
2. Numerical reference drawdown
3. Maximum predicted drawdown
4. Maximum reference drawdown
5. MAE
6. RMSE
7. Production-well drawdown history

The application evaluates the saved final PINN model directly.

---

## Repository Structure

```text
PINN-Reservoir-Flow-and-Well-Pressure/

├── App/
│   └── app.py
│
├── Data/
│   ├── reference_2d_pressure.csv
│   ├── reference_physical_reservoir.npz
│   ├── reference_heterogeneous_reservoir.npz
│   └── reference_well_driven_reservoir.npz
│
├── Models/
│   ├── pinn_well_ic10_mass25.keras
│   └── Archive/
│       ├── pinn_2d_baseline.keras
│       ├── pinn_heterogeneous_boundary_weighted.keras
│       └── pinn_well_conservation_lambda10.keras
│
├── Notebooks/
│   ├── 04_well_driven_reservoir_pinn.ipynb
│   └── Archive/
│
├── Results/
│   ├── well_ic10_mass25_final_evaluation.csv
│   ├── well_ic10_mass25_training_history.npz
│   ├── well_driven_pinn_training_loss.png
│   ├── well_driven_pinn_drawdown_t15.png
│   ├── well_driven_pinn_drawdown_t30.png
│   ├── well_driven_pinn_absolute_error_t15.png
│   ├── well_driven_pinn_absolute_error_t30.png
│   └── well_drawdown_reference_vs_pinn.png
│
├── Src/
│   ├── __init__.py
│   ├── model.py
│   ├── prediction.py
│   ├── reference.py
│   └── visualization.py
│
├── .gitignore
├── .gitattributes
├── README.md
└── final_evaluation_model.py