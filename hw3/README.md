# Partially Linear Regression (PLR) Simulation Study

This repository contains an implementation of the PLR simulation study for both low-dimensional and high-dimensional settings, including both the naïve plug-in OLS and Double Machine Learning (DML) estimators.

## Files Overview

- `plr_simulation.py` - Main simulation implementation (sequential, with parallelization via joblib)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Model Settings

### Setting 1: Low-dimensional, nonlinear nuisances
- **Parameters**: n=500, p=10
- **Data**: X ~ N(0, I_p)
- **Nuisance functions**: 
  - g₀(x) = sin(x₁) + x₂²
  - e₀(x) = 0.5x₃ - 0.3x₄
- **ML methods**: Random Forest (RF) for naïve, RF/GBM for DML

### Setting 2: High-dimensional, linear sparse nuisances
- **Parameters**: n ∈ {500, 1000}, p ∈ {800, 1200}, s ∈ {10, 30}
- **Data**: X ~ N(0, Σ_X) where (Σ_X)_jk = 0.5^|j-k|
- **Nuisance functions**: 
  - g₀(x) = x^T β₀ (s-sparse)
  - e₀(x) = x^T γ₀ (s-sparse)
- **ML methods**: Lasso for naïve, Lasso/Post-Lasso for DML

## Estimators

### 1. Naïve Plug-in OLS
1. Fit ĝ(X) using ML (RF for Setting 1, Lasso for Setting 2)
2. Regress Y - ĝ(X) on W

### 2. DML (Double Machine Learning)
1. Fit ĝ(X) ≈ E[Y | X] and ê(X) ≈ E[W | X] using K-fold cross-fitting
2. Form residuals: Ỹ = Y - ĝ(X) and Ŵ = W - ê(X)
3. Regress Ỹ on Ŵ

## Usage

### Quick Example
```python
python plr_simulation.py
```

### Custom Parameters
```python
from plr_simulation import PLRSimulation

sim = PLRSimulation(random_state=42)

# Run Setting 1 with custom parameters
results1 = sim.run_simulation_setting1(n=500, p=10, rho_values=[0.0, 0.3], R=100)

# Run Setting 2 with custom parameters
results2 = sim.run_simulation_setting2(
    n_values=[500, 1000], 
    p_values=[800, 1200], 
    s_values=[10, 30], 
    rho_values=[0.0, 0.3], 
    R=100
)
```

## Output Metrics

For each method and parameter setting, the simulation reports:

- **Bias**: E[θ̂] - θ₀
- **Variance**: Var(θ̂)
- **RMSE**: √E[(θ̂ - θ₀)²]
- **Coverage**: Empirical coverage of 95% CIs
- **CI Length**: Average confidence interval length

For Setting 2 only (DML method):
- **Average Sparsity**: ŝ_g, ŝ_e (selected model sizes)

## Key Features

1. Both model settings and both estimators
2. Parallel processing using joblib
3. Proper cross-fitting for DML
4. Confidence intervals with plug-in variance estimator
5. Sparsity tracking for high-dimensional setting
6. Reproducibility via random seeds

## Dependencies

See `requirements.txt` for the full list.

## License

MIT License
