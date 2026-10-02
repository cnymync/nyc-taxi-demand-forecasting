"""Phase 10: fit the working benchmark models.

Scope, locked 2026-08-17:
Phase 10's job is narrow -- fit + predict on the existing train/test split, no
hyperparameter tuning, no full evaluation. Phase 11 (`notebooks/07_post_model_figures_and_statistics.ipynb`,
`src/evaluate.py`) is where MAE/RMSE/R², error breakdowns, and DTDI
comparison happen. Kept as two separate notebooks even though built in the
same session, per explicit user instruction.

**Round 2 (2026-08-17, same day)**: after Phase 11 diagnosed the Linear
Regression scale-mismatch finding, the user asked to (a) apply the fix, and
(b) add more models to see if accuracy improves. Four models, two per
feature-set family:

- **linear** feature set (`borough` + `baseline_pickup_count`): Linear
  Regression (unregularised, original benchmark) and **Ridge Regression**
  (L2-regularised -- shrinks extreme coefficients, a natural additional
  safeguard against the same class of instability the log1p-baseline fix
  addresses more directly).
- **gradient_boosting** feature set (full `pu_location_id`, native
  categorical encoding): **HistGradientBoostingRegressor** (sklearn's
  histogram-based gradient boosting, fast at this scale via binning
  rather than exhaustive per-node splits).

LASSO Regression (L1-regularised, on the same `linear` feature set,
standardized) is fit separately below with its own tuned `alpha`.

Usage:
    python -m src.modeling.train            # fit + save all models
"""

from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
import typer

from src.config import MODELS_DIR

# sklearn default alpha=1.0 -- a first-pass, undecided value like every
# other hyperparameter here, not tuned against the validation slice yet.
RIDGE_PARAMS = {"alpha": 1.0, "random_state": 42}

# sklearn defaults (learning_rate=0.1, max_iter=100, max_leaf_nodes=31) plus
# a fixed random_state -- also a first-pass, undecided configuration.
# `categorical_features="from_dtype"` is explicit here even though it's
# sklearn's own default, because it's load-bearing: this model must be fit
# on the "gradient_boosting" feature set (src/features.py -- native pandas
# 'category' dtype columns, NOT one-hot). One-hot-encoding pu_location_id
# (263 columns) instead made 30 boosting iterations alone take over 2 hours
# in a real timing test (2026-08-17) --
# native categorical support is not an optimisation here, it's the
# difference between tractable and not at this scale.
GRADIENT_BOOSTING_PARAMS = {"random_state": 42, "categorical_features": "from_dtype"}


def fit_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit the interpretable benchmark. No scaling -- feature scaling for
    Linear Regression is still an open question; the log1p-baseline fix
    (src/features.py) addresses the specific scale mismatch that was
    actually causing problems."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fit_ridge_regression(X_train: pd.DataFrame, y_train: pd.Series, **params) -> Ridge:
    """L2-regularised Linear Regression -- shrinks extreme coefficients
    toward zero, a general-purpose safeguard against the coefficient-
    instability class of problem, on the same `linear` feature set."""
    model = Ridge(**{**RIDGE_PARAMS, **params})
    model.fit(X_train, y_train)
    return model


# Regularization strength, not an L1/L2 mixing ratio (that's ElasticNet's
# `l1_ratio`, deliberately not used here per explicit user instruction --
# 2026-08-22). Log-spaced from near-zero (~OLS, no sparsity) to a value
# large enough to plausibly zero out most coefficients, given the
# standardized "linear" feature set's coefficients were seen to be small
# (Ridge's on the same features: 0.01-1.2) -- `alpha=1.0`, sklearn's Ridge
# default, is already a comparatively strong relative penalty at this
# feature scale, so the grid deliberately extends below it, not just above.
LASSO_ALPHA_GRID = np.logspace(-5, 1, 25)

# max_iter raised well above sklearn's default (1000): coordinate descent
# converges slowly for small alpha (close to unpenalized OLS) on a 9.2M-row
# training set -- found by observing ConvergenceWarning at the default
# during a real tuning run.
LASSO_PARAMS = {"random_state": 42, "max_iter": 10_000}


def fit_lasso_regression(
    X_train: pd.DataFrame, y_train: pd.Series, alpha: float, **params
) -> Lasso:
    """L1-regularised Linear Regression on the standardized `linear` feature
    set (see `src.features.fit_standard_scaler`/`apply_standard_scaler` --
    LASSO is scale-sensitive the same way Ridge is). `alpha` has no default
    on purpose: the caller must pass a value chosen by `tune_lasso_alpha`
    (or deliberately override it), rather than silently reusing an untuned
    default the way `RIDGE_PARAMS` still does."""
    model = Lasso(alpha=alpha, **{**LASSO_PARAMS, **params})
    model.fit(X_train, y_train)
    return model


def tune_lasso_alpha(
    X_train_sub: pd.DataFrame,
    y_train_sub: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    alpha_grid: np.ndarray = LASSO_ALPHA_GRID,
) -> tuple[float, pd.DataFrame]:
    """Leakage-safe LASSO alpha search.

    Fits each candidate alpha on `X_train_sub`/`y_train_sub` only, scores on
    `X_validation`/`y_validation` -- never on the 2026 test set. Callers
    must pass `split_train_validation`'s output here, the same discipline
    `compute_dtdi_percentile_thresholds` (src/evaluate.py) already applies
    to DTDI's own thresholds: tuning-relevant statistics come from training
    data only, evaluation-relevant statistics never leak backward into a
    decision made before the model saw them.

    Selection criterion is real-count-scale MAE (`expm1` back-transform,
    matching `src.modeling.predict.predict_demand`'s convention) -- MAE is
    this project's primary reported regression metric throughout.

    Returns `(best_alpha, results_df)`; `results_df` has one row per
    candidate alpha with its validation MAE and the number of non-zero
    coefficients at that alpha, so the sparsity/accuracy trade-off across
    the whole grid is visible, not just the single chosen point.
    """
    actual_validation = np.expm1(y_validation.to_numpy())
    rows = []
    for alpha in alpha_grid:
        model = fit_lasso_regression(X_train_sub, y_train_sub, alpha=alpha)
        predicted = np.clip(np.expm1(model.predict(X_validation)), a_min=0, a_max=None)
        rows.append(
            {
                "alpha": alpha,
                "validation_MAE": mean_absolute_error(actual_validation, predicted),
                "n_nonzero_coefs": int(np.sum(model.coef_ != 0)),
            }
        )
    results_df = pd.DataFrame(rows)
    best_alpha = float(results_df.loc[results_df["validation_MAE"].idxmin(), "alpha"])
    return best_alpha, results_df


def fit_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series, **params
) -> HistGradientBoostingRegressor:
    """Fit sklearn's histogram-based gradient boosting on the
    `gradient_boosting` feature set (full zone identity, native categorical
    encoding) -- added 2026-08-17 alongside Ridge to see whether either
    beats the original two models."""
    model = HistGradientBoostingRegressor(**{**GRADIENT_BOOSTING_PARAMS, **params})
    model.fit(X_train, y_train)
    return model


# Registry driving both this module's main() and src/evaluate.py's
# build_evaluation_frame -- add a model here once and both pick it up.
MODEL_REGISTRY = {
    "linear_regression": {"feature_set": "linear", "fit_fn": fit_linear_regression},
    "ridge": {"feature_set": "linear", "fit_fn": fit_ridge_regression},
    "gradient_boosting": {"feature_set": "gradient_boosting", "fit_fn": fit_gradient_boosting},
}


def save_model(model, path: Path) -> None:
    """Persist a fitted model to `path` as a `.joblib` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")


def load_model(path: Path):
    """Load a model previously saved by `save_model`."""
    return joblib.load(path)


app = typer.Typer()


@app.command()
def main():
    """Fit every model in MODEL_REGISTRY on the full training set and save them."""
    from src.features import build_feature_matrix, split_train_test
    from src.geospatial import load_processed_zone_hour

    processed_df = load_processed_zone_hour()
    train_df, _test_df = split_train_test(processed_df)

    feature_matrices = {}
    for model_name, spec in MODEL_REGISTRY.items():
        feature_set = spec["feature_set"]
        if feature_set not in feature_matrices:
            feature_matrices[feature_set] = build_feature_matrix(train_df, feature_set)
        X_train, y_train = feature_matrices[feature_set]

        logger.info(f"Fitting {model_name} ({feature_set} feature set, {X_train.shape})...")
        model = spec["fit_fn"](X_train, y_train)
        save_model(model, MODELS_DIR / f"{model_name}.joblib")

    logger.success("Phase 10 benchmark models fit and saved.")


if __name__ == "__main__":
    app()
