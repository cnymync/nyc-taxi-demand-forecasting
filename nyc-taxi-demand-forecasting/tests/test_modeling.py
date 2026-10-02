import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from src.modeling.predict import predict_demand
from src.modeling.train import (
    MODEL_REGISTRY,
    fit_gradient_boosting,
    fit_lasso_regression,
    fit_linear_regression,
    fit_ridge_regression,
    load_model,
    save_model,
    tune_lasso_alpha,
)


def _tiny_features_and_target():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "baseline_pickup_count": rng.uniform(0, 20, 200),
            "tmpf": rng.uniform(20, 90, 200),
            "hour_of_day_0": rng.integers(0, 2, 200).astype(bool),
        }
    )
    y = pd.Series(
        np.log1p(
            X["baseline_pickup_count"]
            + rng.normal(0, 1, 200).clip(min=-X["baseline_pickup_count"])
        )
    )
    return X, y


def test_fit_linear_regression_produces_a_fitted_model():
    X, y = _tiny_features_and_target()
    model = fit_linear_regression(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_fit_ridge_regression_produces_a_fitted_model():
    X, y = _tiny_features_and_target()
    model = fit_ridge_regression(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_ridge_shrinks_coefficients_relative_to_unregularised_linear_regression():
    X, y = _tiny_features_and_target()
    linear = fit_linear_regression(X, y)
    ridge = fit_ridge_regression(X, y, alpha=10.0)  # strong regularisation for a clear effect
    assert np.abs(ridge.coef_).sum() < np.abs(linear.coef_).sum()


def test_fit_gradient_boosting_produces_a_fitted_model():
    X, y = _tiny_features_and_target()
    model = fit_gradient_boosting(X, y, max_iter=10)  # small for a fast test
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_fit_lasso_regression_produces_a_fitted_model():
    X, y = _tiny_features_and_target()
    model = fit_lasso_regression(X, y, alpha=0.01)
    predictions = model.predict(X)
    assert len(predictions) == len(y)


def test_fit_lasso_regression_has_no_default_alpha():
    """`alpha` must be supplied explicitly -- an untuned default would defeat
    the whole point of `tune_lasso_alpha` existing."""
    X, y = _tiny_features_and_target()
    with pytest.raises(TypeError):
        fit_lasso_regression(X, y)


def test_lasso_higher_alpha_zeros_out_an_irrelevant_feature():
    X, y = _tiny_features_and_target()
    rng = np.random.default_rng(1)
    X = X.assign(pure_noise=rng.normal(0, 1, len(X)))  # unrelated to y by construction

    weak = fit_lasso_regression(X, y, alpha=1e-5)
    strong = fit_lasso_regression(X, y, alpha=1.0)
    noise_idx = list(X.columns).index("pure_noise")
    assert weak.coef_[noise_idx] != 0.0
    assert strong.coef_[noise_idx] == 0.0
    assert np.sum(strong.coef_ != 0) <= np.sum(weak.coef_ != 0)


def test_tune_lasso_alpha_fits_on_train_sub_only_and_scores_on_validation():
    """The leakage-safety property that matters: fit happens on
    `X_train_sub`/`y_train_sub` only, `X_validation`/`y_validation` is used
    only to *score* the already-fit model -- reproduced independently here
    (fit once outside the function, predict on validation) and compared to
    what `tune_lasso_alpha` reports for the same single alpha."""
    X_train_sub, y_train_sub = _tiny_features_and_target()
    rng = np.random.default_rng(2)
    X_validation = pd.DataFrame(
        {
            "baseline_pickup_count": rng.uniform(0, 20, 50),
            "tmpf": rng.uniform(20, 90, 50),
            "hour_of_day_0": rng.integers(0, 2, 50).astype(bool),
        }
    )
    y_validation = pd.Series(np.log1p(X_validation["baseline_pickup_count"]))
    alpha = 0.01

    best_alpha, results_df = tune_lasso_alpha(
        X_train_sub, y_train_sub, X_validation, y_validation, alpha_grid=np.array([alpha])
    )
    assert best_alpha == alpha
    assert len(results_df) == 1

    model = fit_lasso_regression(X_train_sub, y_train_sub, alpha=alpha)
    expected_predicted = np.clip(np.expm1(model.predict(X_validation)), a_min=0, a_max=None)
    expected_mae = mean_absolute_error(np.expm1(y_validation.to_numpy()), expected_predicted)
    assert results_df.loc[0, "validation_MAE"] == pytest.approx(expected_mae)


def test_tune_lasso_alpha_picks_the_alpha_with_lowest_validation_mae():
    X_train_sub, y_train_sub = _tiny_features_and_target()
    X_validation, y_validation = _tiny_features_and_target()

    best_alpha, results_df = tune_lasso_alpha(
        X_train_sub,
        y_train_sub,
        X_validation,
        y_validation,
        alpha_grid=np.array([1e-5, 0.01, 5.0]),
    )
    assert best_alpha == results_df.loc[results_df["validation_MAE"].idxmin(), "alpha"]
    assert best_alpha in {1e-5, 0.01, 5.0}


def test_model_registry_has_all_models_with_valid_feature_sets():
    assert set(MODEL_REGISTRY) == {"linear_regression", "ridge", "gradient_boosting"}
    for name, spec in MODEL_REGISTRY.items():
        assert spec["feature_set"] in {"linear", "gradient_boosting"}, name
        assert callable(spec["fit_fn"]), name


def test_save_and_load_model_round_trips(tmp_path):
    X, y = _tiny_features_and_target()
    model = fit_linear_regression(X, y)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_predict_demand_back_transforms_and_clips_at_zero():
    X, y = _tiny_features_and_target()
    model = fit_linear_regression(X, y)
    predictions = predict_demand(model, X)
    assert (predictions >= 0).all()  # never a negative pickup count
    # Back-transform should differ from the raw (log-space) model output.
    raw_log_predictions = model.predict(X)
    assert not np.allclose(predictions, raw_log_predictions)
