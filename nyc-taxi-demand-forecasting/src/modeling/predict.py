"""Phase 10: generate predictions from a fitted model, in real-count units.

Both Phase 10 models are trained on `log1p(pickup_count)` (see
`src/features.py`) -- `predict_demand` back-transforms with `expm1` so
callers always get real pickup counts, never log-space values. This matters
for every downstream use: Phase 11's MAE/RMSE/R² must be computed on real
counts, and DTDI (`predicted demand / baseline demand`) needs real counts too.

Usage:
    python -m src.modeling.predict            # predict on the test set for both models
"""

from loguru import logger
import numpy as np
import pandas as pd
import typer

from src.config import MODELS_DIR
from src.modeling.train import MODEL_REGISTRY, load_model


def predict_demand(model, X: pd.DataFrame) -> np.ndarray:
    """Predict and back-transform in one step -- returns real pickup counts
    (non-negative, since expm1 of any real log1p-space prediction is >= -1,
    and predictions are clipped at 0 since a negative predicted count isn't
    meaningful)."""
    log1p_predictions = model.predict(X)
    return np.clip(np.expm1(log1p_predictions), a_min=0, a_max=None)


app = typer.Typer()


@app.command()
def main():
    """Load every saved model in MODEL_REGISTRY and predict on the test set."""
    from src.features import build_feature_matrix, split_train_test
    from src.geospatial import load_processed_zone_hour

    processed_df = load_processed_zone_hour()
    _train_df, test_df = split_train_test(processed_df)

    for model_name, spec in MODEL_REGISTRY.items():
        model_path = MODELS_DIR / f"{model_name}.joblib"
        model = load_model(model_path)
        X_test, y_test = build_feature_matrix(test_df, spec["feature_set"])
        predictions = predict_demand(model, X_test)
        actual = np.expm1(y_test.to_numpy())
        logger.info(
            f"[{model_name}] predicted {len(predictions):,} rows -- "
            f"pred mean={predictions.mean():.2f}, actual mean={actual.mean():.2f}"
        )


if __name__ == "__main__":
    app()
