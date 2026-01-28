from __future__ import annotations

from typing import Dict

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import ARTIFACTS_DIR


def train_models(X_train, y_train) -> Dict[str, object]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=2000),
        "rf": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=42,
        ),
        "gbdt": GradientBoostingClassifier(random_state=42),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, ARTIFACTS_DIR / f"{name}.joblib")
        trained[name] = model

    return trained
