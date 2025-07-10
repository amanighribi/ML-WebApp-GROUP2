from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def train_model(model_name, X_train, y_train, params):
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None)
        )
    elif model_name == "SVM":
        model = SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            probability=True
        )
    elif model_name == "XGBoost":
        model = XGBClassifier(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
    else:
        raise ValueError("Modèle non reconnu.")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, feature_names=None):
    y_pred = model.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    # Ajout des importances de features si disponibles
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if feature_names is not None:
            results["feature_importances"] = dict(zip(feature_names, importances))
        else:
            results["feature_importances"] = dict(enumerate(importances))
    else:
        results["feature_importances"] = None
    return results
