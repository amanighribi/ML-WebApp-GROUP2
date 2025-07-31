import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score, recall_score, precision_score, roc_auc_score

def train_model(model_name, X_train, y_train, params):
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            random_state=42
        )
    elif model_name == "SVM":
        model = SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "linear"),
            probability=True,
            random_state=42
        )
    elif model_name == "XGBoost":
        model = XGBClassifier(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    elif model_name == "KNN":
        model = KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform")
        )
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=params.get("max_depth", 5),
            criterion=params.get("criterion", "gini"),
            random_state=42
        )
    elif model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            C=params.get("C", 1.0),
            penalty=params.get("penalty", "l2"),
            solver=params.get("solver", "liblinear"),
            random_state=42
        )
    elif model_name == "XGBoost Regressor":
        model = XGBRegressor(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 100),
            random_state=42
        )
    elif model_name == "KMeans":
        model = KMeans(
            n_clusters=params.get("n_clusters", 3),
            init=params.get("init", "k-means++"),
            max_iter=params.get("max_iter", 300),
            random_state=42
        )
    elif model_name == "DBSCAN":
        model = DBSCAN(
            eps=params.get("eps", 0.5),
            min_samples=params.get("min_samples", 5)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name in ["KMeans", "DBSCAN"]:
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test=None):
    results = {}

    # Clustering case (no y_test)
    if hasattr(model, "labels_") or model.__class__.__name__ in ["KMeans", "DBSCAN"]:
        # For clustering models, we need to fit and predict in one step or use labels_
        if model.__class__.__name__ == "DBSCAN":
            # DBSCAN doesn't have predict method, use fit_predict
            labels = model.fit_predict(X_test)
        else:
            # For KMeans, we can use predict
            labels = model.predict(X_test)
        results["labels"] = labels
        results["accuracy"] = None
        results["f1_score"] = None
        results["confusion_matrix"] = None
        results["y_proba"] = None
        results["recall"] = None
        results["precision"] = None
        results["auc_roc"] = None
        return results

    # Linear Regression case
    if model.__class__.__name__ == "LinearRegression":
        y_pred = model.predict(X_test)
        results["mse"] = mean_squared_error(y_test, y_pred)
        results["r2"] = r2_score(y_test, y_pred)
        results["y_pred"] = y_pred
        results["recall"] = None
        results["precision"] = None
        results["auc_roc"] = None
        return results

    # XGBoost Regressor case
    if model.__class__.__name__ == "XGBRegressor":
        y_pred = model.predict(X_test)
        results["mse"] = mean_squared_error(y_test, y_pred)
        results["r2"] = r2_score(y_test, y_pred)
        results["y_pred"] = y_pred
        results["recall"] = None
        results["precision"] = None
        results["auc_roc"] = None
        return results

    # Classification case
    y_pred = model.predict(X_test)
    results["accuracy"] = accuracy_score(y_test, y_pred)
    results["f1_score"] = f1_score(y_test, y_pred, average="weighted")
    results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
    results["y_pred"] = y_pred
    results["recall"] = recall_score(y_test, y_pred, average="weighted")
    results["precision"] = precision_score(y_test, y_pred, average="weighted")

    # Gestion améliorée des probabilités pour la courbe ROC
    auc_roc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            # Vérifier si c'est une classification binaire via model.classes_
            if len(model.classes_) == 2:
                pos_class_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
                results["y_proba"] = y_proba[:, pos_class_index]
                try:
                    auc_roc = roc_auc_score(y_test, results["y_proba"])
                except Exception:
                    auc_roc = None
            else:
                results["y_proba"] = None
        except Exception:
            results["y_proba"] = None
    else:
        results["y_proba"] = None
    results["auc_roc"] = auc_roc

    return results