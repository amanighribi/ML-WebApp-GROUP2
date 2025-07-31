import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from scipy.stats import zscore, iqr
import joblib
import os

def clean_data(df):
    # Supprimer les lignes avec valeurs manquantes
    df_clean = df.dropna()
    return df_clean

def encode_categoricals(X, encoders=None):
    """
    Encode categorical columns using category codes.
    """
    X_encoded = X.copy()
    if encoders is None:
        encoders = {}
        for col in X.columns:
            if X[col].dtype in ['object', 'category']:  # Check for both object and category types
                X_encoded[col] = X[col].astype('category').cat.codes
                encoders[col] = dict(enumerate(X[col].astype('category').cat.categories))
    else:
        for col in X.columns:
            if X[col].dtype in ['object', 'category'] and col in encoders:
                X_encoded[col] = X[col].map(encoders[col]).fillna(-1).astype('category').cat.codes
    return X_encoded, encoders

def split_data(df, target_column):
    """
    Split data into training and testing sets, with encoding for categorical features.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if stratification is possible
    stratify = None
    counts = y.value_counts()
    if counts.min() >= 2:
        stratify = y  # Stratification possible

    # Split without encoding to avoid errors
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.33, stratify=stratify, random_state=42)

    # Encode using the existing function, retrieve encoders from training data
    X_train, encoders = encode_categoricals(X_train_raw)

    # For testing data, encode using encoders learned from training data
    X_test, _ = encode_categoricals(X_test_raw, encoders=encoders)

    return X_train, X_test, y_train, y_test

def detect_task_type(y):
    """
    Detect if the task is classification or regression based on the target column.
    Returns 'classification' or 'regression'.
    """
    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique()
        if n_unique <= 20 or (y.dropna().apply(float.is_integer).all() and n_unique < 50):
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'


def select_features(X, y, task_type='auto', method='kbest', k=10, estimator=None):
    """
    Feature selection using SelectKBest or RFE.
    - method: 'kbest' or 'rfe'
    - k: number of features to select
    - estimator: estimator for RFE (default: LogisticRegression or LinearRegression)
    Returns: X_new (DataFrame with selected features), selected feature names
    """
    if task_type == 'auto':
        task_type = detect_task_type(y)
    if method == 'kbest':
        if task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
        return pd.DataFrame(X_new, columns=selected_features, index=X.index), list(selected_features)
    elif method == 'rfe':
        if estimator is None:
            estimator = LogisticRegression(max_iter=1000) if task_type == 'classification' else LinearRegression()
        selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
        return pd.DataFrame(X_new, columns=selected_features, index=X.index), list(selected_features)
    else:
        raise ValueError("Unknown feature selection method: {}".format(method))


def handle_imbalance(X, y, method='smote', random_state=42):
    """
    Handle class imbalance using SMOTE or return class_weight='balanced' for compatible models.
    Returns: X_res, y_res, imbalance_info
    """
    task_type = detect_task_type(y)
    if task_type != 'classification':
        return X, y, {'method': None, 'info': 'Not a classification task'}
    if method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        before = dict(Counter(y))
        after = dict(Counter(y_res))
        return X_res, y_res, {'method': 'smote', 'before': before, 'after': after}
    elif method == 'class_weight':
        # For models that support class_weight='balanced'
        before = dict(Counter(y))
        return X, y, {'method': 'class_weight', 'before': before, 'after': before}
    else:
        return X, y, {'method': None, 'info': 'Unknown method'}


def simple_automl(X_train, y_train, X_test, y_test, task_type='auto', models=None, k_features=10):
    """
    Simple AutoML: test several models and return the best one (by accuracy or R2).
    Returns: best_model_name, best_model, best_score, all_scores
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score, r2_score

    if task_type == 'auto':
        task_type = detect_task_type(y_train)
    if models is None:
        if task_type == 'classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }
        else:
            models = {
                'Linear Regression': LinearRegression(),
            }
    all_scores = {}
    best_score = -np.inf if task_type == 'regression' else 0
    best_model = None
    best_model_name = None
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if task_type == 'classification':
                score = accuracy_score(y_test, y_pred)
            else:
                score = r2_score(y_test, y_pred)
            all_scores[name] = score
            if (task_type == 'classification' and score > best_score) or (task_type == 'regression' and score > best_score):
                best_score = score
                best_model = model
                best_model_name = name
        except Exception as e:
            all_scores[name] = f'Error: {e}'

    # After training and selecting the best model
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(list(X_train.columns), "model_features.pkl")

    return best_model_name, best_model, best_score, all_scores

def impute_missing(df, strategy_num="mean", strategy_cat="most_frequent"):
    """
    Impute missing values: numeric columns with strategy_num, categorical with strategy_cat.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy=strategy_num)
        df[num_cols] = imputer_num.fit_transform(df[num_cols])
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy=strategy_cat)
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    return df

def remove_outliers_zscore(df, threshold=3):
    """
    Remove rows with outliers in numeric columns using z-score method.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df
    z_scores = np.abs(zscore(df[num_cols], nan_policy='omit'))
    mask = (z_scores < threshold).all(axis=1)
    return df[mask]

def winsorize_outliers(df, limits=0.01):
    """
    Winsorize numeric columns (clip extreme values to given quantiles).
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        lower = df[col].quantile(limits)
        upper = df[col].quantile(1 - limits)
        df[col] = df[col].clip(lower, upper)
    return df

def scale_data(df, method="standard"):
    """
    Scale numeric columns using the specified method: 'minmax', 'standard', 'robust'.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    if len(num_cols) > 0:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def encode_categoricals_advanced(df, method="onehot", encoder=None):
    """
    Encode categorical columns using 'onehot', 'ordinal'.
    Returns transformed DataFrame and fitted encoder.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if method == "onehot":
        if encoder is None:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            arr = encoder.fit_transform(df[cat_cols])
        else:
            arr = encoder.transform(df[cat_cols])
        df_onehot = pd.DataFrame(arr, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        df = df.drop(columns=cat_cols).join(df_onehot)
        return df, encoder
    elif method == "ordinal":
        if encoder is None:
            encoder = OrdinalEncoder()
            df[cat_cols] = encoder.fit_transform(df[cat_cols])
        else:
            df[cat_cols] = encoder.transform(df[cat_cols])
        return df, encoder
    else:
        raise ValueError("Unknown encoding method")

def detect_useless_columns(df, threshold_quasi_constant=0.98, exclude_columns=None):
    """
    Detect columns that are quasi-constant, IDs, or have only one unique value.
    Returns a list of column names to drop.
    """
    if exclude_columns is None:
        exclude_columns = []
    cols_to_drop = []
    for col in df.columns:
        if col in exclude_columns:
            continue
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
        elif df[col].dtype in ["object", "category"] and df[col].nunique() / len(df) < (1-threshold_quasi_constant):
            # Quasi-constant categorical
            counts = df[col].value_counts(normalize=True)
            if counts.iloc[0] > threshold_quasi_constant:
                cols_to_drop.append(col)
        elif col.lower() in ["id", "index"] or df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
            cols_to_drop.append(col)
    return cols_to_drop



def advanced_preprocessing_pipeline(df, impute=True, outlier_method=None, scale_method=None, 
                                  encode_method=None, drop_useless=True, encoder=None, 
                                  fit_encoder=True, target_column=None):
    """
    Apply a full advanced preprocessing pipeline.
    """
    # Make a copy of the dataframe
    df_proc = df.copy()
    
    # Store target column if specified
    target_data = None
    if target_column and target_column in df_proc.columns:
        target_data = df_proc[target_column]
        df_proc = df_proc.drop(columns=[target_column])
    
    # Drop useless columns (excluding target)
    if drop_useless:
        cols_to_drop = detect_useless_columns(df_proc)
        df_proc = df_proc.drop(columns=cols_to_drop)
    
    # Apply preprocessing steps
    if impute:
        df_proc = impute_missing(df_proc)
    if outlier_method == "zscore":
        df_proc = remove_outliers_zscore(df_proc)
    elif outlier_method == "winsorize":
        df_proc = winsorize_outliers(df_proc)
    if scale_method:
        df_proc = scale_data(df_proc, method=scale_method)
    if encode_method:
        if fit_encoder or encoder is None:
            df_proc, encoder = encode_categoricals_advanced(df_proc, method=encode_method)
        else:
            df_proc, _ = encode_categoricals_advanced(df_proc, method=encode_method, encoder=encoder)
    
    # Add target column back if it exists
    if target_data is not None:
        df_proc[target_column] = target_data
    
    return df_proc, encoder
