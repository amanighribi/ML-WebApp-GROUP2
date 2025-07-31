from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

def is_quasi_constant(series, threshold=0.98):
    return series.value_counts(normalize=True).max() > threshold

def is_high_cardinality(series, threshold=20):
    return series.nunique() > threshold

def is_potential_id(series):
    # Unicité élevée, type object ou int, longueur élevée
    return series.nunique() > 0.95 * len(series) and (series.dtype == 'object' or pd.api.types.is_integer_dtype(series))

def make_recommendation(dtype, uniques, null_rate, util, col, series, df):
    if util == "🚫 Constant (to ignore)":
        return "Colonne constante : à ignorer pour le machine learning."
    if is_quasi_constant(series):
        return "Colonne quasi-constante : à supprimer, n'apporte aucune information utile."
    if util == "⚠️ Too many missing values":
        return "Beaucoup de valeurs manquantes : à imputer ou à ignorer selon l'importance de la variable."
    if util == "🔁 Identifier / index" or is_potential_id(series):
        return "Identifiant ou index (ou variable unique) : à ignorer pour l'apprentissage."
    if dtype == 'object' and is_high_cardinality(series):
        return "Colonne catégorielle à forte cardinalité : regrouper les catégories rares ou utiliser un encodage avancé (target encoding, embeddings, etc.)."
    if dtype == 'object' and uniques <= 20:
        return "Bonne variable catégorielle pour la classification. Pensez à l'encoder (OneHot, Ordinal, etc.)."
    if pd.api.types.is_numeric_dtype(series):
        if uniques > 20:
            if abs(series.max() - series.min()) > 1e2:
                return "Variable numérique à grande amplitude : à normaliser ou standardiser pour certains modèles (SVM, KNN, etc.)."
            else:
                return "Bonne variable numérique pour la régression."
        else:
            return "Variable numérique discrète : peut être utilisée en classification."
    if null_rate > 0:
        return "Attention : valeurs manquantes à traiter (imputation recommandée)."
    return "Colonne exploitable pour le machine learning."

@app.route('/analyze-dataset', methods=['POST'])
def analyze_dataset():
    file = request.files['file']
    df = pd.read_csv(file)
    summary = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        uniques = int(series.nunique())
        null_rate = float(series.isnull().mean())
        examples = series.dropna().unique()[:3]
        example_vals = ", ".join(map(str, examples)) + ("..." if uniques > 3 else "")
        if dtype == 'object':
            description = f"Categorical ({uniques} categories)"
        elif pd.api.types.is_numeric_dtype(series):
            description = f"Numerical ({series.min()} to {series.max()})"
        elif 'date' in col.lower():
            description = "Temporal (date format)"
        else:
            description = f"Type {dtype}"
        if uniques == 1:
            util = "🚫 Constant (to ignore)"
        elif null_rate > 0.5:
            util = "⚠️ Too many missing values"
        elif col.lower() in ['id', 'index'] or series.is_monotonic_increasing or series.is_monotonic_decreasing:
            util = "🔁 Identifier / index"
        else:
            util = "✅ Potentially useful"
        recommendation = make_recommendation(dtype, uniques, null_rate, util, col, series, df)
        summary.append({
            "column": col,
            "type": dtype,
            "uniques": uniques,
            "examples": example_vals,
            "description": description,
            "ml_utility": util,
            "recommendation": recommendation
        })
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(port=5001, debug=True) 