import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="ML Studio", page_icon="🤖", layout="wide")


def read_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if file.name.lower().endswith(".json"):
        return pd.read_json(file)
    raise ValueError("Format non pris en charge")


def build_pipeline(X, model):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num))
    if cat:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("model", model)])


def get_models(problem):
    if problem == "Classification":
        return {
            "Régression logistique": LogisticRegression(max_iter=2000),
            "K plus proches voisins": KNeighborsClassifier(),
            "Arbre de décision": DecisionTreeClassifier(random_state=42),
            "Forêt aléatoire": RandomForestClassifier(random_state=42),
            "SVM": SVC(),
        }
    return {
        "Régression linéaire": LinearRegression(),
        "Arbre de décision": DecisionTreeRegressor(random_state=42),
        "Forêt aléatoire": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
    }


def score_model(problem, pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    if problem == "Classification":
        return {"Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, average="weighted", zero_division=0), "Recall": recall_score(y_test, pred, average="weighted", zero_division=0), "F1": f1_score(y_test, pred, average="weighted", zero_division=0)}, pred
    return {"R²": r2_score(y_test, pred), "MAE": mean_absolute_error(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** 0.5}, pred


def nav_buttons():
    col1, col2 = st.columns([85, 15]) # Give more space to the left

    # Back Button (Bottom Left)
    with col1:
        if st.session_state.get("page_index", 0) > 0:
            st.button("⬅️ Précédent", key="prev_btn", on_click=lambda: set_page(st.session_state.page_index - 1))

    # Next Button (Bottom Right)
    with col2:
        if st.session_state.get("page_index", 0) < len(pages) - 1:
            st.button("Suivant ➡️", key="next_btn", on_click=lambda: set_page(st.session_state.page_index + 1))

def set_page(idx):
    st.session_state.page_index = idx
    st.rerun()


if "data" not in st.session_state:
    st.session_state.data = None
if "results" not in st.session_state:
    st.session_state.results = None
if "page_index" not in st.session_state:
    st.session_state.page_index = 0

pages = ["1. Importer", "2. Dashboard", "3. Prétraitement", "4. Modèles", "5. Test Manuel"]

st.title("🤖 ML Studio")
st.caption("Une application simple pour explorer vos données et entraîner des modèles de Machine Learning.")

with st.sidebar:
    st.header("Navigation")
    # Map the current page_index to the radio button
    selected_page = st.radio("Choisir une étape", pages, index=st.session_state.page_index)
    st.session_state.page_index = pages.index(selected_page)

    if st.session_state.data is not None:
        st.success(f"Dataset chargé : {st.session_state.data.shape[0]} lignes × {st.session_state.data.shape[1]} colonnes")

page = selected_page

if page == "1. Importer":
    st.header("Importer un dataset")
    file = st.file_uploader("Fichier CSV, Excel ou JSON", type=["csv", "xlsx", "xls", "json"])
    if file:
        try:
            st.session_state.data = read_file(file)
            st.session_state.results = None
            st.success("Dataset importé avec succès.")
        except Exception as e:
            st.error(f"Impossible de lire le fichier : {e}")
    if st.session_state.data is not None:
        st.dataframe(st.session_state.data.head(20), use_container_width=True)

    st.markdown("---")
    nav_buttons()

elif st.session_state.data is None:
    st.info("Importez d'abord un dataset dans la section « 1. Importer ».")

elif page == "2. Dashboard":
    df = st.session_state.data
    st.header("Dashboard du dataset")

    st.subheader("Analyse des valeurs manquantes")
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ['Colonne', 'Valeurs Manquantes']
    missing_df = missing_df[missing_df['Valeurs Manquantes'] > 0]
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("Aucune valeur manquante détectée !")

    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

elif page == "3. Prétraitement":
    st.header("Prétraitement des données")
    df = st.session_state.data.copy()
    missing = st.selectbox("Valeurs manquantes", [
        "Ne rien faire",
        "Supprimer les lignes incomplètes",
        "Médiane / mode (Remplacement automatique lors de l'entraînement)"
    ])
    duplicates = st.checkbox("Supprimer les doublons")
    columns = st.multiselect("Colonnes à supprimer", df.columns.tolist())
    if st.button("Appliquer le prétraitement", type="primary"):
        if missing == "Supprimer les lignes incomplètes": df = df.dropna()
        if duplicates: df = df.drop_duplicates()
        if columns: df = df.drop(columns=columns)
        st.session_state.data = df.reset_index(drop=True)
        st.session_state.results = None
        st.success("Prétraitement appliqué.")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")
    nav_buttons()
    st.download_button("Télécharger le dataset nettoyé", df.to_csv(index=False).encode("utf-8"), "dataset_nettoye.csv", "text/csv")

elif page == "4. Modèles":
    st.header("Entraîner et comparer les modèles")
    df = st.session_state.data
    target = st.selectbox("Variable cible", df.columns.tolist())
    inferred = "Classification" if df[target].dtype == "object" or df[target].nunique() <= 15 else "Régression"
    problem = st.radio("Type de problème", ["Classification", "Régression"], index=0 if inferred == "Classification" else 1, horizontal=True)
    choices = st.multiselect("Algorithmes", list(get_models(problem).keys()), default=list(get_models(problem).keys()))
    test_size = st.slider("Part réservée au test", 0.1, 0.4, 0.2, 0.05)
    if st.button("Entraîner les modèles", type="primary"):
        X, y = df.drop(columns=target), df[target]
        if X.empty or y.isna().all(): st.error("Le dataset ou la cible n'est pas valide.")
        else:
            stratify = y if problem == "Classification" and y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
            records, fitted = [], {}
            for name in choices:
                try:
                    pipe = build_pipeline(X, get_models(problem)[name]); pipe.fit(X_train, y_train)
                    metrics, pred = score_model(problem, pipe, X_test, y_test)
                    records.append({"Modèle": name, **metrics}); fitted[name] = (pipe, pred, y_test)
                except Exception as e: st.warning(f"{name} ignoré : {e}")
            st.session_state.results = {"table": pd.DataFrame(records), "fitted": fitted, "problem": problem, "X": X, "y": y}
    r = st.session_state.results
    if r is not None:
        table = r["table"].sort_values("Accuracy" if r["problem"] == "Classification" else "R²", ascending=False)

        # --- NEW: Color Coding for Metrics ---
        def color_metrics(val):
            try:
                val_float = float(val)
                if r["problem"] == "Classification":
                    # Accuracy, Precision, Recall, F1
                    color = 'color: green' if val_float >= 0.7 else 'color: red' if val_float < 0.5 else 'color: orange'
                else:
                    # R2, MAE, RMSE (R2 is good high, others good low)
                    # This is a simplification; usually we compare to baseline
                    color = 'color: green' if val_float >= 0.7 else 'color: red' if val_float < 0.4 else 'color: orange'
                return color
            except:
                return ''

        st.dataframe(table.style.map(color_metrics, subset=[c for c in table.columns if c != "Modèle"]).format(precision=3), use_container_width=True)

        # --- NEW: Legend for colors ---
        st.markdown("""
        **Légende des performances :**
        - :color[green] **Vert** : Très bonne performance ($\ge 0.7$)
        - :color[orange] **Orange** : Performance moyenne ($0.5$ à $0.7$)
        - :color[red] **Rouge** : Performance faible ($< 0.5$)
        """)

        metric = "Accuracy" if r["problem"] == "Classification" else "R²"
        st.plotly_chart(px.bar(table, x="Modèle", y=metric, color=metric, title="Comparaison des performances"), use_container_width=True)
        best = table.iloc[0]["Modèle"]
        pipe, pred, y_test = r["fitted"][best]
        st.success(f"Meilleur modèle : {best}")
        if r["problem"] == "Classification":
            st.subheader("Matrice de confusion")
            labels = sorted(pd.Series(y_test).unique())
            st.plotly_chart(px.imshow(confusion_matrix(y_test, pred, labels=labels), x=labels, y=labels, text_auto=True, labels={"x":"Prédit", "y":"Réel"}), use_container_width=True)
            st.text(classification_report(y_test, pred, zero_division=0))

            # --- NEW: ROC-AUC Curve ---
            st.subheader("Courbe ROC")
            try:
                if len(labels) == 2: # ROC is primarily for binary classification
                    pipe_prob = r["fitted"][best][0]
                    y_prob = pipe_prob.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    auc = roc_auc_score(y_test, y_prob)
                    fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={auc:.3f})",
                                     labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
                    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("La courbe ROC est disponible uniquement pour la classification binaire.")
            except Exception as e:
                st.warning(f"Impossible de générer la courbe ROC : {e}")

        # --- NEW: Feature Importance ---
        st.subheader("Importance des variables")
        try:
            model_obj = r["fitted"][best][0].named_steps['model']
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                feat_names = r["X"].columns
                feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=False)
                st.plotly_chart(px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Importance des caractéristiques"), use_container_width=True)
            elif hasattr(model_obj, 'coef_'):
                coefs = np.abs(model_obj.coef_[0])
                feat_names = r["X"].columns
                feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': coefs}).sort_values('Importance', ascending=False)
                st.plotly_chart(px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Poids des coefficients (Valeur Absolue)"), use_container_width=True)
            else:
                st.info("L'algorithme choisi ne supporte pas l'affichage de l'importance des variables.")
        except Exception as e:
            st.warning(f"Erreur lors du calcul de l'importance : {e}")

        buffer = io.BytesIO(); joblib.dump(pipe, buffer)
        st.download_button("Télécharger le meilleur modèle", buffer.getvalue(), "meilleur_modele.joblib", "application/octet-stream")

    st.markdown("---")
    nav_buttons()

elif page == "5. Test Manuel":
    st.header("🧪 Test Manuel")
    if st.session_state.results is None:
        st.info("Veuillez d'abord entraîner un modèle dans la section « 4. Modèles ».")
    else:
        r = st.session_state.results
        best_model_name = r["table"].sort_values("Accuracy" if r["problem"] == "Classification" else "R²", ascending=False).iloc[0]["Modèle"]
        pipe = r["fitted"][best_model_name][0]

        st.success(f"Test avec le meilleur modèle : {best_model_name}")

        st.subheader("Saisissez les valeurs des caractéristiques")

        # Create inputs for each feature
        X_columns = r["X"].columns
        user_inputs = {}

        # Organise inputs in columns for better UI
        cols = st.columns(2)
        for i, col_name in enumerate(X_columns):
            with cols[i % 2]:
                # Check if column was numeric or categorical in the original data
                if r["X"][col_name].dtype == "object":
                    # For categorical, provide a text input or dropdown of existing values
                    options = r["X"][col_name].unique().tolist()
                    user_inputs[col_name] = st.selectbox(f"{col_name}", options)
                else:
                    # Ensure we only call median on numeric data to avoid TypeError with strings
                    default_val = r["X"][col_name].median() if pd.api.types.is_numeric_dtype(r["X"][col_name]) else 0.0
                    user_inputs[col_name] = st.number_input(f"{col_name}", value=float(default_val))

        if st.button("Prédire", type="primary"):
            # Convert inputs to DataFrame
            input_df = pd.DataFrame([user_inputs])
            prediction = pipe.predict(input_df)[0]

            st.markdown("### Résultat de la prédiction :")
            st.markdown(f"## 🎯 **{prediction}**")

            if r["problem"] == "Classification" and hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(input_df)[0]
                max_prob = np.max(probs)
                st.write(f"Confiance : {max_prob:.2%}")

    st.markdown("---")
    nav_buttons()
