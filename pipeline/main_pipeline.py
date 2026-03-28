import os
import json
import joblib
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.train_model import (
    train_regression_models, tune_knn, get_rf_classifier, 
    get_rf_regressor, train_classification_models
)
from evaluation.metrics import (
    regression_metrics, classification_metrics, classification_metrics_full
)
from evaluation.plots import plot_regression, plot_clustering, plot_confusion_matrix
from models.clustering import run_kmeans
from features.clean_data import clean_data
from features.build_features import build_features

def classification_pipeline(df):
    # Selección de variables para clasificación según la actividad
    features_enhanced = [
    "u", "g", "r", "i", "z", "redshift",
    "u_g", "g_r", "r_i", "i_z"
    ]
    X = df[features_enhanced]
    y = df["class"]

    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Escalado + KNN
    # KNN necesita escalado para rendir mejor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # KNN optimizado
    knn_model, best_params = tune_knn(X_train_scaled, y_train)

    models = train_classification_models(X_train_scaled, y_train)

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)

        # Métricas
        metrics = classification_metrics_full(y_test, y_pred)
        results[name] = metrics

        joblib.dump(model, f"models/trained/{name}_enhanced.pkl")

        plot_confusion_matrix(
            y_test,
            y_pred,
            labels=sorted(y.unique()),
            save_path=f"reports/figures/{name}_enhanced_confusion.png",
            title=f"{name.upper()} Enhanced"
        )

        print(f"\nModelo: {name}")
        print(metrics)

    # Guardar comparación
    with open("reports/metrics/classification_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Clasificación finalizada.")

def regression_pipeline(df):
    # =========================
    # REGRESIÓN
    # =========================

    print("\nIniciando regresión...")

    reg_features = [
        "u", "g", "r", "i", "z",
        "u_g", "g_r", "r_i", "i_z"
    ]
    X_reg = df[reg_features]
    y_reg = df["redshift"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg,
        y_reg,
        test_size=0.3,
        random_state=42
    )

    # Escalado (no obligatorio para regresión lineal, pero consistente)
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)

    reg_models = train_regression_models(X_train_r_scaled, y_train_r)

    results = {}

    for name, model in reg_models.items():
        y_pred_r = model.predict(X_test_r_scaled)

        reg_metrics = regression_metrics(y_test_r, y_pred_r)
        results[name] = reg_metrics

        joblib.dump(model, f"models/trained/{name}_enhanced.pkl")

        plot_regression(
            y_test_r,
            y_pred_r,
            f"reports/figures/{name}_enhanced_regression.png"
        )

        print(f"\nModelo: {name}")
        print(reg_metrics)

    # Guardar comparación
    with open("reports/metrics/regression_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Regresión finalizada.")


def clustering_pipeline(df):
    # =========================
    # CLUSTERING
    # =========================

    print("\nIniciando clustering...")

    # Ampliamos variables para clustering para integrarlas a PCA
    cluster_features = ["u", "g", "r", "i", "z", "u_g", "g_r", "r_i", "i_z"]
    X_cluster = df[cluster_features].values

    # Escalado (crítico para KMeans)
    scaler_c = StandardScaler()
    X_cluster_scaled = scaler_c.fit_transform(X_cluster)

    clusters, kmeans_model, cluster_metrics = run_kmeans(
        X_cluster_scaled,
        df["class"],
        n_clusters=3
    )

    # Guardar modelo
    joblib.dump(kmeans_model, "models/trained/kmeans.pkl")

    # Gráfico 1: clusters
    plot_clustering(
        X_cluster,
        clusters,
        "KMeans Clusters",
        "reports/figures/clusters_kmeans.png"
    )

    # Gráfico 2: clases reales
    # Convertimos class a números para graficar
    class_mapping = {label: idx for idx, label in enumerate(sorted(df["class"].unique()))}
    y_numeric = df["class"].map(class_mapping).values

    plot_clustering(
        X_cluster,
        y_numeric,
        "Real Classes",
        "reports/figures/classes_real.png"
    )

    # Guardar métricas
    with open("reports/metrics/clustering_metrics.json", "w") as f:
        json.dump(cluster_metrics, f, indent=4)

    print(f"Silhouette Score: {cluster_metrics['silhouette']:.4f}")
    print("Clustering finalizado.")

def main():
    input_path = "data/raw/sdss_sample.csv"

    # Cargar datos
    df = pd.read_csv(input_path)

    # Limpiar
    df = clean_data(df)

    # Guardar versión limpia
    df.to_csv("data/processed/processed_data.csv", index=False)

    # Construir features
    df_feat = build_features(df)
    df_feat.to_csv("data/features/features_data.csv", index=False)

    # Ejecución de pipelines
    classification_pipeline(df_feat)
    regression_pipeline(df_feat)
    clustering_pipeline(df_feat)


if __name__ == "__main__":
    main()