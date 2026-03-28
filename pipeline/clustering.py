import json
import joblib
from sklearn.preprocessing import StandardScaler
from models.clustering import run_kmeans
from evaluation.plots import plot_clustering

def clustering_pipeline(df, config):
    # =========================
    # CLUSTERING
    # =========================

    print("\nIniciando clustering...")

    cluster_features = config["clustering"]["features"]
    X_cluster = df[cluster_features].values

    # Escalado (crítico para KMeans)
    scaler_c = StandardScaler()
    X_cluster_scaled = scaler_c.fit_transform(X_cluster)

    clusters, kmeans_model, cluster_metrics = run_kmeans(
        X_cluster_scaled,
        df["class"],
        n_clusters=config["clustering"]["n_clusters"]
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
