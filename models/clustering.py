from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

def run_kmeans(X, y_true=None, n_clusters=3):
    # PCA para reducir ruido y correlacion dimensional
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_pca)
    
    results = {
        "silhouette": silhouette_score(X_pca, clusters, random_state=42)
    }
    if y_true is not None:
        results["ari"] = adjusted_rand_score(y_true, clusters)
        
    return clusters, model, results