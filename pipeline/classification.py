import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.train_model import tune_knn, train_classification_models
from evaluation.metrics import classification_metrics_full
from evaluation.plots import plot_confusion_matrix

def classification_pipeline(df, config):
    # Selección de variables para clasificación
    features_enhanced = config["classification"]["features"]
    X = df[features_enhanced]
    y = df[config["classification"]["target"]]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["split"]["test_size"], 
        random_state=config["split"]["random_state"], 
        stratify=y
    )

    # Escalado + KNN
    # KNN necesita escalado para rendir mejor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # KNN optimizado
    knn_model, best_params = tune_knn(X_train_scaled, y_train)
    print(f"Mejores hiperparámetros KNN: {best_params}")

    models = train_classification_models(X_train_scaled, y_train)
    
    # Reemplazamos el modelo KNN por defecto de train_classification_models 
    # por el que acabamos de optimizar
    models["knn"] = knn_model

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
