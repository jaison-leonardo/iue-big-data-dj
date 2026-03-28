import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.train_model import train_regression_models
from evaluation.metrics import regression_metrics
from evaluation.plots import plot_regression

def regression_pipeline(df, config):
    # =========================
    # REGRESIÓN
    # =========================

    print("\nIniciando regresión...")

    reg_features = config["regression"]["features"]
    X_reg = df[reg_features]
    y_reg = df[config["regression"]["target"]]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg,
        y_reg,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"]
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
