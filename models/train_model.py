from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.model_selection import GridSearchCV

def train_regression_models(X_train, y_train):
    models = {
        "ridge": Ridge(alpha=1.0),
        "hgbr": HistGradientBoostingRegressor(random_state=42),
        "rf": get_rf_regressor()
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def train_classification_models(X_train, y_train):
    models = {
        "knn": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "hgbc": HistGradientBoostingClassifier(random_state=42),
        "rf": get_rf_classifier()
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def tune_knn(X_train, y_train):
    params = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }
    grid = GridSearchCV(
        KNeighborsClassifier(),
        params,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def get_rf_classifier():
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

def get_rf_regressor():
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )