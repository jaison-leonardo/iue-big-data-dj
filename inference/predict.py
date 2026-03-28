import joblib
import pandas as pd
import numpy as np
import os

def load_model(model_path):
    """
    Carga un modelo guardado desde la ruta especificada.
    
    Args:
        model_path (str): Ruta al archivo .pkl del modelo.
        
    Returns:
        El modelo entrenado cargado en memoria.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo no se encontró en {model_path}")
    return joblib.load(model_path)

def predict_classification(model, data):
    """
    Realiza predicciones de clasificación.
    Asegúrate de que los datos de entrada ('data') hayan sido preprocesados
    y escalados de la misma manera que en el entrenamiento.
    
    Args:
        model: Modelo de clasificación cargado.
        data (pd.DataFrame o np.ndarray): Datos de características.
        
    Returns:
        np.ndarray: Predicciones de clase.
    """
    return model.predict(data)

def predict_regression(model, data):
    """
    Realiza predicciones de regresión.
    Asegúrate de que los datos de entrada ('data') hayan sido preprocesados
    y escalados de la misma manera que en el entrenamiento.
    
    Args:
        model: Modelo de regresión cargado.
        data (pd.DataFrame o np.ndarray): Datos de características.
        
    Returns:
        np.ndarray: Predicciones continuas.
    """
    return model.predict(data)

def predict_clustering(model, data):
    """
    Realiza predicciones de clustering (asignación de cluster a nuevos datos).
    Asegúrate de que los datos de entrada ('data') hayan sido preprocesados
    y escalados de la misma manera que en el entrenamiento.
    
    Args:
        model: Modelo de clustering cargado (ej. KMeans).
        data (pd.DataFrame o np.ndarray): Datos de características.
        
    Returns:
        np.ndarray: Etiquetas de cluster asignadas.
    """
    return model.predict(data)

if __name__ == "__main__":
    # Ejemplo de uso:
    try:
        # Cargar un modelo de clasificación
        clf_model_path = "models/trained/hgbc_enhanced.pkl"
        if os.path.exists(clf_model_path):
            classifier = load_model(clf_model_path)
            print("Modelo de clasificación cargado exitosamente.")
            
            # Datos simulados (deberían estar escalados si el modelo lo requiere)
            sample_data = np.random.rand(1, 10) # Asumiendo 10 features
            
            prediccion = predict_classification(classifier, sample_data)
            print(f"Predicción de clasificación: {prediccion}")
    except Exception as e:
        print(f"Error en ejemplo: {e}")
