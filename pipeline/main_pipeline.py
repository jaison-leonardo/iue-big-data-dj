import os
import yaml
import pandas as pd
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from features.clean_data import clean_data
from features.build_features import build_features

from pipeline.classification import classification_pipeline
from pipeline.regression import regression_pipeline
from pipeline.clustering import clustering_pipeline

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_path = config["data"]["input_path"]

    # Cargar datos
    df = pd.read_csv(input_path)

    # Limpiar
    df = clean_data(df)

    # Guardar versión limpia
    processed_path = config["data"]["processed_path"]
    df.to_csv(processed_path, index=False)

    # Construir features
    df_feat = build_features(df)
    features_path = config["data"]["features_path"]
    df_feat.to_csv(features_path, index=False)

    # Ejecución de pipelines
    classification_pipeline(df_feat, config)
    regression_pipeline(df_feat, config)
    clustering_pipeline(df_feat, config)


if __name__ == "__main__":
    main()