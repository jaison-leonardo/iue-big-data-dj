import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Asegurar que la variable objetivo exista y no tenga vacíos
    df = df.dropna(subset=["class", "u", "g", "r", "i", "z", "redshift"])

    # Filtrado básico de calidad
    if "snr_r" in df.columns:
        df = df[df["snr_r"] > 0]
        
    df = df[df["redshift"] >= 0]

    # Eliminar valores extremos poco físicos (opcional pero recomendado)
    df = df[(df["redshift"] >= 0) & (df["redshift"] < 10)]

    # Filtrar magnitudes astronomicamente atípicas (evitar ruido instrumental)
    for col in ["u", "g", "r", "i", "z"]:
        # Típicamente los valores útiles de SDSS están en el rango 10-30
        df = df[(df[col] > 10) & (df[col] < 30)]

    return df