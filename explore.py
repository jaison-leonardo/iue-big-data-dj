import pandas as pd

df = pd.read_csv("sdss_sample_pro.csv")

print(df.head())
print(df.columns)
print(df.info())