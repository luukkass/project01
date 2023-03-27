#imports and randomstate
import pandas as pd

df = pd.read_csv("speeddating.csv")

mean = df.mean(axis=0)
df2 = df.fillna(mean)