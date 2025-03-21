import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/USER/Desktop/Uni Files/Y4/gip-ddest/RHTT csv/RHTT_L_DRY_1.csv", encoding='latin1', sep=";")

print(df["Displacement (Strain Gauge 1)"].tail(20))
print(df["Displacement (Strain Gauge 2)"].tail(20))
