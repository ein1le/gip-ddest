import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"

df_DRY = pd.read_csv(r"LFT friction avg\DRY_ratio_array.csv")
df_SG = pd.read_csv(r"LFT friction avg\SG_ratio_array.csv")
df_PTFE = pd.read_csv(r"LFT friction avg\PTFE_ratio_array.csv")
df_2L = pd.read_csv(r"LFT friction avg\TwoL_ratio_array.csv")
df_2LR = pd.read_csv(r"LFT friction avg\TwoLR_ratio_array.csv")

test_type = {
    "DRY": df_DRY,
    "SG": df_SG,
    "PTFE": df_PTFE,
    "2L": df_2L,
    "2LR": df_2LR
}


for test, df in test_type.items():
    plt.figure(figsize=(18, 6))
    for columns in df.columns:
        plt.plot(df[columns], label=columns)
    plt.show()