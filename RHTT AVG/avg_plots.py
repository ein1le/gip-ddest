import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


RHTT_L_DRY_eng = pd.read_csv(r"RHTT AVG\RHTT_L_DRY_Ex_avg_eng_stress_strain.csv")
RHTT_L_DRY_true = pd.read_csv(r"RHTT AVG\RHTT_L_DRY_Ex_avg_true_stress_strain.csv")
RHTT_L_LUB_eng = pd.read_csv(r"RHTT AVG\RHTT_L_LUB_Ex_avg_eng_stress_strain.csv")
RHTT_L_LUB_true = pd.read_csv(r"RHTT AVG\RHTT_L_LUB_Ex_avg_true_stress_strain.csv")
RHTT_S_DRY_eng = pd.read_csv(r"RHTT AVG\RHTT_S_DRY_Ex_avg_eng_stress_strain.csv")
RHTT_S_DRY_true = pd.read_csv(r"RHTT AVG\RHTT_S_DRY_Ex_avg_true_stress_strain.csv")
RHTT_S_LUB_eng = pd.read_csv(r"RHTT AVG\RHTT_S_LUB_Ex_avg_eng_stress_strain.csv")
RHTT_S_LUB_true = pd.read_csv(r"RHTT AVG\RHTT_S_LUB_Ex_avg_true_stress_strain.csv")


fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].plot(RHTT_L_DRY_eng["Engineering Strain"], RHTT_L_DRY_eng["Engineering Stress (MPa)"], label="Small Clearance, High Friction")
axes[0].plot(RHTT_L_LUB_eng["Engineering Strain"], RHTT_L_LUB_eng["Engineering Stress (MPa)"], label="Small Clearance, Low Friction")
axes[0].plot(RHTT_S_DRY_eng["Engineering Strain"], RHTT_S_DRY_eng["Engineering Stress (MPa)"], label="Large Clearance, High Friction")
axes[0].plot(RHTT_S_LUB_eng["Engineering Strain"], RHTT_S_LUB_eng["Engineering Stress (MPa)"], label="Large Clearance, Low Friction")
#axes[0].fill_between(truncated_common_strain, truncated_lower_bound, truncated_upper_bound,alpha=0.3, label="± 1 Std Dev")
axes[1].plot(RHTT_L_DRY_true["True Strain"], RHTT_L_DRY_true["True Stress (MPa)"], label="Small Clearance, High Friction")
axes[1].plot(RHTT_L_LUB_true["True Strain"], RHTT_L_LUB_true["True Stress (MPa)"], label="Small Clearance, Low Friction")
axes[1].plot(RHTT_S_DRY_true["True Strain"], RHTT_S_DRY_true["True Stress (MPa)"],  label="Large Clearance, High Friction")
axes[1].plot(RHTT_S_LUB_true["True Strain"], RHTT_S_LUB_true["True Stress (MPa)"], label="Large Clearance, Low Friction")

axes[0].set_xlabel("Average Engineering Strain")
axes[0].set_ylabel("Average Engineering Stress (MPa)")
axes[0].set_title(f"Average Engineering Stress-Strain")
axes[0].legend()
axes[0].grid(True)

axes[1].set_xlabel("Average True Strain")
axes[1].set_ylabel("Average True Stress (MPa)")
axes[1].set_title("Average True Stress-Strain")
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig(f"ALL_AVG_stress_strain_curves.png")
plt.show()