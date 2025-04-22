import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
font_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\computer-modern\cmunrm.ttf"
fm.fontManager.addfont(font_path)
cm_font = fm.FontProperties(fname=font_path)
font_name = cm_font.get_name()
plt.rcParams['font.family'] = font_name

RHTT_L_DRY_eng = pd.read_csv(r"RHTT AVG\RHTT_L_DRY_Ex_avg_eng_stress_strain.csv")
RHTT_L_DRY_true = pd.read_csv(r"RHTT AVG\RHTT_L_DRY_Ex_avg_true_stress_strain.csv")
RHTT_L_LUB_eng = pd.read_csv(r"RHTT AVG\RHTT_L_LUB_Ex_avg_eng_stress_strain.csv")
RHTT_L_LUB_true = pd.read_csv(r"RHTT AVG\RHTT_L_LUB_Ex_avg_true_stress_strain.csv")
RHTT_S_DRY_eng = pd.read_csv(r"RHTT AVG\RHTT_S_DRY_Ex_avg_eng_stress_strain.csv")
RHTT_S_DRY_true = pd.read_csv(r"RHTT AVG\RHTT_S_DRY_Ex_avg_true_stress_strain.csv")
RHTT_S_LUB_eng = pd.read_csv(r"RHTT AVG\RHTT_S_LUB_Ex_avg_eng_stress_strain.csv")
RHTT_S_LUB_true = pd.read_csv(r"RHTT AVG\RHTT_S_LUB_Ex_avg_true_stress_strain.csv")
UTT_eng = pd.read_csv(r"RHTT AVG\Average_UTT_Engineering.csv")

# -------------------------------
# 🔹 FIRST PLOT: Engineering Stress-Strain
# -------------------------------
fig1, ax1 = plt.subplots(figsize=(15, 10))

ax1.plot(RHTT_L_DRY_eng["Engineering Strain"], RHTT_L_DRY_eng["Engineering Stress (MPa)"], label="Small Clearance, High Friction",linewidth=2)
ax1.plot(RHTT_L_LUB_eng["Engineering Strain"], RHTT_L_LUB_eng["Engineering Stress (MPa)"], label="Small Clearance, Low Friction",linewidth=2)
ax1.plot(RHTT_S_DRY_eng["Engineering Strain"], RHTT_S_DRY_eng["Engineering Stress (MPa)"], label="Large Clearance, High Friction",linewidth=2)
ax1.plot(RHTT_S_LUB_eng["Engineering Strain"], RHTT_S_LUB_eng["Engineering Stress (MPa)"], label="Large Clearance, Low Friction",linewidth=2)
ax1.plot(UTT_eng["strain"], UTT_eng["stress (MPa)"], label="Uniaxial Tension Test")
# ax1.fill_between(truncated_common_strain, truncated_lower_bound, truncated_upper_bound, alpha=0.3, label="± 1 Std Dev")

ax1.set_xlabel("Average Engineering Strain",fontsize = 28)
ax1.set_ylabel("Average Engineering Stress (MPa)",fontsize = 28)
ax1.tick_params(axis='both', labelsize=22)
#ax1.set_title("Average Engineering Stress-Strain")
ax1.legend(fontsize = 24)
ax1.grid(True)

plt.tight_layout()
plt.savefig("ENG_stress_strain.png", dpi=300)
plt.show()


# -------------------------------
# 🔹 SECOND PLOT: Friction Adjusted Engineering Stress-Strain
# -------------------------------
fig2, ax2 = plt.subplots(figsize=(15, 10))

mu_dry = 0.2595
mu_lub = 0.0471

adjust_dry = np.exp(-mu_dry * 30 * np.pi / 180)
adjust_lub = np.exp(-mu_lub * 30 * np.pi / 180)

ax2.plot(RHTT_L_DRY_eng["Engineering Strain"], RHTT_L_DRY_eng["Engineering Stress (MPa)"] * adjust_dry, label="Small Clearance, High Friction",linewidth=2)
ax2.plot(RHTT_L_LUB_eng["Engineering Strain"], RHTT_L_LUB_eng["Engineering Stress (MPa)"] * adjust_lub, label="Small Clearance, Low Friction",linewidth=2)
ax2.plot(RHTT_S_DRY_eng["Engineering Strain"], RHTT_S_DRY_eng["Engineering Stress (MPa)"] * adjust_dry, label="Large Clearance, High Friction",linewidth=2)
ax2.plot(RHTT_S_LUB_eng["Engineering Strain"], RHTT_S_LUB_eng["Engineering Stress (MPa)"] * adjust_lub, label="Large Clearance, Low Friction",linewidth=2)
ax2.plot(UTT_eng["strain"], UTT_eng["stress (MPa)"], label="Uniaxial Tension Test")

ax2.set_xlabel("Average Engineering Strain",fontsize = 28)
ax2.tick_params(axis='both', labelsize=22)
ax2.set_ylabel("Capstan Adjusted Stress (MPa)",fontsize = 28)
#ax2.set_title("Friction Adjusted Average Engineering Stress-Strain")
ax2.legend(fontsize = 24)
ax2.grid(True)

plt.tight_layout()
plt.savefig("FRICTION_ADJUSTED_stress_strain.png", dpi=300)
plt.show()