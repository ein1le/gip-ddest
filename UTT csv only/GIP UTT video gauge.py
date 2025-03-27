#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import t
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "Times New Roman"

# Fundamental constants
e = np.e
π = np.pi

# Number of experiments 
n = 5

data = []

for i in range(2, n + 2):  # Loop from 2 to n, to include all tests
    csv_name2 = fr"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\UTT csv only\uniaxial_test_{i}.csv"
    csv2 = pd.read_csv(csv_name2, header=None, skiprows=5)
    data.append(csv2)

# Load data F (N)
F = []
for j in range(0, n):  # Change range to process all n tests
    F_data = data[j].iloc[75:, 0].values * 1000
    F.append(F_data - F_data[0])

# Displacement data dL (mm)
ε = []

for k in range(0, n):  # Change range to process all n tests
    # Read and replace 'invalid' values with NaN, then replace NaN with 0
    ε_data = pd.to_numeric(data[k].iloc[75:, 3], errors='coerce').fillna(0).values
    ε.append(ε_data - ε_data[0])
    
'''
UNIAXIAL TENSION TEST GRAPHS
----------------------------
'''

# Initial cross-sectional area (mm^2)
A_0 = 0.006 * 0.002

# Initialize lists to store all resampled stress-strain data
all_ε_t = []
all_σ_t = []

UTS_list = []
E_list = []
yield_list = []

# Loop through all experiments to compute intersection points
for i in range(n):  # Loop through all n experiments
    # Engineering Strain and Engineering Stress for the current experiment
    ε_e = ε[i] / 100
    σ_e = F[i] / A_0

    # True Strain and True Stress for the current experiment
    ε_t = np.log(1 + ε_e)
    σ_t = σ_e * (1 + ε_e)

    # Add the resampled data to the lists
    all_ε_t.append(ε_t[:-170])
    all_σ_t.append(σ_t[:-170])

    # Calculate True Young's modulus (gradient) for the current experiment
    σ_t_initial = σ_t[100:300] / 1e6  # In MPa
    ε_t_initial = ε_t[100:300]
    
    # Calculate Ultimate Tensile Strength
    UTS = max(σ_e)
    UTS_list.append(UTS)
    
    # Calculate True Young's modulus
    E_t = (σ_t_initial[-1] - σ_t_initial[0]) / (ε_t_initial[-1] - ε_t_initial[0])  
    E_list.append(E_t)

    # Calculate yield stress
    offset_line = E_t * (ε_t - 0.002)
    diff = σ_t - offset_line
    
    sign_change_indices = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change_indices) == 0:
        print(f"Test {i+1}: No intersection found.")
        yield_list.append(np.nan)
        continue
    
    idx = sign_change_indices[0]
    x1, x2 = ε_t[idx], ε_t[idx+1]
    y1, y2 = diff[idx], diff[idx+1]
    yield_strain = x1 - y1 * (x2 - x1) / (y2 - y1)
    yield_stress = np.interp(yield_strain, ε_t, σ_t)
    yield_list.append(yield_stress)

# Plotting only True Stress-Strain curves
plt.figure(figsize=(16, 8))

for i in range(n):  # Loop through all n experiments
    plt.plot(all_ε_t[i], all_σ_t[i] / 1e6, label=f"Test {i+1}")

plt.xlabel('True Strain')
plt.ylabel('True Stress (MPa)')
plt.title('True Stress-Strain Curves - INSTRON')
plt.ylim(0, 400)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("True Stress-Strain Curves.png")
plt.show()

# Convert lists to numpy arrays for easier processing (if not already)
all_ε_t_arr = [np.array(arr) for arr in all_ε_t]
all_σ_t_arr = [np.array(arr) for arr in all_σ_t]
# Define the number of points for the common strain grid
num_points = 1000

# Step 1: Define the union of strain ranges
union_strain_min = min(np.min(arr) for arr in all_ε_t_arr)
union_strain_max = max(np.max(arr) for arr in all_ε_t_arr)
common_strain = np.linspace(union_strain_min, union_strain_max, num_points)

# Step 2: Interpolate stress data from each experiment onto the common strain grid
stress_interp = []
for strain, stress in zip(all_ε_t_arr, all_σ_t_arr):
    f_interp = interp1d(strain, stress, bounds_error=False, fill_value=np.nan)
    stress_interp.append(f_interp(common_strain))
stress_interp = np.array(stress_interp)

# Step 3: Filter points based on data availability
# For example, only include grid points where at least half the experiments have valid data.
n_experiments = stress_interp.shape[0]
valid_counts = np.sum(~np.isnan(stress_interp), axis=0)
min_valid = n_experiments // 2  # you can adjust this threshold as needed
mask = valid_counts >= min_valid

# Apply the mask to get the final grid and corresponding stress values
common_strain_filtered = common_strain[mask]
mean_stress = np.nanmean(stress_interp, axis=0)[mask]
std_stress = np.nanstd(stress_interp, axis=0)[mask]
se_stress = std_stress / np.sqrt(valid_counts[mask])

# For a 95% confidence interval, get the t-value for degrees of freedom = n_valid - 1 at each point
# Here, for simplicity, we assume n_experiments (if most points have full data)
from scipy.stats import t
df = n_experiments - 1
t_val = t.ppf(1 - 0.025, df)
ci = t_val * se_stress

avg_max_strain = np.mean([np.max(arr) for arr in all_ε_t_arr])
print(f"Average maximum strain across experiments: {avg_max_strain}")

# Filter the common strain grid further based on the average maximum strain
mask2 = common_strain_filtered <= avg_max_strain
common_strain_final = common_strain_filtered[mask2]
mean_stress_final = mean_stress[mask2]
ci_final = ci[mask2]

# Plot the truncated average stress curve with 95% CI
plt.figure(figsize=(16, 8))
plt.plot(common_strain_final, mean_stress_final / 1e6, color='blue', label='Mean True Stress')
plt.fill_between(common_strain_final,
                 (mean_stress_final - ci_final) / 1e6,
                 (mean_stress_final + ci_final) / 1e6,
                 color='blue', alpha=0.3, label='95% CI')
plt.xlabel('True Strain')
plt.ylabel('True Stress (MPa)')
plt.title('Average True Stress-Strain Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Average True Stress-Strain Curve.png")
plt.show()

df_points_avg = pd.DataFrame({
"strain": common_strain_final,
"stress (MPa)": mean_stress_final / 1e6
})
df_points_avg.to_csv("Average_UTT.csv", index=False)

def calculate_yield_strength_and_points(common_strain_final, mean_stress_final, n):
    
    # 1. Estimate the elastic modulus (E) by fitting the initial linear region.
    # Here, we use the first 10 data points (or fewer if the dataset is small).
    num_points_for_fit = min(10, len(common_strain_final))
    p = np.polyfit(common_strain_final[:num_points_for_fit], mean_stress_final[:num_points_for_fit], 1)
    E = p[0]  # Slope of the line is our elastic modulus
    
    # 2. Compute the 0.2% offset line:
    offset_line = E * (common_strain_final - 0.002)
    
    # 3. Find the intersection between the stress-strain curve and the offset line.
    # Calculate the difference between the actual stress and the offset line.
    diff = mean_stress_final - offset_line
    
    # Identify indices where the difference changes sign (indicating an intersection).
    sign_change_indices = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_change_indices) == 0:
        print("No intersection found between the stress-strain curve and the offset line.")
        return
    else:
        # We'll take the first sign change as our yield point.
        i = sign_change_indices[0]
        # Linear interpolation between the points (i) and (i+1)
        x1, x2 = common_strain_final[i], common_strain_final[i+1]
        y1, y2 = diff[i], diff[i+1]
        yield_strain = x1 - y1 * (x2 - x1) / (y2 - y1)
        # Interpolate to get the yield strength from the actual stress data
        yield_stress = np.interp(yield_strain, common_strain_final, mean_stress_final)
    
    # 4. Determine the maximum strain value.
    max_strain = np.max(common_strain_final)
    
    # 5. Generate n equally spaced points between yield strain and max strain.
    n_points = np.linspace(yield_strain, max_strain, n)
    # For each of these strain values, interpolate to get the corresponding stress values.
    n_stress_points = np.interp(n_points, common_strain_final, mean_stress_final)
    
    # 6. Plot the stress-strain curve, the yield point, the n equally spaced points,
    #    and mark the maximum strain point with a distinct marker.
    plt.figure(figsize=(8,6))
    plt.plot(common_strain_final, mean_stress_final, label="Stress-Strain Curve")
    plt.scatter(n_points, n_stress_points, color='green', label=f"{n} Equally Spaced Points")
    # Mark maximum strain point specially (e.g., using a star marker)
    max_stress = np.interp(max_strain, common_strain_final, mean_stress_final)
    plt.scatter([max_strain], [max_stress], color='blue', label="Maximum Strain Point")
    plt.scatter([yield_strain], [yield_stress], color='red', label="Yield Point")

    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.title("Points on Average Stress-Strain Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Yield Strain:", yield_strain)
    print("Yield Strength:", yield_stress)
    print("Maximum Strain:", max_strain)
    for idx, (strain_val, stress_val) in enumerate(zip(n_points, n_stress_points), 1):
        print(f"Point {idx}: Strain = {strain_val}, Stress = {stress_val* 1e-6} MPa")

    df_points = pd.DataFrame({
    "idx": np.arange(1, n + 1),
    "strain": n_points,
    "stress (MPa)": n_stress_points / 1e6
    })
    df_points.to_csv("UTT_yield_to_UTS_points.csv", index=False)

calculate_yield_strength_and_points(common_strain_final, mean_stress_final, n=30)


yield_idx = np.argmax(common_strain_final > 0.002)
avg_sigma_0 = mean_stress_final[yield_idx]  # Approximate yield strength

def ramberg_osgood_model(sigma, alpha, n):
    return (sigma / 70.7065e9) + alpha * (sigma / avg_sigma_0) ** n

popt, _ = curve_fit(ramberg_osgood_model, mean_stress_final, common_strain_final, p0=[0.002, 5])
alpha_fit, n_fit = popt

print(f"Fitted alpha: {alpha_fit:.6f}")
print(f"Fitted n: {n_fit:.4f}")

sigma_extrap = np.linspace(0, np.max(mean_stress_final) * 1.2, 500)  # Extend a bit beyond
strain_extrap = ramberg_osgood_model(sigma_extrap, alpha_fit, n_fit)

stress_from_strain = interp1d(strain_extrap, sigma_extrap, kind='linear', fill_value="extrapolate")
stress_at_15_strain = stress_from_strain(0.15)

print(f"Predicted stress at 15% strain: {stress_at_15_strain/1e6:.2f} MPa")

plt.figure(figsize=(8, 5))
plt.plot(common_strain_final, mean_stress_final / 1e6, label='Original Data')
plt.plot(strain_extrap, sigma_extrap / 1e6, '-', label='Ramberg-Osgood Extrapolation')
plt.axvline(0.15, color='r', linestyle='--', label='15% Strain')
plt.axhline(stress_at_15_strain / 1e6, color='g', linestyle='--', label=f'{stress_at_15_strain/1e6:.1f} MPa')
plt.xlim(0,0.20)
plt.xlabel('Strain')
plt.ylabel('Stress (MPa)')
plt.title('Ramberg-Osgood Extrapolation to 15% Strain (Aluminium)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def generate_extrapolated_points(n, common_strain, stress_function, final_strain=0.15):
    """
    Prints and saves n points between the last strain in common_strain and final_strain.
    Each row includes the index, strain value, and corresponding stress (in Pa and MPa).
    """
    import pandas as pd

    # 1. Identify the last recorded strain
    last_measured_strain = common_strain[-1]

    # 2. Generate n equally spaced strain values
    strain_array = np.linspace(last_measured_strain, final_strain, n)

    # 3. Compute corresponding stress values
    stress_array = stress_function(strain_array)  # stress in Pa

    # 4. Print results
    print(f"Generating {n} equally spaced points from strain={last_measured_strain:.4f} to {final_strain:.4f}:\n")
    for i, (strain_val, stress_val) in enumerate(zip(strain_array, stress_array), start=1):
        print(f"Point {i}: Strain = {strain_val:.4f}, Stress = {stress_val / 1e6:.2f} MPa")

    # 5. Save to CSV
    df_extrapolated = pd.DataFrame({
        "idx": np.arange(1, n + 1),
        "strain": strain_array,
        "stress (MPa)": stress_array / 1e6
    })

    df_extrapolated.to_csv("UTT_UTS_to_15strain.csv", index=False)


generate_extrapolated_points(30, common_strain_final, stress_from_strain, final_strain=0.15)


'''
STATISTICAL ANALYSIS
--------------------
'''

# Mean of the Ultimate Tensile Strength
UTS_mean = np.mean(UTS_list)
print(f"\nUTS_mean is {UTS_mean / 1e6} MPa")

# Median of the Ultimate Tensile Strength
UTS_median = np.median(UTS_list)
print(f'UTS_median = {UTS_median / 1e6} MPa')

# Standard deviation of the Ultimate Tensile Strength
UTS_std = np.std(UTS_list)
print(f'UTS_std = {UTS_std / 1e6} MPa')

# Mean of the Young's Modulus
E_mean = np.mean(E_list)
print(f"\nE_mean is {E_mean / 1e3} GPa")

# Median of the Young's Modulus
E_median = np.median(E_list)
print(f'E_median = {E_median / 1e3} GPa')

# Standard deviation of the Young's Modulus
E_std = np.std(E_list)
print(f'E_std = {E_std / 1e3} GPa')

# Mean of the Yield Strength
yield_mean = np.mean(yield_list)
print(f"\nYS_mean is {yield_mean / 1e6} MPa")

# Median of the Yield Strength
yield_median = np.median(yield_list)
print(f'YS_median = {yield_median / 1e6} MPa')

# Standard deviation of the Yield Strength
yield_std = np.std(yield_list)
print(f'YS_std = {yield_std / 1e6} MPa')

print(yield_list)