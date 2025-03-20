import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants
A = 12  # Cross-sectional area
strain_col = "Strain - RC (Extensometer 1)"
load_col = "Load ADC 1 channel 1"  # Load column
alt_load_col = "Voltage ADC 1 channel 1"  # Alternative if load_col is missing

# List to store all processed dataframes
all_dataframes = []

# Process each test file
for i in range(1, 6):
    EXP_file = f"C:/Users/USER/Desktop/Uni Files/Y4/gip-ddest/RHTT csv/RHTT_L_DRY_{i}.csv"
    
    try:
        print(f"\nProcessing file: {EXP_file}")

        # Read experimental data
        df = pd.read_csv(EXP_file, encoding='latin1', sep=";")
        df.columns = df.columns.str.strip()  # Clean column names
        print(f"Loaded {len(df)} rows from: {EXP_file}")

        # Handle missing load column by using alternative voltage column
        if load_col not in df.columns:
            if alt_load_col in df.columns:
                df[alt_load_col] = pd.to_numeric(df[alt_load_col], errors="coerce")  # Convert to numeric
                df[load_col] = df[alt_load_col] * 2.5  # Convert voltage to load
                print(f"Using '{alt_load_col}' instead of '{load_col}' for load calculations.")
            else:
                print(f"Skipping {EXP_file}: Required columns not found.")
                continue  # Skip this file if necessary columns are missing

        # Ensure necessary columns exist
        if strain_col not in df.columns or load_col not in df.columns:
            print(f"Skipping {EXP_file}: Required columns not found.")
            continue

        # Convert columns to numeric and drop NaNs
        df[strain_col] = pd.to_numeric(df[strain_col], errors="coerce")
        df[load_col]   = pd.to_numeric(df[load_col], errors="coerce")
        df.dropna(subset=[strain_col, load_col], inplace=True)

        # Debug print to verify strain range
        print("EXP Strain Min/Max:", df[strain_col].min(), df[strain_col].max())

        # Calculate Engineering Stress and Strain
        df["Engineering Strain"] = df[strain_col]
        df["Engineering Stress"] = (df[load_col] * 1e3 / 2) / A  # Convert load to N, then divide by area → MPa

        # Keep only necessary columns
        df = df[["Engineering Strain", "Engineering Stress"]].copy()

        # Sort by strain to ensure ascending order (important for interpolation)
        df.sort_values(by="Engineering Strain", inplace=True)

        # Store the processed dataframe
        all_dataframes.append(df)

    except FileNotFoundError:
        print(f"File not found: {EXP_file}")
    except Exception as e:
        print(f"Error processing {EXP_file}: {e}")

# Ensure we have data before proceeding
if not all_dataframes:
    print("Error: No valid data was loaded. Check file paths and column names.")
    exit()

# 1) Determine the overall maximum strain across *all* datasets.
max_strain = max(df["Engineering Strain"].max() for df in all_dataframes)
n_points = 1000
strain_values = np.linspace(0, max_strain, n_points)

# 2) Interpolate each dataset onto the common strain array (no extrapolation)
interpolated_stresses = []
for df in all_dataframes:
    stress_interp = np.interp(
        strain_values,
        df["Engineering Strain"],
        df["Engineering Stress"],
        left=np.nan,
        right=np.nan
    )
    interpolated_stresses.append(stress_interp)

# Convert list of arrays to a 2D numpy array
interpolated_stresses = np.array(interpolated_stresses)

# 3) Average stress at each strain point, ignoring NaNs
average_stress = np.nanmean(interpolated_stresses, axis=0)

# 4) Mask to filter out any NaNs in the final average
valid_mask = ~np.isnan(average_stress)
valid_strains = strain_values[valid_mask]
valid_stresses = average_stress[valid_mask]

# -----------------------
# 5) 0.2% OFFSET YIELD
# -----------------------
def find_yield_strength(strain, stress, offset=0.002):
    """Return index, strain, and stress at the 0.2% offset yield point."""

    # 5a) Approximate the elastic modulus (E) by linear fit of the first region (e.g., first 5% of data).
    #     Or you could use first 50 points, etc.
    idx_linear = min(50, len(strain))  # if fewer than 50 points, take what we have
    # Fit a line to the initial portion
    slope, intercept = np.polyfit(strain[:idx_linear], stress[:idx_linear], 1)

    # 5b) For each strain[i], define offset_line[i] = slope * (strain[i] - offset)
    #     Then find where stress[i] ~ offset_line[i].
    offset_line = slope * (strain - offset)

    # 5c) Difference between actual stress and offset line
    diff = stress - offset_line

    # 5d) Find the index where diff is closest to 0
    idx_yield = np.argmin(np.abs(diff))

    return idx_yield, strain[idx_yield], stress[idx_yield]

idx_yield, yield_strain, yield_stress = find_yield_strength(valid_strains, valid_stresses, offset=0.002)

print(f"\nYield Strength Calculation (0.2% Offset):")
print(f"  Index at yield:   {idx_yield}")
print(f"  Strain at yield:  {yield_strain:.4f}")
print(f"  Stress at yield:  {yield_stress:.2f} MPa")


max_idx = np.argmax(valid_stresses)
max_strain = valid_strains[max_idx]
max_stress = valid_stresses[max_idx]

print(f"\nMaximum Stress in the Averaged Curve:")
print(f"  Index:  {max_idx}")
print(f"  Strain: {max_strain:.4f}")
print(f"  Stress: {max_stress:.2f} MPa")

def find_points_between(n_points):
    """Finds `n_points` evenly spaced values between idx_yield and max_idx."""
    
    indices = np.linspace(idx_yield, max_idx, n_points, dtype=int)
    selected_strains = valid_strains[indices]
    selected_stresses = valid_stresses[indices]

    print("\nIndex    Strain    Stress")
    print("-" * 30)
    for i, (strain, stress) in zip(indices, zip(selected_strains, selected_stresses)):
        print(f"{i:<8} {strain:.6f} {stress:.2f}")


find_points_between(n_points=30)


# 6) Plot the final average stress-strain curve
plt.figure(figsize=(8, 5))
plt.plot(valid_strains, valid_stresses, label="Average Stress-Strain Curve")

# Mark the yield point
plt.plot(yield_strain, yield_stress, 'o', label=f"Yield Strength ({yield_stress:.2f} MPa)")

plt.plot(max_strain, max_stress, 'o', label=f"Max Stress ({max_stress:.2f} MPa)")


plt.xlabel("Engineering Strain")
plt.ylabel("Engineering Stress (MPa)")
plt.title("Averaged Stress-Strain Curve with Yield Strength (0.2% Offset)")
plt.legend()
plt.grid(True)
plt.show()
