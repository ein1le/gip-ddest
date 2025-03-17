import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define test type options
test_types = ["RHTT_L_DRY", "RHTT_S_DRY", "RHTT_L_LUB", "RHTT_S_LUB"]


# Prompt user to manually enter a test type
while True:
    test_type = input(f"Enter the test type ({', '.join(test_types)}): ").strip()
    if test_type in test_types:
        break
    print("Invalid test type. Please enter one of the specified options.")

A = 12  # Cross-sectional area in mm² (6mm x 2mm)

# Define columns for the plot
strain_col = "Strain - RC (Extensometer 1)"  # Strain column (already in decimal form)
load_col = "Load ADC 1 channel 1"  # Load column
alt_load_col = "Voltage ADC 1 channel 1"  # Alternative column if load_col is missing

# Initialize subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

Ultimate_Tensile_Strength = {}
Ductility = {}  

# Loop through files and plot data
for i in range(1, 6):
    file_path = f"C:/Users/USER/Desktop/Uni Files/Y4/gip-ddest/RHTT csv/{test_type}_{i}.csv"

    try:
        # Read CSV with semicolon delimiter
        df = pd.read_csv(file_path, encoding='latin1', sep=";")

        # Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()

        # Handle missing load column by substituting the voltage column
        if load_col not in df.columns:
            if alt_load_col in df.columns:
                print(f"Using '{alt_load_col}' instead of '{load_col}' for {file_path} (converted to load).")
                df[alt_load_col] = pd.to_numeric(df[alt_load_col], errors="coerce")  # Convert to numeric
                df[load_col] = df[alt_load_col] * 2.5  # Convert voltage to load
            else:
                print(f"Skipping {file_path} - Required columns not found.")
                continue  # Skip to the next file

        # Ensure necessary columns exist
        if strain_col not in df.columns or load_col not in df.columns:
            print(f"Skipping {file_path} - Required columns not found.")
            continue
        
        # Drop NaN values
        df = df[[strain_col, load_col]].dropna()

        # Convert to numeric
        df[strain_col] = pd.to_numeric(df[strain_col], errors="coerce")
        df[load_col] = pd.to_numeric(df[load_col], errors="coerce")

        # Drop rows where conversion failed
        df = df.dropna()

        # Engineering Strain (unchanged since it's already in decimal form)
        df["Engineering Strain"] = df[strain_col]

        # Engineering Stress (Force / Area)
        df["Engineering Stress"] = (df[load_col] * 1e3 / 2) / A  # Stress in MPa (N/mm²)


        # True Strain: ε_true = ln(1 + ε_eng)
        df["True Strain"] = np.log(1 + df["Engineering Strain"])

        # True Stress: σ_true = σ_eng * (1 + ε_eng)
        df["True Stress"] = df["Engineering Stress"] * (1 + df["Engineering Strain"])

        
        stress_threshold = 10  # Define threshold for valid stress values
        search_start_index = 400  # Only check for drops after this index

        # Identify the last index where Engineering Stress remains valid (above threshold)
        valid_indices = df.loc[search_start_index:, "Engineering Stress"].where(df.loc[search_start_index:, "Engineering Stress"] >= stress_threshold).dropna().index

        if not valid_indices.empty:
            last_valid_index = valid_indices[-1]  # Take the last valid index before a major drop
            print(f"Truncating {file_path} at index {last_valid_index}.")
            df = df.loc[:last_valid_index]  # Keep data up to this index
        else:
            print(f"No valid truncation index found. Keeping full dataset for {file_path}.")
        
        if not df.empty:
            axes[0].plot(df["Engineering Strain"], df["Engineering Stress"], label=f"Test {i}")  # Left plot
            axes[1].plot(df["True Strain"], df["True Stress"], label=f"Test {i}")  # Right plot
        else:
            print(f"No valid data in {file_path}. Skipping.")

        
        max_stress = df["Engineering Stress"].max()
        Ultimate_Tensile_Strength[f"Test {i}"] = max_stress  # Store result

        max_strain = df["Engineering Strain"].max() * 100  # Convert to percentage
        Ductility[f"Test {i}"] = max_strain  # Store result
        
    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Formatting the left plot (Engineering Stress vs. Engineering Strain)
axes[0].set_xlabel("Engineering Strain (mm/mm)")
axes[0].set_ylabel("Engineering Stress (MPa)")
axes[0].set_title(f"Engineering Stress-Strain Curve - {test_type}")
axes[0].legend()
axes[0].grid(True)

# Formatting the right plot (True Stress vs. True Strain)
axes[1].set_xlabel("True Strain (mm/mm)")
axes[1].set_ylabel("True Stress (MPa)")
axes[1].set_title(f"True Stress-Strain Curve - {test_type}")
axes[1].legend()
axes[1].grid(True)

# Display the final plots
plt.tight_layout()
plt.show()


print(f"Average UTS for {test_type}: {round(sum(Ultimate_Tensile_Strength.values()) / len(Ultimate_Tensile_Strength), 2)} MPa")
for test, stress in Ultimate_Tensile_Strength.items():
    print(f"{test}: {round(stress, 2)} MPa")

print(f"\nAverage Ductility for {test_type}: {round(sum(Ductility.values()) / len(Ductility), 2)}%")
for test, strain in Ductility.items():
    print(f"{test}: {round(strain, 2)}%")