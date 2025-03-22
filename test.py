import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

# Define test type options
test_types = ["RHTT_L_DRY", "RHTT_S_DRY", "RHTT_L_LUB", "RHTT_S_LUB"]

# Prompt user to manually enter a test type
while True:
    test_type = input(f"Enter the test type ({', '.join(test_types)}): ").strip()
    if test_type in test_types:
        break
    print("Invalid test type. Please enter one of the specified options.")

A = 12  # Cross-sectional area in mm² (6 mm x 2 mm)
initial_length = 25.75  # Initial position of the specimen in mm

# Define which columns to use for displacement/strain
while True:
    strain_method = input("Enter the strain method | Extensometer (Ex), Strain Gauges (SG), Euclidean Strain Gauge (ESG): ").strip()
    if strain_method in ["Ex", "SG", "ESG"]:
        break
    print("Invalid strain method")

# For ESG, we will define these gauge columns
gauge_1_cols = [
    "X-displacement (Strain Gauge 1)",
    "Y-displacement (Strain Gauge 1)",
    "Z-displacement (Strain Gauge 1)"
]
gauge_2_cols = [
    "X-displacement (Strain Gauge 2)",
    "Y-displacement (Strain Gauge 2)",
    "Z-displacement (Strain Gauge 2)"
]

# Modify the if clause to define primary vs. secondary strain columns
if strain_method == "Ex":
    strain_col_1 = "Strain - RC (Extensometer 1)"  # Primary strain column
    strain_col_2 = "Dummy Extensometer"            # Dummy column will be set to zero
elif strain_method == "SG":
    strain_col_1 = "Displacement (Strain Gauge 1)"
    strain_col_2 = "Displacement (Strain Gauge 2)"
elif strain_method == "ESG":
    # We won't rely on these directly for the final strain calculation,
    # but let's still define them for checking columns exist & numeric conversions.
    strain_col_1 = "Displacement (Strain Gauge 1)"
    strain_col_2 = "Displacement (Strain Gauge 2)"

# Define columns for the plot
load_col = "Load ADC 1 channel 1"
alt_load_col = "Voltage ADC 1 channel 1"

# Initialize subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

Ultimate_Tensile_Strength = {}
Ductility = {}

# Loop through files and plot data
for i in range(1, 6):
    file_path = f"C:/Users/USER/Desktop/Uni Files/Y4/gip-ddest/RHTT csv/{test_type}_{i}.csv"

    try:
        # Read CSV (semicolon delimiter)
        df = pd.read_csv(file_path, encoding='latin1', sep=";")

        # Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()

        # Handle missing load column by substituting the voltage column if needed
        if load_col not in df.columns:
            if alt_load_col in df.columns:
                print(f"Using '{alt_load_col}' instead of '{load_col}' for {file_path} (converted to load).")
                df[alt_load_col] = pd.to_numeric(df[alt_load_col], errors="coerce")
                df[load_col] = df[alt_load_col] * 2.5  # Convert voltage to load
            else:
                print(f"Skipping {file_path} - Required load columns not found.")
                continue  # Skip to the next file

        # If using an Extensometer, create the second strain column with zeros
        if strain_method == "Ex":
            df[strain_col_2] = 0.0

        # If ESG, we need all 6 displacement columns from gauge_1_cols + gauge_2_cols
        # but for initial numeric checks, let's define the minimal set of columns required
        if strain_method == "ESG":
            required_cols = gauge_1_cols + gauge_2_cols + [load_col]
        else:
            # SG or Ex just rely on the two strain columns + load
            required_cols = [strain_col_1, strain_col_2, load_col]

        # Check if these columns exist
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"Skipping {file_path} - Missing columns: {missing_cols}")
            continue
        
        # Subset the DataFrame to required columns
        df = df[required_cols]

        # Convert them all to numeric at once
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")

        # Drop any rows that failed conversion
        df.dropna(subset=required_cols, inplace=True)

        # If there's no data left, skip
        if df.empty:
            print(f"No valid data after numeric conversion in {file_path}. Skipping.")
            continue

        # --- Now perform the strain calculations depending on the method ---

        if strain_method == "Ex":
            # Extensometer method: sum of the two columns (second is zero)
            df["Engineering Strain"] = df[strain_col_1] + df[strain_col_2]

        elif strain_method == "SG":
            # Strain gauge method: sum of displacements / initial length
            df["Engineering Strain"] = (df[strain_col_1] + df[strain_col_2]) / initial_length

        elif strain_method == "ESG":
            # 1) Subtract initial displacement for each gauge column if first row not NaN
            for col in gauge_1_cols + gauge_2_cols:
                if pd.isna(df[col].iloc[0]):
                    print(f"First row of {col} is NaN, skipping offset subtraction.")
                else:
                    df[col] -= df[col].iloc[0]
            
            # 2) Compute Euclidean distance from origin for each strain gauge
            df["strain_1_xyz_pos"] = np.sqrt((df[gauge_1_cols] ** 2).sum(axis=1))
            df["strain_2_xyz_pos"] = np.sqrt((df[gauge_2_cols] ** 2).sum(axis=1))

            # 3) Sum of the two Euclidean distances, divided by initial length
            df["Engineering Strain"] = (df["strain_1_xyz_pos"] + df["strain_2_xyz_pos"]) / initial_length

        # --- Engineering Stress (Force / Area) ---
        df["Engineering Stress"] = (df[load_col] * 1e3 / 2) / A

        # --- True Strain: ε_true = ln(1 + ε_eng) ---
        df["True Strain"] = np.log(1 + df["Engineering Strain"])

        # --- True Stress: σ_true = σ_eng * (1 + ε_eng) ---
        df["True Stress"] = df["Engineering Stress"] * (1 + df["Engineering Strain"])

        # --- Apply stress threshold to truncate if needed ---
        stress_threshold = 10
        search_start_index = 400
        valid_indices = df.loc[search_start_index:, "Engineering Stress"] \
                          .where(df.loc[search_start_index:, "Engineering Stress"] >= stress_threshold) \
                          .dropna().index

        if not valid_indices.empty:
            last_valid_index = valid_indices[-1]
            print(f"Truncating {file_path} at index {last_valid_index}.")
            df = df.loc[:last_valid_index]
        else:
            print(f"No valid truncation index found for {file_path}. Keeping full dataset.")

        # Final check after truncation
        if df.empty:
            print(f"No data left after truncation for {file_path}. Skipping.")
            continue

        # Optionally save True Stress & Strain if needed
        if test_type == "RHTT_L_DRY" and strain_method == "ESG" and i == 1:
            output_file = f"{test_type}_{i}_{strain_method}_True.csv"
            df[['True Stress', 'True Strain']].to_csv(output_file, index=False)
            print(f"Saved True Stress and Strain of test {i} {test_type} to {output_file}")

        # Plot Engineering Stress-Strain and True Stress-Strain
        axes[0].plot(df["Engineering Strain"], df["Engineering Stress"], label=f"Test {i}")
        axes[1].plot(df["True Strain"], df["True Stress"], label=f"Test {i}")

                # -- NEW: Plot the respective Engineering Strain curve in a separate figure --
        """
        plt.figure(figsize=(8, 4))
        plt.plot(df["Engineering Strain"], color='blue', label="Engineering Strain")
        plt.title(f"Engineering Strain for {test_type} - Test {i}")
        plt.xlabel("Time Step")
        plt.ylabel("Strain")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        
        # Collect UTS and ductility
        max_stress = df["Engineering Stress"].max()
        Ultimate_Tensile_Strength[f"Test {i}"] = max_stress

        max_strain = df["Engineering Strain"].max() * 100  # Convert to percentage
        Ductility[f"Test {i}"] = max_strain

    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Formatting the left plot (Engineering Stress vs. Engineering Strain)
axes[0].set_xlabel("Engineering Strain")
axes[0].set_ylabel("Engineering Stress (MPa)")
axes[0].set_title(f"Engineering Stress-Strain Curve - {test_type}")
axes[0].legend()
axes[0].grid(True)

# Formatting the right plot (True Stress vs. True Strain)
axes[1].set_xlabel("True Strain")
axes[1].set_ylabel("True Stress (MPa)")
axes[1].set_title(f"True Stress-Strain Curve - {test_type}")
axes[1].legend()
axes[1].grid(True)

# Display the final plots
plt.tight_layout()
plt.show()

# Print out summary
avg_UTS = round(sum(Ultimate_Tensile_Strength.values()) / len(Ultimate_Tensile_Strength), 2)
avg_ductility = round(sum(Ductility.values()) / len(Ductility), 2)

print(f"Average UTS for {test_type}: {avg_UTS} MPa")
for test_id, stress in Ultimate_Tensile_Strength.items():
    print(f"{test_id}: {round(stress, 2)} MPa")

print(f"\nAverage Ductility for {test_type}: {avg_ductility}%")
for test_id, strain in Ductility.items():
    print(f"{test_id}: {round(strain, 2)}%")

