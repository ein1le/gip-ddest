import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"


while True:
    strain_method = input("Enter the strain method | Extensometer (Ex) or Strain Gauges (SG): ").strip()
    if strain_method in ["Ex", "SG"]:
        break
    print("Invalid strain method. Please enter 'Ex' for Extensometer or 'SG' for Strain Gauges.")

# Modify the if clause to define both strain columns
if strain_method == "Ex": #starts immediately 
    strain_col_1 = "Strain - RC (Extensometer 1)"  # Primary strain column
    strain_col_2 = "Dummy Extensometer"             # Dummy column that will be set to zero

    # File path for FEA data (remains the same for all tests)
    FEA_file = "C:/Users/USER/Desktop/Uni Files/Y4/gip-ddest/RHTT csv/FEA_L_DRY_postoffset.csv"

    # Read FEA data
    df_fea = pd.read_csv(FEA_file, encoding='latin1', header=None)
    df_fea.columns = ["Strain", "Stress"]
    print(f"Loaded FEA data from: {FEA_file}, {len(df_fea)} rows")


elif strain_method == "SG": #has a lag
    strain_col_1  = "Displacement (Strain Gauge 1)"  # Displacement column for strain gauge 1
    strain_col_2  = "Displacement (Strain Gauge 2)"  # Displacement column for strain gauge 2

    # File path for FEA data (remains the same for all tests)
    FEA_file = "C:/Users/USER/Desktop/Uni Files/Y4/gip-ddest/RHTT csv/FEA_L_DRY_preoffset.csv"

    # Read FEA data
    df_fea = pd.read_csv(FEA_file, encoding='latin1', header=None)
    df_fea.columns = ["Strain", "Stress"]
    print(f"Loaded FEA data from: {FEA_file}, {len(df_fea)} rows")


# Drop NaN values (if any conversion failed)
df_fea = df_fea.dropna()

# Normalize FEA strain if needed
df_fea["Strain"] = pd.to_numeric(df_fea["Strain"], errors="coerce")  # Convert to float, set errors to NaN
df_fea["Stress"] = pd.to_numeric(df_fea["Stress"], errors="coerce")  # Convert stress as well




# Debug print to verify strain range
print("FEA Strain Min/Max:", df_fea["Strain"].min(), df_fea["Strain"].max())

# Cross-sectional area in mm² (6mm x 2mm)
A = 12  
initial_length = 25.75 # Initial position of the specimen in mm

# Define experimental strain and load columns
load_col = "Load ADC 1 channel 1"  # Load column
alt_load_col = "Voltage ADC 1 channel 1"  # Alternative if load_col is missing

# Initialize subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Loop through multiple test files (1 to 5)
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
                raise ValueError(f"Required columns not found in {EXP_file}.")

        if strain_method == "Ex":
             df[strain_col_2] = 0.0

        # Drop NaN values
        df = df[[strain_col_1,strain_col_2, load_col]].dropna()
        print(f"After dropping NaN: {len(df)} rows remaining.")

        # Convert to numeric
        df[strain_col_1] = pd.to_numeric(df[strain_col_1], errors="coerce")
        df[strain_col_2] = pd.to_numeric(df[strain_col_2], errors="coerce")
        df[load_col] = pd.to_numeric(df[load_col], errors="coerce")
        df = df.dropna()  # Drop rows where conversion failed
        print(f"After numeric conversion: {len(df)} rows remaining.")

        # Calculate Engineering Stress and Strain
        if strain_method == "Ex":
            df["Engineering Strain"] = (df[strain_col_1] + df[strain_col_2])
        elif strain_method == "SG":
            df["Engineering Strain"] = (df[strain_col_1] + df[strain_col_2])/ initial_length
        df["Engineering Stress"] = (df[load_col] * 1e3 / 2) / A  # Stress in MPa (N/mm²)

        # Calculate True Stress and Strain
        df["True Strain"] = np.log(1 + df["Engineering Strain"])
        df["True Stress"] = df["Engineering Stress"] * (1 + df["Engineering Strain"])

        # Debug prints for strain range
        print(f"Exp Test {i} - Engineering Strain Min/Max:", df["Engineering Strain"].min(), df["Engineering Strain"].max())
        print(f"Exp Test {i} - True Strain Min/Max:", df["True Strain"].min(), df["True Strain"].max())

        # Plot Engineering Stress-Strain with FEA overlay
        axes[0].plot(df["Engineering Strain"], df["Engineering Stress"], label=f"Exp Test {i}")
        
        # Plot True Stress-Strain with FEA overlay
        axes[1].plot(df["True Strain"], df["True Stress"], label=f"Exp Test {i}")

    except FileNotFoundError:
        print(f"File {EXP_file} not found. Skipping.")
    except Exception as e:
        print(f"Error loading {EXP_file}: {e}")

# Overlay FEA data on both plots
axes[0].plot(df_fea["Strain"], df_fea["Stress"], label="FEA", color='orange', linestyle="--")
axes[1].plot(df_fea["Strain"], df_fea["Stress"], label="FEA", color='orange', linestyle="--")

# Formatting Engineering Stress-Strain Curve
axes[0].set_xlabel("Engineering Strain")
axes[0].set_ylabel("Engineering Stress (MPa)")
axes[0].set_title("Engineering Stress-Strain Curve, RHTT_L_DRY")
axes[0].legend()
axes[0].grid(True)

# Formatting True Stress-Strain Curve
axes[1].set_xlabel("True Strain")
axes[1].set_ylabel("True Stress (MPa)")
axes[1].set_title("True Stress-Strain Curve, RHTT_L_DRY")
axes[1].legend()
axes[1].grid(True)

# Show plots
plt.tight_layout()
plt.show()
