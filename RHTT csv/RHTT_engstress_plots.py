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

# Prompt user for cross-sectional area A (in mm²)
while True:
    try:
        A = float(input("Enter the cross-sectional area (in mm²): "))
        if A > 0:
            break
        print("Area must be a positive number.")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

# Define columns for the plot
strain_col = "Strain - RC (Extensometer 1)"  # Strain column (already in decimal form)
load_col = "Load ADC 1 channel 1"  # Load column

# Initialize subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Loop through files and plot data
for i in range(1, 6):
    file_path = f"C:/Users/USER/Desktop/Uni Files/Y4/gip-ddest/RHTT csv/{test_type}_{i}.csv"

    try:
        # Read CSV with semicolon delimiter
        df = pd.read_csv(file_path, encoding='latin1', sep=";")

        # Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()

        # Ensure necessary columns exist
        if strain_col in df.columns and load_col in df.columns:
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
            df["Engineering Stress"] = df[load_col] / A  # Stress in MPa (N/mm²)

            # True Strain: ε_true = ln(1 + ε_eng)
            df["True Strain"] = np.log(1 + df["Engineering Strain"])

            # True Stress: σ_true = σ_eng * (1 + ε_eng)
            df["True Stress"] = df["Engineering Stress"] * (1 + df["Engineering Strain"])

            # Plot only if there is valid data left
            if not df.empty:
                axes[0].plot(df["Engineering Strain"], df["Engineering Stress"], label=f"Test {i}")  # Left plot
                axes[1].plot(df["True Strain"], df["True Stress"], label=f"Test {i}")  # Right plot
            else:
                print(f"No valid data in {file_path}. Skipping.")

        else:
            print(f"Columns not found in {file_path}. Available columns: {df.columns.tolist()}")

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
axes[1].set_xlabel("True Strain")
axes[1].set_ylabel("True Stress (MPa)")
axes[1].set_title(f"True Stress-Strain Curve - {test_type}")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
