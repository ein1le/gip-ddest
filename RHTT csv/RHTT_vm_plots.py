import pandas as pd
import matplotlib.pyplot as plt

# Define test type options
test_types = ["RHTT_L_DRY", "RHTT_S_DRY", "RHTT_L_LUB", "RHTT_S_LUB"]

# Prompt user to manually enter a test type
while True:
    test_type = input(f"Enter the test type ({', '.join(test_types)}): ").strip()
    if test_type in test_types:
        break
    print("Invalid test type. Please enter one of the specified options.")

# Define columns for the plot
strain = "Strain - RC (Extensometer 1)"  # Strain column
vm_stress_g1 = "Von Mises - RC (Strain Gauge 1)"  # Von Mises Stress from Strain Gauge 1
vm_stress_g2 = "Von Mises - RC (Strain Gauge 2)"  # Von Mises Stress from Strain Gauge 2

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
        if strain in df.columns and vm_stress_g1 in df.columns and vm_stress_g2 in df.columns:
            # Drop NaN values
            df = df[[strain, vm_stress_g1, vm_stress_g2]].dropna()

            # Convert data to numeric
            df[strain] = pd.to_numeric(df[strain], errors="coerce")
            df[vm_stress_g1] = pd.to_numeric(df[vm_stress_g1], errors="coerce")
            df[vm_stress_g2] = pd.to_numeric(df[vm_stress_g2], errors="coerce")

            # Drop rows where conversion failed
            df = df.dropna()

            # Plot only if there is valid data left
            if not df.empty:
                axes[0].plot(df[strain], df[vm_stress_g1], label=f"Test {i}")  # Left plot
                axes[1].plot(df[strain], df[vm_stress_g2], label=f"Test {i}")  # Right plot
            else:
                print(f"No valid data in {file_path}. Skipping.")

        else:
            print(f"Columns not found in {file_path}. Available columns: {df.columns.tolist()}")

    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Formatting the left plot (Strain vs. Von Mises Stress - Gauge 1)
axes[0].set_xlabel(strain)
axes[0].set_ylabel(vm_stress_g1)
axes[0].set_title(f"Von Mises Stress (Gauge 1) - {test_type}")
axes[0].legend()
axes[0].grid(True)

# Formatting the right plot (Strain vs. Von Mises Stress - Gauge 2)
axes[1].set_xlabel(strain)
axes[1].set_ylabel(vm_stress_g2)
axes[1].set_title(f"Von Mises Stress (Gauge 2) - {test_type}")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
