import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

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

# Modify the if clause to define both strain columns
if strain_method == "Ex":
    strain_col_1 = "Strain - RC (Extensometer 1)"  # Primary strain column
    strain_col_2 = "Dummy Extensometer"            # Dummy column will be set to zero
elif strain_method == "SG":
    strain_col_1 = "Displacement (Strain Gauge 1)"
    strain_col_2 = "Displacement (Strain Gauge 2)"
elif strain_method == "ESG":
    strain_col_1 = "Displacement (Strain Gauge 1)"
    strain_col_2 = "Displacement (Strain Gauge 2)"

# Define columns for the plot
load_col = "Load ADC 1 channel 1"
alt_load_col = "Voltage ADC 1 channel 1"

# Initialize empty lists to store the strain and stress data
strain_data = []
stress_data = []

Ultimate_Tensile_Strength = {}
Ductility = {}

# Loop through files and process data
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

        # Prepare the required columns
        required_cols = [strain_col_1, strain_col_2, load_col]

        # Check if these columns exist
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"Skipping {file_path} - Missing columns: {missing_cols}")
            continue
        
        # Subset the dataframe to required columns
        df = df[required_cols]

        # Convert them all to numeric at once
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")

        # Drop any rows that failed conversion
        df.dropna(subset=required_cols, inplace=True)

        # If there's no data left, skip
        if df.empty:
            print(f"No valid data after numeric conversion in {file_path}. Skipping.")
            continue

        # Now perform the engineering calculations

        # Engineering Strain
        if strain_method == "Ex":
            # Extensometer method: sum of the two columns (second is zero)
            df["Engineering Strain"] = df[strain_col_1] + df[strain_col_2]
        elif strain_method in ["SG", "ESG"]:
            # Strain gauge method: sum of displacements / initial length
            df["Engineering Strain"] = (df[strain_col_1] + df[strain_col_2]) / initial_length

        # Engineering Stress (Force / Area)
        df["Engineering Stress"] = (df[load_col] * 1e3 / 2) / A

        # True Strain: ε_true = ln(1 + ε_eng)
        df["True Strain"] = np.log(1 + df["Engineering Strain"])

        # True Stress: σ_true = σ_eng * (1 + ε_eng)
        df["True Stress"] = df["Engineering Stress"] * (1 + df["Engineering Strain"])

        stress_threshold = 10
        search_start_index = 400
        
        # Make sure we have at least search_start_index rows before proceeding
        if len(df) > search_start_index:
            valid_indices = df.loc[search_start_index:, "Engineering Stress"] \
                            .where(df.loc[search_start_index:, "Engineering Stress"] >= stress_threshold) \
                            .dropna().index

            if not valid_indices.empty:
                last_valid_index = valid_indices[-1]
                print(f"Truncating {file_path} at index {last_valid_index}.")
                df = df.loc[:last_valid_index]
            else:
                print(f"No valid truncation index found for {file_path}. Keeping full dataset.")
        else:
            print(f"Dataset too small to apply truncation for {file_path}. Keeping full dataset.")

        # Final check after truncation
        if df.empty:
            print(f"No data left after truncation for {file_path}. Skipping.")
            continue
      
        # Store paired strain-stress data as numpy arrays
        strain_data.append(df["Engineering Strain"].values)
        stress_data.append(df["Engineering Stress"].values)

        # Collect UTS and ductility
        max_stress = df["Engineering Stress"].max()
        Ultimate_Tensile_Strength[f"Test {i}"] = max_stress

        max_strain = df["Engineering Strain"].max()  # engineering strain value
        Ductility[f"Test {i}"] = max_strain * 100  # convert to percentage

    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Define a common strain range for interpolation
# Find the minimum and maximum strain across all tests
all_strains = np.concatenate([s for s in strain_data if len(s) > 0])
min_strain = np.min(all_strains)
max_strain = np.max(all_strains)

# Create a common strain array for interpolation
common_strain = np.linspace(min_strain, max_strain, 1000)

# Interpolate each test's stress values to the common strain values
interpolated_stresses = []
for i, (strain, stress) in enumerate(zip(strain_data, stress_data)):
    if len(strain) == 0 or len(stress) == 0:
        print(f"Skipping empty dataset in interpolation")
        continue
        
    # Remove any NaN values by filtering both arrays together
    mask = ~np.isnan(strain) & ~np.isnan(stress)
    clean_strain = strain[mask]
    clean_stress = stress[mask]
    
    if len(clean_strain) == 0 or len(clean_stress) == 0:
        print(f"No valid data points after NaN removal")
        continue
        
    # Sort strain and stress values (strain should be increasing for interpolation)
    sort_idx = np.argsort(clean_strain)
    clean_strain = clean_strain[sort_idx]
    clean_stress = clean_stress[sort_idx]
    
    try:
        # Create an interpolation function and get the interpolated stress values
        # Use bounds_error=False to handle extrapolation gracefully
        f = interpolate.interp1d(
            clean_strain, 
            clean_stress, 
            kind='linear', 
            bounds_error=False, 
            fill_value=np.nan
        )
        interpolated_stress = f(common_strain)
        interpolated_stresses.append(interpolated_stress)
        print(f"Successfully interpolated dataset {i+1}")
    except Exception as e:
        print(f"Error interpolating dataset {i+1}: {e}")

# Calculate the average stress and uncertainty at each strain point, ignoring NaN values
if interpolated_stresses:
    # Stack all interpolated stresses into a 2D array
    all_interpolated_stresses = np.vstack(interpolated_stresses)
    
    # Calculate mean along axis 0 (across all tests), ignoring NaN values
    avg_interpolated_stress = np.nanmean(all_interpolated_stresses, axis=0)
    
    # Calculate standard deviation for uncertainty, ignoring NaN values
    std_interpolated_stress = np.nanstd(all_interpolated_stresses, axis=0)
    
    # Calculate upper and lower bounds for the uncertainty shading (mean ± standard deviation)
    upper_bound = avg_interpolated_stress + std_interpolated_stress
    lower_bound = avg_interpolated_stress - std_interpolated_stress
    
    # Calculate true stress and strain from engineering values
    true_strain = np.log(1 + common_strain)
    true_stress = avg_interpolated_stress * (1 + common_strain)
    
    # Calculate uncertainty bounds for true stress
    true_upper_bound = upper_bound * (1 + common_strain)
    true_lower_bound = lower_bound * (1 + common_strain)
    
    # ====== Truncate the stress and strain graph at the average ductility ======
    # Calculate the average ductility (in %), then convert it to an engineering strain cutoff (ductility in decimal)
    if Ductility:
        avg_ductility = round(sum(Ductility.values()) / len(Ductility), 2)  # in %
        cutoff_strain = avg_ductility / 100  # convert percentage to engineering strain
    else:
        cutoff_strain = max_strain  # fallback if no ductility data

    # Create a mask to truncate arrays where engineering strain exceeds the average ductility cutoff
    mask_trunc = common_strain <= cutoff_strain

    # Truncate common strain and related arrays using the mask
    truncated_common_strain = common_strain[mask_trunc]
    truncated_avg_interpolated_stress = avg_interpolated_stress[mask_trunc]
    truncated_lower_bound = lower_bound[mask_trunc]
    truncated_upper_bound = upper_bound[mask_trunc]

    # For true stress-strain, compute corresponding true strain and stress
    truncated_true_strain = np.log(1 + truncated_common_strain)
    truncated_true_stress = truncated_avg_interpolated_stress * (1 + truncated_common_strain)
    truncated_true_lower_bound = truncated_lower_bound * (1 + truncated_common_strain)
    truncated_true_upper_bound = truncated_upper_bound * (1 + truncated_common_strain)
    
    # Plotting the results (truncated)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Engineering Stress-Strain with uncertainty (truncated)
    axes[0].plot(truncated_common_strain, truncated_avg_interpolated_stress, label=f"Avg Engineering Stress-Strain - {test_type}", color='blue')
    axes[0].fill_between(truncated_common_strain, truncated_lower_bound, truncated_upper_bound, alpha=0.3, color='blue', label="± 1 Std Dev")
    
    # Plot individual stress-strain curves (truncated)
    for i, interp_stress in enumerate(interpolated_stresses):
        interp_stress_trunc = interp_stress[mask_trunc]
        axes[0].plot(truncated_common_strain, interp_stress_trunc, alpha=0.3, linewidth=0.8, color='gray', label=f"Test {i+1}" if i == 0 else "")
    
    axes[0].set_xlabel("Engineering Strain")
    axes[0].set_ylabel("Engineering Stress (MPa)")
    axes[0].set_title(f"Engineering Stress-Strain Curve - {test_type}\n(Truncated at Avg Ductility: {cutoff_strain:.4f})")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot True Stress-Strain with uncertainty (truncated)
    axes[1].plot(truncated_true_strain, truncated_true_stress, label=f"Avg True Stress-Strain - {test_type}", color='red')
    axes[1].fill_between(truncated_true_strain, truncated_true_lower_bound, truncated_true_upper_bound, alpha=0.3, color='red', label="± 1 Std Dev")
    
    # Plot individual true stress-strain curves (truncated)
    for i, interp_stress in enumerate(interpolated_stresses):
        interp_stress_trunc = interp_stress[mask_trunc]
        true_stress_i_trunc = interp_stress_trunc * (1 + truncated_common_strain)
        axes[1].plot(truncated_true_strain, true_stress_i_trunc, alpha=0.3, linewidth=0.8, color='gray', label=f"Test {i+1}" if i == 0 else "")
    
    axes[1].set_xlabel("True Strain")
    axes[1].set_ylabel("True Stress (MPa)")
    axes[1].set_title(f"True Stress-Strain Curve - {test_type}\n(Truncated at Avg Ductility)")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

    df_true = pd.DataFrame({
        "True Strain": truncated_true_strain,
        "True Stress (MPa)": truncated_true_stress,
        "Lower Bound (MPa)": truncated_true_lower_bound,
        "Upper Bound (MPa)": truncated_true_upper_bound
    })

    csv_filename = f"{test_type}_{strain_method}_true_stress_strain.csv"
    df_true.to_csv(csv_filename, index=False)
    print(f"Truncated True Stress-Strain data saved to {csv_filename}")
    
    
else:
    print("No valid data sets to interpolate and average.")

# Print out summary
if Ultimate_Tensile_Strength:
    avg_UTS = round(sum(Ultimate_Tensile_Strength.values()) / len(Ultimate_Tensile_Strength), 2)
    std_UTS = round(np.std(list(Ultimate_Tensile_Strength.values())), 2)
    cv_UTS = round((std_UTS / avg_UTS) * 100, 2) if avg_UTS != 0 else "N/A"
    
    print(f"Average UTS for {test_type}: {avg_UTS} ± {std_UTS} MPa (CV: {cv_UTS}%)")
    for test_id, stress in Ultimate_Tensile_Strength.items():
        print(f"{test_id}: {round(stress, 2)} MPa")
else:
    print("No UTS data available.")

if Ductility:
    avg_ductility = round(sum(Ductility.values()) / len(Ductility), 2)
    std_ductility = round(np.std(list(Ductility.values())), 2)
    cv_ductility = round((std_ductility / avg_ductility) * 100, 2) if avg_ductility != 0 else "N/A"
    
    print(f"\nAverage Ductility for {test_type}: {avg_ductility} ± {std_ductility}% (CV: {cv_ductility}%)")
    for test_id, strain in Ductility.items():
        print(f"{test_id}: {round(strain, 2)}%")
else:
    print("No ductility data available.")
