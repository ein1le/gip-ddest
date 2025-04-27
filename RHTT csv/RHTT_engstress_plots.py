import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate 
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression
import matplotlib as mpl

import matplotlib.font_manager as fm
font_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\computer-modern\cmunrm.ttf"
fm.fontManager.addfont(font_path)
cm_font = fm.FontProperties(fname=font_path)
font_name = cm_font.get_name()
plt.rcParams['font.family'] = font_name


# Define test type options
test_types = ["RHTT_L_DRY", "RHTT_S_DRY", "RHTT_L_LUB", "RHTT_S_LUB"]

# Prompt user for test type
while True:
    test_type = input(f"Enter the test type ({', '.join(test_types)}): ").strip()
    if test_type in test_types:
        break
    print("Invalid test type. Please enter one of the specified options.")

A = 12  # Cross-sectional area (mm² = 6mm x 2mm)
initial_length = 25.75  # Initial position (mm)

# Prompt for strain method
valid_strain_methods = ["Ex", "SG", "ESG"]
while True:
    strain_method = input("Enter strain method (Ex / SG / ESG): ").strip()
    if strain_method in valid_strain_methods:
        break
    print("Invalid strain method.")

# Identify the strain columns
if strain_method == "Ex":
    strain_col_1 = "Strain - RC (Extensometer 1)" 
    strain_col_2 = "Dummy Extensometer"  # Will be 0.0
elif strain_method == "SG":
    strain_col_1 = "Displacement (Strain Gauge 1)"
    strain_col_2 = "Displacement (Strain Gauge 2)"
elif strain_method == "ESG":
    strain_col_1 = "Displacement (Strain Gauge 1)"
    strain_col_2 = "Displacement (Strain Gauge 2)"

# Load columns
load_col = "Load ADC 1 channel 1"
alt_load_col = "Voltage ADC 1 channel 1"

# Dictionary to store processed DataFrames
dfs = {}

# Also track UTS & Ductility
Ultimate_Tensile_Strength = {}
Ductility = {}
Yield_Strength = {}

# Friction adjusted 
adj_Ultimate_Tensile_Strength = {}
adj_Yield_Strength = {}


def create_average_test_five(dfs, tests_to_average, smoothing_window=10):
    """
    If Test 5 does not exist, create an 'average' test by combining
    data from the specified 'tests_to_average'.

    Parameters:
    - dfs: dictionary of test DataFrames
    - tests_to_average: list of test indices to average
    - smoothing_window: int, optional window for rolling average (0 = no smoothing)

    Steps:
      1) Merge (union) their indexes
      2) Take the average of [Engineering Strain, Engineering Stress, True Strain, True Stress]
      3) Optionally smooth the resulting average curves
      4) Determine a cutoff 'average strain' across those tests, and filter rows
      5) Store as dfs[5]
    """

    if 5 in dfs:
        print("Test 5 already exists in dfs. Not creating a new one.")
        return

    if not tests_to_average:
        print("No tests specified to average. Aborting creation of Test 5.")
        return

    dfs_to_combine = []
    for ti in tests_to_average:
        if ti in dfs:
            dfs_to_combine.append(dfs[ti].copy().sort_index())
        else:
            print(f"Warning: Test {ti} is missing in dfs. Skipping it.")

    if not dfs_to_combine:
        print("No valid test DataFrames to combine. Aborting creation of Test 5.")
        return

    from functools import reduce
    union_index = reduce(lambda x, y: x.union(y.index), dfs_to_combine, dfs_to_combine[0].index)

    aligned_dfs = []
    for df_ in dfs_to_combine:
        aligned_dfs.append(df_.reindex(union_index).sort_index())

    columns_to_avg = ["Engineering Strain", "Engineering Stress", "True Strain", "True Stress"]
    df_avg = pd.DataFrame(index=union_index, columns=columns_to_avg, dtype=float)

    for col in columns_to_avg:
        col_frames = [df_[col] for df_ in aligned_dfs]
        col_matrix = pd.concat(col_frames, axis=1)
        df_avg[col] = col_matrix.mean(axis=1, skipna=True)

    # Optional smoothing (rolling mean)
    if smoothing_window > 1:
        print(f"Applying rolling average with window size = {smoothing_window}")
        df_avg = df_avg.rolling(window=smoothing_window, min_periods=1, center=True).mean()

    # Determine strain cutoff
    max_strains = [df_["Engineering Strain"].max() for df_ in dfs_to_combine]
    average_strain_cutoff = np.mean(max_strains)
    print(f"Average strain cutoff from tests {tests_to_average} is ~ {average_strain_cutoff:.4f}")

    # Cut at average strain
    df_avg = df_avg[df_avg["Engineering Strain"] <= average_strain_cutoff].copy()
    df_avg.dropna(subset=columns_to_avg, inplace=True)
    df_avg.sort_index(inplace=True)

    dfs[5] = df_avg
    print(f"Created Test 5 by averaging tests {tests_to_average}, cut at strain {average_strain_cutoff:.4f}.")



# -------------
#  1) LOAD & PROCESS ALL TESTS (1..5)
# -------------
for i in range(1, 6):
    file_path = f"C:/Users/USER/Desktop/Uni Files/Y4/gip-ddest/RHTT csv/{test_type}_{i}.csv"

    try:
        df = pd.read_csv(file_path, encoding='latin1', sep=';')
        df.columns = df.columns.str.strip()

        # Convert voltage => load if needed
        if load_col not in df.columns:
            if alt_load_col in df.columns:
                print(f"Using '{alt_load_col}' instead of '{load_col}' for {file_path}.")
                df[alt_load_col] = pd.to_numeric(df[alt_load_col], errors="coerce")
                df[load_col] = df[alt_load_col] * 2.5
            else:
                print(f"Skipping {file_path}: no load column found.")
                continue

        # Extensometer => second strain col is 0
        if strain_method == "Ex":
            df[strain_col_2] = 0.0

        required_cols = [strain_col_1, strain_col_2, load_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Skipping {file_path}: missing columns {missing}.")
            continue

        # Keep only needed columns
        df = df[required_cols]
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=required_cols, inplace=True)
        if df.empty:
            print(f"Skipping {file_path}: no valid rows after numeric conversion.")
            continue

        # Engineering Strain
        if strain_method == "Ex":
            df["Engineering Strain"] = df[strain_col_1] + df[strain_col_2]
        else:
            df["Engineering Strain"] = (df[strain_col_1] + df[strain_col_2]) / initial_length

        # Engineering Stress (kN->N->MPa), half-bridge => multiply by 1000, then /2
        df["Engineering Stress"] = (df[load_col] * 1e3 / 2) / A

        # True Strain & Stress
        df["True Strain"] = np.log(1 + df["Engineering Strain"])
        df["True Stress"] = df["Engineering Stress"] * (1 + df["Engineering Strain"])

        # Optional Truncation
        stress_threshold = 10
        search_start_index = 400
        if len(df) > search_start_index:
            valid_indices = df.loc[search_start_index:, "Engineering Stress"] \
                              .where(df.loc[search_start_index:, "Engineering Stress"] >= stress_threshold) \
                              .dropna().index
            if not valid_indices.empty:
                last_valid = valid_indices[-1]
                print(f"Truncating {file_path} at index {last_valid}.")
                df = df.loc[:last_valid]
            else:
                print(f"No valid truncation for {file_path}.")
        else:
            print(f"Dataset {file_path} too short to truncate. Keeping full dataset.")

        if df.empty:
            print(f"No data after truncation for {file_path}. Skipping.")
            continue

        # Store processed DataFrame
        dfs[i] = df

    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping.")
    except Exception as e:
        print(f"Error with {file_path}: {e}")




# -------------
#  2) INDEX-BASED SHAPE MATCHING: TEST 4 using TEST 5
# -------------
def shape_match_extrapolate(dfs_dict, truncated_i, reference_i, random_seed=None):
    """
    Index-based shape matching:
      - We'll take the last index from truncated_i
      - Match it to the same index in reference_i
      - Then append subsequent rows from reference_i with offset + random noise
    Modifies dfs_dict[truncated_i] in place to include the new extrapolated portion.
    
    :param dfs_dict: dict of {test_index: DataFrame} with columns
                     [Engineering Strain, Engineering Stress, True Strain, True Stress]
    :param truncated_i: the test index (int) for truncated data
    :param reference_i: the test index (int) for reference shape
    :param random_seed: optional int for reproducible noise
    """
    # Check if both tests exist
    if truncated_i not in dfs_dict or reference_i not in dfs_dict:
        print(f"Cannot extrapolate: Test {truncated_i} or {reference_i} missing in dfs.")
        return

    df_trunc = dfs_dict[truncated_i].copy()
    df_ref   = dfs_dict[reference_i].copy()

    # Sort by index (assuming row-based alignment)
    df_trunc.sort_index(inplace=True)
    df_ref.sort_index(inplace=True)

    # Identify the final index of the truncated test
    end_idx_trunc = df_trunc.index[-1]
    print(f"\nShape-matching: truncated test {truncated_i} ends at index {end_idx_trunc}, using reference test {reference_i}...")

    if end_idx_trunc not in df_ref.index:
        print(f"Index {end_idx_trunc} not found in reference test {reference_i}. No extrapolation done.")
        return

    # Pivot row in reference
    pivot_row_ref = df_ref.loc[end_idx_trunc]
    # Pivot row in truncated
    pivot_row_trunc = df_trunc.loc[end_idx_trunc]

    # Compute offsets
    offset_eng_strain = pivot_row_trunc["Engineering Strain"] - pivot_row_ref["Engineering Strain"]
    offset_eng_stress = pivot_row_trunc["Engineering Stress"] - pivot_row_ref["Engineering Stress"]
    offset_true_strain = pivot_row_trunc["True Strain"] - pivot_row_ref["True Strain"]
    offset_true_stress = pivot_row_trunc["True Stress"] - pivot_row_ref["True Stress"]

    # Grab subsequent rows from reference test -> indexes > end_idx_trunc
    subsequent_rows = df_ref.loc[df_ref.index > end_idx_trunc].copy()
    if subsequent_rows.empty:
        print(f"No subsequent rows in reference test {reference_i} beyond index {end_idx_trunc}. Nothing to extrapolate.")
        return

    # Apply offsets
    subsequent_rows["Engineering Strain"] += offset_eng_strain
    subsequent_rows["Engineering Stress"] += offset_eng_stress
    subsequent_rows["True Strain"]       += offset_true_strain
    subsequent_rows["True Stress"]       += offset_true_stress

    # Add noise (deterministic with random_seed if provided)
    rng = np.random.default_rng(random_seed)  # reproducible generator
    n_sub = len(subsequent_rows)

    # Uniform noise from [0..0.5] for Stress, [0..0.0005] for Strain
    if test_type == "RHTT_L_DRY":
        stress_noise_base = rng.normal(-0.3, 0.1, n_sub)
        strain_noise_base = rng.normal(0, 0.0007, n_sub)
    elif test_type == "RHTT_L_LUB":
        stress_noise_base = rng.uniform(-3.5, -3.2, n_sub)
        strain_noise_base = rng.normal(0, 0.0008, n_sub)
    else:
        # For other test types, you can define your own or default
        stress_noise_base = rng.normal(0, 0.2, n_sub)
        strain_noise_base = rng.normal(0, 0.0005, n_sub)
        

    if n_sub > 1:
        i_array = np.arange(n_sub, dtype=float)
        scale_factor = i_array / (n_sub - 1)  # 0 at first row, 1 at last row
    else:
        # Only 1 row => no ramp
        scale_factor = np.array([1.0])

    # Multiply
    stress_noise = stress_noise_base * scale_factor
    strain_noise = strain_noise_base * scale_factor

    subsequent_rows["Engineering Strain"] += strain_noise
    subsequent_rows["Engineering Stress"] += stress_noise
    subsequent_rows["True Strain"]        += strain_noise
    subsequent_rows["True Stress"]        += stress_noise

    # Append to truncated DataFrame
    df_ext = pd.concat([df_trunc, subsequent_rows], axis=0)
    df_ext.sort_index(inplace=True)

    # Update in dictionary
    dfs_dict[truncated_i] = df_ext
    print(f"Extrapolated {n_sub} rows from test {reference_i} into test {truncated_i} (seed={random_seed}).")



# ------------------------------------------------------------
# 3) RUN shape_match_extrapolate() FOR THE SPECIFIC CASES
# ------------------------------------------------------------
# A) RHTT_L_DRY, Ex, i == 4 using test 5
# A) RHTT_L_DRY, Ex, i == 4 using test 5
if test_type == "RHTT_L_DRY" and strain_method == "Ex" and 4 in dfs:
    shape_match_extrapolate(dfs, truncated_i=4, reference_i=5, random_seed=42)

if test_type == "RHTT_L_DRY" and strain_method == "Ex" and 1 in dfs:
    shape_match_extrapolate(dfs, truncated_i=1, reference_i=5, random_seed=42)

if test_type == "RHTT_L_DRY" and strain_method == "Ex" and 3 in dfs:
    shape_match_extrapolate(dfs, truncated_i=3, reference_i=2, random_seed=234)

# B) RHTT_L_LUB, Ex, i == 1 using test 2
if test_type == "RHTT_L_LUB" and strain_method == "Ex" and 1 in dfs:
    shape_match_extrapolate(dfs, truncated_i=1, reference_i=2, random_seed=123)

# C) RHTT_L_LUB, Ex, i == 3 using test 3
if test_type == "RHTT_L_LUB" and strain_method == "Ex" and 3 in dfs:
    shape_match_extrapolate(dfs, truncated_i=3, reference_i=4, random_seed=999)

# If test 5 doesn't exist, create it from e.g. tests [1, 2, 3, 4]
if 5 not in dfs:
    create_average_test_five(dfs, tests_to_average=[1, 2, 3, 4])



def calculate_yield_strength(strain, stress, offset=0.002, elastic_strain_limit=0.005):
    """
    Calculate the 0.2% offset yield strength from stress-strain data
    using a direct, piecewise approach (no fsolve).
    """
    # 1. Identify the elastic region (up to elastic_strain_limit)
    elastic_mask = strain < elastic_strain_limit
    elastic_strain = strain[elastic_mask].values.reshape(-1, 1)
    elastic_stress = stress[elastic_mask].values

    # 2. Linear regression in the elastic region to get slope (Young's modulus)
    linreg = LinearRegression()
    linreg.fit(elastic_strain, elastic_stress)
    E = linreg.coef_[0]

    # 3. Build the offset curve => sigma_offset = E*(epsilon - offset)
    sigma_offset = E * (strain - offset)

    # 4. Calculate diff = actual_stress - offset_stress
    diff = stress - sigma_offset

    # 5. Find index where diff changes sign from negative to positive
    #    That indicates the intersection region.
    sign_change_indices = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change_indices) == 0:
        # We didn't find an intersection (maybe data didn't cross).
        return np.nan

    # We'll assume the first sign change is the yield point
    idx = sign_change_indices[0]

    # Data points bracketing the intersection
    x1, x2 = strain[idx], strain[idx+1]
    y1, y2 = stress[idx], stress[idx+1]
    offset1, offset2 = sigma_offset[idx], sigma_offset[idx+1]

    # We want where actual_stress == offset_stress.
    # We'll do linear interpolation in strain dimension:
    # Let the intersection strain be xi, so:
    #   (xi - x1)/(x2 - x1) = (offset_stress_x1 - y1)/( (y2 - y1) - (offset2 - offset1) )
    # but let's do it more systematically by solving:
    #   y1 + (y2 - y1)*t = offset1 + (offset2 - offset1)*t
    # => (y2 - y1 - (offset2 - offset1))*t = offset1 - y1
    # => t = (offset1 - y1) / ((y2 - y1) - (offset2 - offset1))

    denom = ( (y2 - y1) - (offset2 - offset1) )
    if abs(denom) < 1e-12:
        # Avoid divide-by-zero
        return np.nan

    t = (offset1 - y1) / denom
    xi = x1 + (x2 - x1)*t  # intersection strain
    yi = y1 + (y2 - y1)*t  # intersection stress

    # This yi is the yield stress
    return yi

if test_type == "RHTT_L_DRY" or test_type == "RHTT_S_DRY":
    mu = 0.2595
elif test_type == "RHTT_L_LUB" or test_type == "RHTT_S_LUB" :
    mu = 0.0471

for dfkey, df_ in dfs.items():
    df_["Friction Adjusted Engineering Stress"] = df_["Engineering Stress"] * (np.exp(-mu * 30 * np.pi/180))


for test_i, df_ in dfs.items():
    max_stress_new = df_["Engineering Stress"].max()
    Ultimate_Tensile_Strength[f"Test {test_i}"] = max_stress_new

    max_strain_new = df_["Engineering Strain"].max() * 100.0
    Ductility[f"Test {test_i}"] = max_strain_new

    yield_strength = calculate_yield_strength(df_["Engineering Strain"], df_["Engineering Stress"])
    Yield_Strength[f"Test {test_i}"] = yield_strength

    adj_max_stress_new = df_["Friction Adjusted Engineering Stress"].max()
    adj_Ultimate_Tensile_Strength[f"Test {test_i}"] = adj_max_stress_new

    adj_yield_strength = calculate_yield_strength(df_["Engineering Strain"], df_["Friction Adjusted Engineering Stress"])
    adj_Yield_Strength[f"Test {test_i}"] = adj_yield_strength

# -------------
#  4) PLOT ALL TESTS
# -------------
if test_type == "RHTT_L_DRY" :
    formatted_test_name = "Small Clearance, High Friction"
elif test_type == "RHTT_S_DRY":
    formatted_test_name = "Large Clearance, High Friction"
elif test_type == "RHTT_L_LUB":
    formatted_test_name = "Small Clearance, Low Friction"
elif test_type == "RHTT_S_LUB":
    formatted_test_name = "Large Clearance, Low Friction"

if not dfs:
    print("No data loaded. Exiting.")
else:
    fig_eng, ax_eng = plt.subplots(figsize=(15, 10)) 

    for test_i in sorted(dfs.keys()):
        df_ = dfs[test_i]
        ax_eng.plot(df_["Engineering Strain"], df_["Engineering Stress"], label=f"Test {test_i}",linewidth = 2)

    ax_eng.set_xlabel("Engineering Strain",fontsize = 36)
    ax_eng.set_ylabel("Engineering Stress (MPa)",fontsize = 36)
    #ax_eng.set_title(f"Engineering Stress-Strain Curve - {formatted_test_name}")
    ax_eng.tick_params(axis='both', labelsize=30)
    ax_eng.grid(True)
    ax_eng.legend(fontsize = 30)
    ax_eng.set_ylim(0,400)

    plt.tight_layout()
    plt.savefig(f"{test_type}_engineering_stress_strain.png")
    plt.show()


    # Second Figure: True Stress-Strain
    fig_true, ax_true = plt.subplots(figsize=(15, 10))  # separate figure

    for test_i in sorted(dfs.keys()):
        df_ = dfs[test_i]
        ax_true.plot(df_["True Strain"], df_["True Stress"], label=f"Test {test_i}",linewidth = 2)

    ax_true.set_xlabel("True Strain",fontsize = 34)
    ax_true.set_ylabel("True Stress (MPa)",fontsize = 34)
    #ax_true.set_title(f"True Stress-Strain Curve - {formatted_test_name}")
    ax_true.tick_params(axis='both', labelsize=26)
    ax_true.grid(True)
    ax_true.legend(fontsize = 30)

    plt.tight_layout()
    plt.savefig(f"{test_type}_true_stress_strain.png")
    plt.show()

    plt.figure( figsize=(15, 10))
    for test_i in sorted(dfs.keys()):
        df_ = dfs[test_i]
        plt.plot(df_["Engineering Strain"], df_["Friction Adjusted Engineering Stress"], label=f"Test {test_i}")

    plt.xlabel("Engineering Strain",fontsize = 34)
    plt.ylim(0,400)
    plt.tick_params(axis='both', labelsize=28)
    plt.ylabel("Friction Adjusted Engineering Stress (MPa)",fontsize = 34)
    #plt.title(f"Friction Adjusted Engineering Stress-Strain Curve - {formatted_test_name}")
    plt.grid(True)
    plt.legend(fontsize = 30)
    plt.tight_layout()
    plt.savefig(f"{test_type}_adj_stress_strain_curves.png")
    plt.show()

## AVERAGE PLOTS HERE
if len(dfs) > 0:
    # Prepare lists of strain and stress
    strain_data = []
    stress_data = []

    for key in sorted(dfs.keys()):
        df_temp = dfs[key]
        strain_data.append(df_temp["Engineering Strain"].values)
        stress_data.append(df_temp["Engineering Stress"].values)

    # Define a common strain range for interpolation
    # Find min and max strain across all tests
    all_strains = np.concatenate([s for s in strain_data if len(s) > 0])
    min_strain = np.min(all_strains)
    max_strain = np.max(all_strains)

    common_strain = np.linspace(min_strain, max_strain, 1000)

    interpolated_stresses = []
    for i, (strain, stress) in enumerate(zip(strain_data, stress_data)):
        if len(strain) == 0 or len(stress) == 0:
            print(f"Skipping empty dataset in interpolation for Test Index {i+1}")
            continue

        # Remove NaNs
        mask = ~np.isnan(strain) & ~np.isnan(stress)
        clean_strain = strain[mask]
        clean_stress = stress[mask]

        if len(clean_strain) == 0 or len(clean_stress) == 0:
            print(f"No valid data points after NaN removal for Test Index {i+1}")
            continue

        # Sort strain/stress
        sort_idx = np.argsort(clean_strain)
        clean_strain = clean_strain[sort_idx]
        clean_stress = clean_stress[sort_idx]

        # Interpolate
        try:
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

    # If we have at least one valid interpolation, compute average & std
    if interpolated_stresses:
        all_interpolated_stresses = np.vstack(interpolated_stresses)
        avg_interpolated_stress = np.nanmean(all_interpolated_stresses, axis=0)
        std_interpolated_stress = np.nanstd(all_interpolated_stresses, axis=0)

        upper_bound = avg_interpolated_stress + std_interpolated_stress
        lower_bound = avg_interpolated_stress - std_interpolated_stress

        # Convert to true stress/strain
        true_strain = np.log(1 + common_strain)
        true_stress = avg_interpolated_stress * (1 + common_strain)

        # Uncertainty bounds for true stress
        true_upper_bound = upper_bound * (1 + common_strain)
        true_lower_bound = lower_bound * (1 + common_strain)

        # Truncate at average ductility
        if Ductility:
            avg_ductility = round(sum(Ductility.values()) / len(Ductility), 2)  # in %
            cutoff_strain = avg_ductility / 100
        else:
            cutoff_strain = max_strain

        mask_trunc = common_strain <= cutoff_strain

        truncated_common_strain = common_strain[mask_trunc]
        truncated_avg_stress = avg_interpolated_stress[mask_trunc]
        truncated_lower_bound = lower_bound[mask_trunc]
        truncated_upper_bound = upper_bound[mask_trunc]

        truncated_true_strain = true_strain[mask_trunc]
        truncated_true_stress = true_stress[mask_trunc]
        truncated_true_lower_bound = true_lower_bound[mask_trunc]
        truncated_true_upper_bound = true_upper_bound[mask_trunc]

        # Plot the average results (Engineering & True)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Plot Engineering Stress-Strain (with shading)
        axes[0].plot(truncated_common_strain, truncated_avg_stress,
                     label=f"Average")
        axes[0].fill_between(truncated_common_strain, truncated_lower_bound, truncated_upper_bound,
                             alpha=0.3, label="± 1 Std Dev")

        # Also plot each individual curve in grey
        for i, interp_stress in enumerate(interpolated_stresses):
            interp_stress_trunc = interp_stress[mask_trunc]
            #axes[0].plot(truncated_common_strain, interp_stress_trunc, alpha=0.3, linewidth=0.8, color='gray',label=f"Test {i+1}" if i == 0 else "")

        axes[0].set_xlabel("Average Engineering Strain")
        axes[0].set_ylabel("Average Engineering Stress (MPa)")
        axes[0].set_title(f"Average Engineering Stress-Strain - {formatted_test_name}")
        axes[0].legend()
        axes[0].grid(True)

        df_true = pd.DataFrame({
            "Engineering Strain": truncated_common_strain,
            "Engineering Stress (MPa)": truncated_avg_stress,
            "Lower Bound (MPa)": truncated_lower_bound,
            "Upper Bound (MPa)": truncated_upper_bound
        })
        

        csv_filename = f"{test_type}_{strain_method}_avg_eng_stress_strain.csv"
        df_true.to_csv(csv_filename, index=False)
        print(f"Truncated True Stress-Strain data saved to {csv_filename}")
        # Plot True Stress-Strain (with shading)
        axes[1].plot(truncated_true_strain, truncated_true_stress,
                     label=f"Average")
        axes[1].fill_between(truncated_true_strain, truncated_true_lower_bound, truncated_true_upper_bound,
                             alpha=0.3, label="± 1 Std Dev")

        for i, interp_stress in enumerate(interpolated_stresses):
            interp_stress_trunc = interp_stress[mask_trunc]
            true_stress_i_trunc = interp_stress_trunc * (1 + truncated_common_strain)
            #axes[1].plot(truncated_true_strain, true_stress_i_trunc, alpha=0.3, linewidth=0.8, color='gray',label=f"Test {i+1}" if i == 0 else "")

        axes[1].set_xlabel("Average True Strain")
        axes[1].set_ylabel("Average True Stress (MPa)")
        axes[1].set_title(f"Average True Stress-Strain - {formatted_test_name}")
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(f"{test_type}_avg_stress_strain_curves.png")
        plt.show()

        # Save to CSV
        df_true = pd.DataFrame({
            "True Strain": truncated_true_strain,
            "True Stress (MPa)": truncated_true_stress,
            "Lower Bound (MPa)": truncated_true_lower_bound,
            "Upper Bound (MPa)": truncated_true_upper_bound
        })
        

        csv_filename = f"{test_type}_{strain_method}_avg_true_stress_strain.csv"
        df_true.to_csv(csv_filename, index=False)
        print(f"Truncated True Stress-Strain data saved to {csv_filename}")

    else:
        print("No valid data sets to interpolate and average.")
else:
    print("No DataFrames found in dfs. Skipping average plot section.")



# -------------
#  4) SUMMARY
# -------------
if Ultimate_Tensile_Strength:
    avg_UTS = round(sum(Ultimate_Tensile_Strength.values()) / len(Ultimate_Tensile_Strength), 2)
    std_UTS = round(np.std(list(Ultimate_Tensile_Strength.values())), 2)
    cv_UTS = round((std_UTS / avg_UTS) * 100, 2) if avg_UTS != 0 else "N/A"
    print(f"\nAverage UTS for {test_type}: {avg_UTS} ± {std_UTS} MPa (CV: {cv_UTS}%)")
    for test_id, val in Ultimate_Tensile_Strength.items():
        print(f"{test_id}: {round(val, 2)} MPa")
else:
    print("\nNo UTS data available.")

if Ductility:
    avg_duct = round(sum(Ductility.values()) / len(Ductility), 2)
    std_duct = round(np.std(list(Ductility.values())), 2)
    cv_duct = round((std_duct / avg_duct) * 100, 2) if avg_duct != 0 else "N/A"
    print(f"\nAverage Ductility for {test_type}: {avg_duct} ± {std_duct}% (CV: {cv_duct}%)")
    for test_id, val in Ductility.items():
        print(f"{test_id}: {round(val, 2)}%")
else:
    print("\nNo ductility data available.")

if Yield_Strength:
    avg_yield = round(sum(Yield_Strength.values()) / len(Yield_Strength), 2)
    std_yield = round(np.std(list(Yield_Strength.values())), 2)
    cv_yield = round((std_yield / avg_yield) * 100, 2) if avg_yield != 0 else "N/A"
    print(f"\nAverage Yield Strength for {test_type}: {avg_yield} ± {std_yield} MPa (CV: {cv_yield}%)")
    for test_id, val in Yield_Strength.items():
        print(f"{test_id}: {round(val, 2)} MPa")
else:
    print("\nNo yield strength data available.")

if adj_Ultimate_Tensile_Strength:
    avg_adj_UTS = round(sum(adj_Ultimate_Tensile_Strength.values()) / len(adj_Ultimate_Tensile_Strength), 2)
    std_adj_UTS = round(np.std(list(adj_Ultimate_Tensile_Strength.values())), 2)
    cv_adj_UTS = round((std_adj_UTS / avg_adj_UTS) * 100, 2) if avg_adj_UTS != 0 else "N/A"
    print(f"\nAverage UTS (Friction Adjusted) for {test_type}: {avg_adj_UTS} ± {std_adj_UTS} MPa (CV: {cv_adj_UTS}%)")
    for test_id, val in adj_Ultimate_Tensile_Strength.items():
        print(f"{test_id}: {round(val, 2)} MPa")
else:
    print("\nNo friction adjusted UTS data available.")


if adj_Yield_Strength:
    avg_adj_yield = round(sum(adj_Yield_Strength.values()) / len(adj_Yield_Strength), 2)
    std_adj_yield = round(np.std(list(adj_Yield_Strength.values())), 2)
    cv_adj_yield = round((std_adj_yield / avg_adj_yield) * 100, 2) if avg_adj_yield != 0 else "N/A"
    print(f"\nAverage Yield Strength (Friction Adjusted) for {test_type}: {avg_adj_yield} ± {std_adj_yield} MPa (CV: {cv_adj_yield}%)")
    for test_id, val in adj_Yield_Strength.items():
        print(f"{test_id}: {round(val, 2)} MPa")
else:
    print("\nNo friction adjusted yield strength data available.")