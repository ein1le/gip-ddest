import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

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
        stress_noise_base = rng.uniform(-1, 0.1, n_sub)
        strain_noise_base = rng.normal(0, 0.0005, n_sub)
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


for test_i, df_ in dfs.items():
    max_stress_new = df_["Engineering Stress"].max()
    Ultimate_Tensile_Strength[f"Test {test_i}"] = max_stress_new

    max_strain_new = df_["Engineering Strain"].max() * 100.0
    Ductility[f"Test {test_i}"] = max_strain_new



# -------------
#  4) PLOT ALL TESTS
# -------------
if not dfs:
    print("No data loaded. Exiting.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for test_i in sorted(dfs.keys()):
        df_ = dfs[test_i]
        axes[0].plot(df_["Engineering Strain"], df_["Engineering Stress"], label=f"Test {test_i}")
        axes[1].plot(df_["True Strain"], df_["True Stress"], label=f"Test {test_i}")

    axes[0].set_xlabel("Engineering Strain")
    axes[0].set_ylabel("Engineering Stress (MPa)")
    axes[0].set_title(f"Engineering Stress-Strain Curve - {test_type}")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("True Strain")
    axes[1].set_ylabel("True Stress (MPa)")
    axes[1].set_title(f"True Stress-Strain Curve - {test_type}")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

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
