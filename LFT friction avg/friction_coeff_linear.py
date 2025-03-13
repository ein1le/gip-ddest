import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# 1) TEST SET CONFIG + INTERPOLATED ARRAYS
# ────────────────────────────────────────────────────────────────────────────────

# Paths to your CSV files
LFT_PTFE_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_PTFE_"
LFT_2L_path   = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_2L_"
LFT_2LR_path  = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_2LR_"
LFT_SG_path  = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_SG_"
LFT_DRY_path  = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_DRY_"


# These are min/max values for each of the 5 tests in each test set
test_sets = {
    "2L": [(-243,-254),(-244,-253),(-250,-259),(-249,-270),(-249,-257)],
    "2LR": [(-251,-283),(-250,-281),(-251,-276),(-251,-290),(-253,-287)],
    "PTFE": [(-243,-251), (-244,-256), (-247,-251 ),(-249, -258),(-247, -271)],
    "DRY": [(-254,-257),(-249,-281),(-248,-267),(-251,-315),(-248,-257)],
    "SG": [(-250,-321),(-247,-333),(-248,-315),(-248,-278),(-248,-328)]
}

def create_interpolated_array(start: float, end: float, length: int = 4800) -> np.ndarray:
    """
    Creates an array of specified length interpolated between start and end values.
    Each value is rounded to 2 decimal places.
    """
    array = np.linspace(start, end, length)  # Evenly spaced values
    return np.round(array, 2)

# Build a dictionary that holds the interpolated arrays for each test set
# e.g. test_interpolations["2L"][0] => array of length 8800 for the first "2L" test
test_interpolations = {
    "2L":   [],
    "2LR":  [],
    "PTFE": [],
    "DRY":  [],
    "SG":   []
}

# Populate test_interpolations
for test_set, min_max_pairs in test_sets.items():
    for (min_val, max_val) in min_max_pairs:
        arr = create_interpolated_array(min_val, max_val, length=4800)
        test_interpolations[test_set].append(arr)

# ────────────────────────────────────────────────────────────────────────────────
# 2) READING CSV FILES + DIVIDING BY INTERPOLATED VALUES
# ────────────────────────────────────────────────────────────────────────────────

# Lists to store the computed friction coefficients
LFT_PTFE_coeffs = []
LFT_2L_coeffs   = []
LFT_2LR_coeffs  = []
LFT_DRY_coeffs = []
LFT_SG_coeffs = []

def compute_coefficients(test_name: str, base_path: str, coeff_list: list):
    """
    For each of the 5 tests in the given test set:
      1) Reads CSV (Load column).
      2) Truncates the load series to 4800 values.
      3) Truncates the corresponding interpolated array to 4800 values.
      4) Divides the load data by the truncated interpolated array.
      5) Computes mean friction coefficient.
      6) Appends to coeff_list.
    """
    for i in range(5):
        file_path = f"{base_path}{i+1}.csv"
        df = pd.read_csv(file_path)

        # Extract load data as a NumPy array and truncate to 4800 values
        load_series = df["Load(8800 (0,1):Load) (N)"].to_numpy()[:4800]

        # Retrieve the matching interpolated array and truncate to 4800 values
        interpolation = test_interpolations[test_name][i][:4800]

        # Ensure both arrays are of the same length
        if len(load_series) != 4800 or len(interpolation) != 4800:
            print(f"⚠️ WARNING: {test_name} Test {i+1} has mismatched array lengths!")
            print(f"   Load Series Length: {len(load_series)}, Interpolation Length: {len(interpolation)}")
            continue  # Skip this test if there's a length mismatch

        # Divide truncated load by truncated interpolation => friction ratio
        ratio = load_series / interpolation  # element-wise division

        # Compute mean friction coefficient for this test
        coeff_avg = float(np.mean(ratio))

        # Append the result
        coeff_list.append(coeff_avg)
        print(f"{test_name} Test {i+1}: Coeff Avg = {coeff_avg:.4f}")

# Compute friction coefficients for each test set
compute_coefficients("DRY",  LFT_DRY_path,  LFT_DRY_coeffs)
compute_coefficients("SG",   LFT_SG_path,   LFT_SG_coeffs)
compute_coefficients("PTFE", LFT_PTFE_path, LFT_PTFE_coeffs)
compute_coefficients("2L",   LFT_2L_path,   LFT_2L_coeffs)
compute_coefficients("2LR",  LFT_2LR_path,  LFT_2LR_coeffs)






# ────────────────────────────────────────────────────────────────────────────────
# 3) DISPLAY FINAL RESULTS
# ────────────────────────────────────────────────────────────────────────────────

print("\nComputed DRY Coefficients (Linear Interpolation):")
print(" | ".join(f"{c:.4f}" for c in LFT_DRY_coeffs))
print(f"Average DRY Coefficient: {sum(LFT_DRY_coeffs)/len(LFT_DRY_coeffs):.4f}")

print("\nComputed SG Coefficients (Linear Interpolation):")
print(" | ".join(f"{c:.4f}" for c in LFT_SG_coeffs))
print(f"Average SG Coefficient: {sum(LFT_SG_coeffs)/len(LFT_SG_coeffs):.4f}")

print("\nComputed PTFE Coefficients (Linear Interpolation):")
print(" | ".join(f"{c:.4f}" for c in LFT_PTFE_coeffs))
print(f"Average PTFE Coefficient: {sum(LFT_PTFE_coeffs)/len(LFT_PTFE_coeffs):.4f}")


print("\nComputed 2L Coefficients (Linear Interpolation):")
print(" | ".join(f"{c:.4f}" for c in LFT_2L_coeffs))
print(f"Average 2L Coefficient: {sum(LFT_2L_coeffs)/len(LFT_2L_coeffs):.4f}")

print("\nComputed 2LR Coefficients (Linear Interpolation):")
print(" | ".join(f"{c:.4f}" for c in LFT_2LR_coeffs))
print(f"Average 2LR Coefficient: {sum(LFT_2LR_coeffs)/len(LFT_2LR_coeffs):.4f}")

# ────────────────────────────────────────────────────────────────────────────────