import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

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
LFT_DRY_coeffs  = []
LFT_SG_coeffs   = []

def compute_coefficients(test_name: str, base_path: str, coeff_list: list):
    """
    For each of the 5 tests in the given test set:
      1) Reads CSV (Load column).
      2) Truncates the load series to 4800 values.
      3) Truncates the corresponding interpolated array to 4800 values.
      4) Divides the load data by the truncated interpolated array => friction ratio.
      5) Computes mean friction coefficient and stores in coeff_list.
      6) After collecting all data, plots *all 5 tests* on the same figure
         with dual y-axes.
    """

    # Lists to hold arrays for plotting after processing all 5 tests
    all_load_series = []
    all_interpolations = []

    # 1) Loop through all 5 tests, read their CSV, compute friction
    for i in range(5):
        file_path = f"{base_path}{i+1}.csv"
        df = pd.read_csv(file_path)

        # Extract the load data and truncate to 4800
        load_series = df["Load(8800 (0,1):Load) (N)"].to_numpy()[:4800]

        # Retrieve the matching interpolated array
        interpolation = test_interpolations[test_name][i][:4800]

        # Quick check for any mismatch
        if len(load_series) != 4800 or len(interpolation) != 4800:
            print(f"⚠️ WARNING: {test_name} Test {i+1} has mismatched array lengths!")
            continue

        # Compute friction ratio
        ratio = load_series / interpolation
        coeff_avg = float(np.mean(ratio))
        coeff_list.append(coeff_avg)
        print(f"{test_name} Test {i+1}: Coeff Avg = {coeff_avg:.4f}")

        # Store data for plotting after the loop
        all_load_series.append(load_series)
        all_interpolations.append(interpolation)

    # 2) Plot all 5 tests in one figure with dual y-axes
    #    (Left y-axis for load_series, right y-axis for interpolation)
    if len(all_load_series) == 5:  # Only plot if we actually got 5 tests
        time = np.arange(4800) / 10.0  # index / 10 => seconds

        fig, ax1 = plt.subplots(figsize=(18, 6))

        # Plot all load_series on left y-axis
        for i, load_arr in enumerate(all_load_series, start=1):
            ax1.plot(time, load_arr, label=f"Test {i}")

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Normal Force (N)")
        ax1.grid(True)

        # Plot all interpolations on right y-axis
        # ax2 = ax1.twinx()
        # for i, interp_arr in enumerate(all_interpolations, start=1):
        #     ax2.plot(time, interp_arr, linestyle='--', label=f"Interpolation (Test {i})")

        # ax2.set_ylabel("Interpolation (N)")

        # Combine legends from both axes into one
        lines1, labels1 = ax1.get_legend_handles_labels()
        #lines2, labels2 = ax2.get_legend_handles_labels()
        #ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(f"{test_name}_load_series.png", dpi=300)
        plt.show()




        

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
print(f"Standard Deviation DRY Coefficient: {np.std(LFT_DRY_coeffs):.4f}")

print("\nComputed SG Coefficients (Linear Interpolation):")
print(" | ".join(f"{c:.4f}" for c in LFT_SG_coeffs))
print(f"Average SG Coefficient: {sum(LFT_SG_coeffs)/len(LFT_SG_coeffs):.4f}")
print(f"Standard Deviation SG Coefficient: {np.std(LFT_SG_coeffs):.4f}")

print("\nComputed PTFE Coefficients (Linear Interpolation):")
print(" | ".join(f"{c:.4f}" for c in LFT_PTFE_coeffs))
print(f"Average PTFE Coefficient: {sum(LFT_PTFE_coeffs)/len(LFT_PTFE_coeffs):.4f}")
print(f"Standard Deviation PTFE Coefficient: {np.std(LFT_PTFE_coeffs):.4f}")


print("\nComputed 2L Coefficients (Linear Interpolation):")
print(" | ".join(f"{c:.4f}" for c in LFT_2L_coeffs))
print(f"Average 2L Coefficient: {sum(LFT_2L_coeffs)/len(LFT_2L_coeffs):.4f}")
print(f"Standard Deviation 2L Coefficient: {np.std(LFT_2L_coeffs):.4f}")

print("\nComputed 2LR Coefficients (Linear Interpolation):")
print(" | ".join(f"{c:.4f}" for c in LFT_2LR_coeffs))
print(f"Average 2LR Coefficient: {sum(LFT_2LR_coeffs)/len(LFT_2LR_coeffs):.4f}")
print(f"Standard Deviation 2LR Coefficient: {np.std(LFT_2LR_coeffs):.4f}")

# ────────────────────────────────────────────────────────────────────────────────