import pandas as pd

# ─── CONFIGURABLE VARIABLES ─────────────────────────────────────────────────────
LFT_PTFE_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_PTFE_"
LFT_2L_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_2L_"
LFT_2LR_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_2LR_"
LFT_SG_path  = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_SG_"
LFT_DRY_path  = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_DRY_"

LFT_PTFE_avg = [-247, -250, -249, -253.5, -259]
LFT_2L_avg = [-248.5,-248.5,-254.5,-259.5,-253]
LFT_2LR_avg = [-267,-265.5,-263.5,-270.5,-270]
LFT_DRY_avg = [-255.5,-265,-257.5,-283,-252.5]
LFT_SG_avg = [-285.5,-290,-281.5,-263,-288]

# ────────────────────────────────────────────────────────────────────────────────

LFT_PTFE_coeffs = []
LFT_2L_coeffs = []
LFT_2LR_coeffs = []
LFT_DRY_coeffs = []
LFT_SG_coeffs = []

def compute_coefficients(base_path, avg_list, coeff_list, test_name):
    """
    Reads CSV files, normalizes 'Load' values, computes averages,
    and stores them in the provided coefficient list.
    """
    for i in range(5):  # Loop through 5 test files
        file_path = f"{base_path}{i+1}.csv"
        df = pd.read_csv(file_path)

        # Extract load series
        load_series = df["Load(8800 (0,1):Load) (N)"]

        # Compute normalized values
        normalized_series = load_series / avg_list[i]

        # Compute average of the normalized series
        coeff_avg = float(normalized_series.mean())  # Convert to standard Python float

        # Append to list
        coeff_list.append(coeff_avg)

        # Debugging print
        print(f"Test {i+1}: Coeff Avg = {coeff_avg:.4f}")

# Compute for each dataset
compute_coefficients(LFT_2L_path, LFT_2L_avg, LFT_2L_coeffs, "2L")
compute_coefficients(LFT_2LR_path, LFT_2LR_avg, LFT_2LR_coeffs, "2LR")
compute_coefficients(LFT_PTFE_path, LFT_PTFE_avg, LFT_PTFE_coeffs, "PTFE")
compute_coefficients(LFT_DRY_path, LFT_DRY_avg, LFT_DRY_coeffs, "DRY")
compute_coefficients(LFT_SG_path, LFT_SG_avg, LFT_SG_coeffs, "SG")

# Display results with clean formatting
print("\nComputed DRY Coefficients (Simple Average):")
print(" | ".join([f"{coeff:.4f}" for coeff in LFT_DRY_coeffs]))
print(f"Average DRY Coefficient: {sum(LFT_DRY_coeffs) / len(LFT_DRY_coeffs):.4f}")

print("\nComputed SG Coefficients (Simple Average):")
print(" | ".join([f"{coeff:.4f}" for coeff in LFT_SG_coeffs]))
print(f"Average SG Coefficient: {sum(LFT_SG_coeffs) / len(LFT_SG_coeffs):.4f}")


print("\nComputed PTFE Coefficients (Simple Average):")
print(" | ".join([f"{coeff:.4f}" for coeff in LFT_PTFE_coeffs]))
print(f"Average PTFE Coefficient: {sum(LFT_PTFE_coeffs) / len(LFT_PTFE_coeffs):.4f}")

print("\nComputed 2L Coefficients (Simple Average):")
print(" | ".join([f"{coeff:.4f}" for coeff in LFT_2L_coeffs]))
print(f"Average 2L Coefficient: {sum(LFT_2L_coeffs) / len(LFT_2L_coeffs):.4f}")

print("\nComputed 2LR Coefficients (Simple Average):")
print(" | ".join([f"{coeff:.4f}" for coeff in LFT_2LR_coeffs]))
print(f"Average 2LR Coefficient: {sum(LFT_2LR_coeffs) / len(LFT_2LR_coeffs):.4f}")




# ────────────────────────────────────────────────────────────────────────────────