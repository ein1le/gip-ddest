import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIGURABLE VARIABLES ─────────────────────────────────────────────────────
LFT_2L_path   = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_2L_"
LFT_2LR_path  = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_2LR_"
LFT_PTFE_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_PTFE_"
LFT_DRY_path  = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_DRY_"
LFT_SG_path   = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_SG_"

# ─── PLOTTING ───────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))

test_type = input("PTFE/2L/2LR/DRY/SG: ")

test_paths = {
    "2LR": LFT_2LR_path,
    "2L": LFT_2L_path,
    "DRY": LFT_DRY_path,
    "PTFE": LFT_PTFE_path,
    "SG": LFT_SG_path
}

if test_type not in test_paths:
    print("Invalid test type")
    exit()

# Load and plot the data
for i in range(1, 6):
    file_path = f"{test_paths[test_type]}{i}.csv"
    
    try:
        df = pd.read_csv(file_path)
        load = df["Load(8800 (0,1):Load) (N)"]

        # Compute time axis assuming 10 Hz recording rate
        time = np.arange(0, len(load)) * 0.1  # Each index is 0.1s

        plt.plot(time, load, linestyle='-', label=f"Test {i}")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except KeyError:
        print(f"❌ Column not found in: {file_path}")

# ─── FINALIZE PLOT ─────────────────────────────────────────────────────────────
plt.xlabel("Time (s)")  # Updated to Time instead of Index
plt.ylabel("Load (N)")
plt.title(f"{test_type} Load vs. Time")
plt.xlim(0,480)
plt.legend()
plt.grid(True)

plt.show()
