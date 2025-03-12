import pandas as pd
import matplotlib.pyplot as plt

LFT_2L_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_2L_"
LFT_2LR_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_2LR_"
LFT_PTFE_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_PTFE_"


plt.figure(figsize=(10, 5))

test_type = input("PTFE/2L/2LR: ")

if test_type == "2LR":
    for i in range(1, 6):
        file_path = f"{LFT_2LR_path}{i}.csv"
        
        try:
            df = pd.read_csv(file_path)
            load = df["Load(8800 (0,1):Load) (N)"] 
            plt.plot(load, linestyle='-', label=f"Test {i}")
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except KeyError:
            print(f"Column not found in: {file_path}")
elif test_type == "2L":
    for i in range(1, 6):
        file_path = f"{LFT_2L_path}{i}.csv"
        
        try:
            df = pd.read_csv(file_path)
            load = df["Load(8800 (0,1):Load) (N)"] 
            plt.plot(load, linestyle='-', label=f"Test {i}")
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except KeyError:
            print(f"Column not found in: {file_path}")
elif test_type == "PTFE":
    for i in range(1, 6):
        file_path = f"{LFT_PTFE_path}{i}.csv"
        
        try:
            df = pd.read_csv(file_path)
            load = df["Load(8800 (0,1):Load) (N)"] 
            plt.plot(load, linestyle='-', label=f"Test {i}")
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except KeyError:
            print(f"Column not found in: {file_path}")

else:
    print("Invalid test type")
    exit()

plt.xlabel("Index")
plt.ylabel("Load (N)")
plt.title(f"{test_type} Load Plots")
plt.legend()
plt.grid(True)


plt.show()
