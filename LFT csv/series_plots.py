import pandas as pd
import matplotlib.pyplot as plt

base_path = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT csv\LFT_2L_"

plt.figure(figsize=(10, 5))

for i in range(1, 6):
    file_path = f"{base_path}{i}.csv"
    
    try:
        df = pd.read_csv(file_path)
        load = df["Load(8800 (0,1):Load) (N)"] 
        plt.plot(load, linestyle='-', label=f"File {i}")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except KeyError:
        print(f"Column not found in: {file_path}")


plt.xlabel("Index")
plt.ylabel("Load (N)")
plt.title("Load Plots")
plt.legend()
plt.grid(True)


plt.show()
