import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
input_csv = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT video\LFT_PTFE_1_vid.csv"
output_csv = r"C:\Users\USER\Desktop\Uni Files\Y4\gip-ddest\LFT video\LFT_PTFE_1_vid_cleaned.csv"
# ──────────────────────────────────────────────────────────────────────────────

def coerce_to_string(value) -> str:
    """
    Ensures the input is a string.
    If numeric (like 2748.0 or 258.0), convert to int -> string, e.g. 2748 -> "2748".
    If it's already a string, return it.
    If it's NaN, return "" (empty).
    """
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        # Convert numeric to int, then to string
        return str(int(round(value)))
    return str(value)

def clean_strain_value(old_value: str) -> str:
    """
    Original 'guidelines' logic:
      1) Replace '07' -> '02'
      2) If data is '02XX' => '002XX'
      3) If data is '02X'  => '002X0'
      4) If data is '002XXX' (6 chars) => remove the last => '002XX'
      5) Return "" if <4 chars
    """
    # 1) Replace '07' -> '02'
    value = old_value.replace("07", "02").strip()

    # 2/3/4: apply length-based rules
    if len(value) == 6 and value.startswith("002"):
        # e.g. '002123' => remove last char => '00212'
        value = value[:-1]

    if len(value) == 4 and value.startswith("02"):
        # e.g. '0212' => '00212'
        value = "0" + value

    if len(value) == 3 and value.startswith("02"):
        # e.g. '021' => '00210'
        value = "0" + value + "0"

    # Final check
    if len(value) < 4:
        return ""

    return value

def main(input_file: str, output_file: str):
    # 1) Read CSV
    df = pd.read_csv(input_file)
    print("\n[DEBUG] Initial DataFrame head():")
    print(df.head(10))
    print("\n[DEBUG] Columns in CSV:", df.columns.tolist())
    print("[DEBUG] Total rows before trimming/reindexing:", len(df))

    # 2) Check required columns
    if "strain_value" not in df.columns:
        raise ValueError("CSV must have a 'strain_value' column.")
    if "timestamp_sec" not in df.columns:
        raise ValueError("CSV must have a 'timestamp_sec' column.")

    # 3) Enforce exactly 480 rows
    if len(df) > 480:
        df = df.iloc[:480].copy()
        print(f"[DEBUG] Trimmed DataFrame to 480 rows. Now has {len(df)} rows.")
    elif len(df) < 480:
        old_len = len(df)
        df = df.reindex(range(480))
        print(f"[DEBUG] DataFrame had {old_len} rows; reindexed to 480. Now has {len(df)} rows.")

    # 4) Convert numeric => string, then apply guidelines
    def combined_cleaner(any_val):
        as_str = coerce_to_string(any_val)
        return clean_strain_value(as_str)

    df["cleaned"] = df["strain_value"].apply(combined_cleaner)
    print("\n[DEBUG] After guidelines cleaning (first 20 rows):")
    print(df["cleaned"].head(20))

    # 5) Convert the entire "cleaned" string to an integer for interpolation
    def parse_entire_string_to_int(s: str):
        """
        If string is empty -> NaN
        Else interpret it as an integer (like "2748", "0258", etc.)
        """
        if not s:
            return np.nan
        try:
            return int(s)
        except ValueError:
            return np.nan

    df["numeric_value"] = df["cleaned"].apply(parse_entire_string_to_int)
    print("\n[DEBUG] numeric_value (before interpolation) head(20):")
    print(df["numeric_value"].head(20))

    # 6) Interpolate any missing
    df["numeric_value"] = df["numeric_value"].interpolate(method="linear", limit_direction="both")
    df["numeric_value"] = df["numeric_value"].round().astype("Int64")

    print("\n[DEBUG] numeric_value (after interpolation, pre-outlier-check) head(20):")
    print(df["numeric_value"].head(20))

    # 7) Outlier check: If a value > 10% above the average of its neighbors, set it to NaN
    # We'll do this for i in [1..len-2], skipping edges
    for i in range(1, len(df) - 1):
        val = df["numeric_value"].iloc[i]
        prev_val = df["numeric_value"].iloc[i - 1]
        next_val = df["numeric_value"].iloc[i + 1]

        # If any neighbor is NaN, skip
        if pd.isna(val) or pd.isna(prev_val) or pd.isna(next_val):
            continue

        neighbor_avg = (prev_val + next_val) / 2
        # If current val is more than 10% higher than neighbor_avg => outlier
        if val > 1.1 * neighbor_avg:
            df.at[i, "numeric_value"] = np.nan

    # 8) Re-interpolate outliers
    df["numeric_value"] = df["numeric_value"].interpolate(method="linear", limit_direction="both")
    df["numeric_value"] = df["numeric_value"].round().astype("Int64")

    print("\n[DEBUG] numeric_value (after outlier-check & final interpolation) head(20):")
    print(df["numeric_value"].head(20))

    # 9) Build final strain_value:
    #    If 4 digits and starts with '2' => insert decimal => "2748" -> "274.8"
    #    Else keep as integer string => e.g. 258 -> "258"
    def numeric_to_strain(num):
        if pd.isna(num):
            return ""
        val_int = int(num)
        val_str = str(abs(val_int))  # handle negative if needed
        # if 4 digits and starts with "2" => insert decimal
        if len(val_str) == 4 and val_str.startswith("2"):
            return val_str[:-1] + "." + val_str[-1]  # e.g. "2748" -> "274.8"
        return val_str

    df["final_strain_value"] = df["numeric_value"].apply(numeric_to_strain)
    print("\n[DEBUG] final_strain_value head(20):")
    print(df["final_strain_value"].head(20))

    # Overwrite original 'strain_value'
    df["strain_value"] = df["final_strain_value"]
    df.drop(columns=["cleaned", "numeric_value", "final_strain_value"], inplace=True)

    # 10) Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"\n[DEBUG] Cleaned data saved to: {output_file}")
    print("[DEBUG] Final DataFrame head():")
    print(df.head(10))

    # 11) Plot entire strain_value as float
    def parse_strain_to_float(s):
        if not s:
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    df["plot_value"] = df["strain_value"].apply(parse_strain_to_float)
    print("\n[DEBUG] plot_value head(20):")
    print(df[["strain_value", "plot_value"]].head(20))

    # Plot the entire strain_value (float) vs. timestamp_sec
    plt.figure(figsize=(10, 5))
    plt.plot(df["timestamp_sec"], df["plot_value"], marker='o', linestyle='-')
    plt.xlabel("Timestamp (sec)")
    plt.ylabel("Strain")
    plt.title("Time")
    plt.ylim(274,275)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main(input_csv, output_csv)
