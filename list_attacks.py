# list_attacks.py
import pandas as pd

df = pd.read_csv("data/Test.csv", engine="python", on_bad_lines="skip")

print("ALL UNIQUE ATTACK TYPES:")
for name in sorted(df["attack_type"].unique()):
    count = (df["attack_type"] == name).sum()
    print(f"  {name:20s}  {count:6d}")

print(f"\nTotal unique attack types: {df['attack_type'].nunique()}")