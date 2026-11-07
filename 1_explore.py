# 1_explore.py
import pandas as pd

CSV_PATH = "data/Test.csv"

df = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip")

print("=" * 60)
print("SHAPE (rows, columns):", df.shape)
print("=" * 60)

print("\nCOLUMNS (%d total):" % len(df.columns))
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 60)
print("LABEL COUNTS (0=normal, 1=attack):")
print(df["label"].value_counts())

print("\nATTACK_TYPE COUNTS (top 15):")
print(df["attack_type"].value_counts().head(15))

print("\n" + "=" * 60)
print("MISSING VALUES (total):", df.isnull().sum().sum())

print("\n" + "=" * 60)
print("✅ EXPLORATION COMPLETE")
print(f"   Rows: {len(df)}")
print(f"   Attack ratio: {df['label'].mean():.2%}")