# part1_eda.py
"""
Part 1 - Exploratory Data Analysis
Produces:
  - Row/col counts for train + test
  - Descriptive statistics for train
  - Class balance (normal vs attack) in train
  - Bar chart: normal vs attack in train AND test
  - Correlation heatmap (train features)
  - Bar chart: attack category distribution (train)
Saves 4 PNGs to ./figures/
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 1. Load data ----------
train = pd.read_csv("data/Train.csv", engine="python", on_bad_lines="skip")
test  = pd.read_csv("data/Test.csv",  engine="python", on_bad_lines="skip")

print("=" * 60)
print("1. DATASET SHAPES")
print("=" * 60)
print(f"Train shape: {train.shape[0]} rows x {train.shape[1]} columns")
print(f"Test  shape: {test.shape[0]} rows x {test.shape[1]} columns")

# ---------- 2. Descriptive statistics (train) ----------
print("\n" + "=" * 60)
print("2. DESCRIPTIVE STATISTICS (Train)")
print("=" * 60)
print(train.describe().T.to_string())

# ---------- 3. Class balance ----------
# Adjust these column names if yours differ
label_col = "label"           # binary: 0=normal, 1=attack  (or however encoded)
cat_col   = "attack_type"     # multi-class: normal, DoS, Probe, R2L, U2R

def class_balance(df, col, name):
    print(f"\n{'='*60}")
    print(f"3. CLASS BALANCE - {name} ('{col}')")
    print(f"{'='*60}")
    counts = df[col].value_counts()
    pcts   = df[col].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"count": counts, "percent": pcts.round(2)})
    print(summary.to_string())
    return counts

if label_col in train.columns:
    train_counts = class_balance(train, label_col, "Train")
if label_col in test.columns:
    test_counts = class_balance(test, label_col, "Test")

# ---------- 4. Bar chart: normal vs attack (train + test) ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if label_col in train.columns:
    train_counts.plot(kind="bar", ax=axes[0], color=["#2ca02c", "#d62728"])
    axes[0].set_title("Train: Normal vs Attack")
    axes[0].set_ylabel("Count")
if label_col in test.columns:
    test_counts.plot(kind="bar", ax=axes[1], color=["#2ca02c", "#d62728"])
    axes[1].set_title("Test: Normal vs Attack")
    axes[1].set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/class_balance.png", dpi=120)
plt.close()
print(f"\n✔ Saved {FIG_DIR}/class_balance.png")

# ---------- 5. Correlation heatmap ----------
# Drop non-numeric cols (label, attack_type, maybe protocol_type/service)
num_df = train.select_dtypes(include=[np.number])
# Drop label if it's numeric so it doesn't dominate the heatmap
if label_col in num_df.columns:
    num_df = num_df.drop(columns=[label_col])

if num_df.shape[1] > 1:
    plt.figure(figsize=(14, 11))
    corr = num_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False,
                cbar_kws={"shrink": 0.7}, square=True)
    plt.title("Feature Correlation Heatmap (Train)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=120)
    plt.close()
    print(f"✔ Saved {FIG_DIR}/correlation_heatmap.png")

# ---------- 6. Attack category distribution ----------
if cat_col in train.columns:
    print(f"\n{'='*60}")
    print("4. ATTACK CATEGORY DISTRIBUTION (Train)")
    print(f"{'='*60}")
    cat_counts = train[cat_col].value_counts()
    cat_pct    = train[cat_col].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": cat_counts, "percent": cat_pct.round(2)}).to_string())

    plt.figure(figsize=(10, 5))
    cat_counts.plot(kind="bar", color="#1f77b4")
    plt.title("Attack Category Distribution (Train)")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(cat_counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/attack_categories.png", dpi=120)
    plt.close()
    print(f"✔ Saved {FIG_DIR}/attack_categories.png")

print("\n✅ Part 1 complete. Check the 'figures/' folder for 3 PNGs.")