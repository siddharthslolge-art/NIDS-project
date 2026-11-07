# part4_compare.py
"""
Part 4 - Final hold-out evaluation on Test.csv.
Loads the shared encoder + scaler from Part 2 → applies to Test.csv.
Both training and evaluation now use identical feature encoding.
"""
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE, "data", "Test.csv")

# ---------- Load Test.csv ----------
test_raw = pd.read_csv(TEST_PATH, engine="python", on_bad_lines="skip")
test_raw = test_raw.dropna(subset=["label", "attack_type"])
y_test = test_raw["label"].astype(int).values
X_test = test_raw.drop(columns=["label", "attack_type"])

# ---------- Load shared bundle ----------
bundle = joblib.load(os.path.join(BASE, "preprocessing_bundle.pkl"))
encoder_maps = bundle["encoder_maps"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
cat_cols = bundle["cat_cols"]

# Apply shared encoder (unknown categories in Test → -1)
for col in cat_cols:
    X_test[col] = X_test[col].map(encoder_maps[col]).fillna(-1).astype(int)
X_test = X_test.reindex(columns=feature_names, fill_value=0).fillna(0)

# Apply shared scaler
X_test_s = scaler.transform(X_test).astype(np.float32)
X_test_df = pd.DataFrame(X_test_s, columns=feature_names)

print("=" * 70)
print("PART 4 - FINAL HOLD-OUT EVALUATION ON Test.csv")
print("=" * 70)
print(f"Test shape: {X_test.shape}  |  Attack ratio: {y_test.mean():.2%}")

results = []

def report(name, y_pred):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    print(f"\n{'-'*70}\n{name}\n{'-'*70}")
    print(f"Accuracy : {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, zero_division=0))
    results.append({"Model": name, "Accuracy": acc,
                    "Precision": prec, "Recall": rec, "F1": f1})

# 1. RF
rf = joblib.load(os.path.join(BASE, "part2_rf.pkl"))["model"]
report("Random Forest", rf.predict(X_test_df))

# 2. KNN
knn = joblib.load(os.path.join(BASE, "part2_knn.pkl"))["model"]
report("KNN (k=5)", knn.predict(X_test_df))

# 3. NN
class NIDS_MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.net(x)

model = NIDS_MLP(len(feature_names))
model.load_state_dict(torch.load(os.path.join(BASE, "part3_nn.pt"),
                                  map_location="cpu"))
model.eval()
with torch.no_grad():
    y_pred_nn = torch.argmax(model(torch.from_numpy(X_test_s)), dim=1).numpy()
report("PyTorch NN", y_pred_nn)

# ---------- Comparison ----------
print("\n" + "=" * 70)
print("FINAL COMPARISON ON Test.csv (hold-out)")
print("=" * 70)
df = pd.DataFrame(results).round(4)
print(df.to_string(index=False))
df.to_csv(os.path.join(BASE, "part4_results.csv"), index=False)
print(f"\n✔ Saved part4_results.csv")

fig, ax = plt.subplots(figsize=(10, 5))
metrics = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(metrics))
width = 0.25
for i, row in enumerate(results):
    ax.bar(x + i * width, [row[m] for m in metrics], width, label=row["Model"])
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Comparison on Test.csv (hold-out)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
plt.savefig(os.path.join(BASE, "figures", "part4_comparison.png"), dpi=120)
plt.close()
print(f"✔ Saved figures/part4_comparison.png")