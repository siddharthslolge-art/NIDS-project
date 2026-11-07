# part2_ml_models.py
"""
Part 2 - Two ML models trained and compared.
  Model A: Random Forest
  Model B: K-Nearest Neighbours
Metrics: Accuracy, Precision, Recall, F1
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(BASE, "data", "Train.csv")
TEST  = os.path.join(BASE, "data", "Test.csv")

# ---------- Load ----------
train = pd.read_csv(TRAIN, engine="python", on_bad_lines="skip")
test  = pd.read_csv(TEST,  engine="python", on_bad_lines="skip")

# ---------- Feature/label split ----------
y_train = train["label"].astype(int).values
y_test  = test["label"].astype(int).values

drop_cols = [c for c in ["label", "attack_type"] if c in train.columns]
X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=[c for c in drop_cols if c in test.columns])

# Align columns (test may be missing some)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ---------- Scale (needed for KNN; harmless for RF) ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# ---------- Helper ----------
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_te)
    pred_time = time.time() - t0

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Train time: {train_time:.2f}s  |  Predict time: {pred_time:.3f}s")
    print(f"  Confusion matrix:\n{confusion_matrix(y_te, y_pred)}")

    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1,
            "TrainTime_s": round(train_time, 2)}

results = []

# ---------- Model A: Random Forest ----------
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
results.append(evaluate("Random Forest", rf, X_train_s, y_train, X_test_s, y_test))

# ---------- Model B: KNN ----------
# Subsample train for KNN speed (KNN is slow on 125k rows)
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_train_s), size=min(20000, len(X_train_s)), replace=False)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
results.append(evaluate("K-Nearest Neighbours (k=5)",
                        knn,
                        X_train_s[sample_idx], y_train[sample_idx],
                        X_test_s, y_test))

# ---------- Comparison table ----------
print(f"\n{'='*60}")
print("COMPARISON SUMMARY (Test set)")
print(f"{'='*60}")
df = pd.DataFrame(results).round(4)
print(df.to_string(index=False))

# Save for the report
df.to_csv(os.path.join(BASE, "part2_results.csv"), index=False)
print(f"\n✔ Results saved to part2_results.csv")