# part2_ml_models.py
"""
Part 2 - Two ML models trained and compared honestly.
Shared encoder + scaler fit on Train.csv, saved for Part 4 to reuse.
"""
import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "data", "Train.csv")

# ---------- Load raw ----------
train_raw = pd.read_csv(TRAIN_PATH, engine="python", on_bad_lines="skip")
train_raw = train_raw.dropna(subset=["label", "attack_type"])

y_binary = train_raw["label"].astype(int).values
X = train_raw.drop(columns=["label", "attack_type"])

# ---------- Shared encoder (fit on full Train) ----------
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
encoder_maps = {}
for col in cat_cols:
    mapping = {val: i for i, val in enumerate(
        X[col].astype("category").cat.categories)}
    encoder_maps[col] = mapping
    X[col] = X[col].map(mapping).astype(int)
X = X.fillna(0)

print("=" * 60)
print("PART 2 - ML MODELS (shared encoder)")
print("=" * 60)
print(f"Full training data: {X.shape[0]} rows x {X.shape[1]} cols")
print(f"Attack ratio      : {y_binary.mean():.2%}")
print(f"Categorical cols  : {cat_cols}")

# ---------- 80/20 stratified split ----------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"Train split: {len(X_tr)}  |  Eval split: {len(X_te)}")

# ---------- Shared scaler (fit on TRAIN split only) ----------
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
X_te_s = scaler.transform(X_te).astype(np.float32)

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
    cm   = confusion_matrix(y_te, y_pred)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Train time: {train_time:.2f}s  |  Predict time: {pred_time:.3f}s")
    print(f"  Confusion matrix (rows=true, cols=pred):\n{cm}")
    print(f"  Classification report:\n{classification_report(y_te, y_pred, zero_division=0)}")

    return {"Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "TrainTime_s": round(train_time, 2)}

results = []

# ---------- Model A: Random Forest ----------
rf = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced"
)
results.append(evaluate("Random Forest", rf, X_tr_s, y_tr, X_te_s, y_te))

# ---------- Model B: KNN ----------
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_tr_s), size=min(20000, len(X_tr_s)), replace=False)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
results.append(evaluate("K-Nearest Neighbours (k=5)",
                        knn,
                        X_tr_s[sample_idx], y_tr[sample_idx],
                        X_te_s, y_te))

# ---------- Save models + encoder + scaler + feature order ----------
joblib.dump({
    "model": rf,
    "encoder_maps": encoder_maps,
    "scaler": scaler,
    "feature_names": X.columns.tolist(),
    "cat_cols": cat_cols,
}, os.path.join(BASE, "part2_rf.pkl"))

joblib.dump({
    "model": knn,
    "encoder_maps": encoder_maps,
    "scaler": scaler,
    "feature_names": X.columns.tolist(),
    "cat_cols": cat_cols,
}, os.path.join(BASE, "part2_knn.pkl"))

# Save the encoder/scaler bundle once for Part 4 to reuse
joblib.dump({
    "encoder_maps": encoder_maps,
    "scaler": scaler,
    "feature_names": X.columns.tolist(),
    "cat_cols": cat_cols,
}, os.path.join(BASE, "preprocessing_bundle.pkl"))

print(f"\n💾 Saved part2_rf.pkl, part2_knn.pkl, preprocessing_bundle.pkl")

# ---------- Comparison ----------
print(f"\n{'='*60}")
print("COMPARISON SUMMARY (Train.csv 80/20 split)")
print(f"{'='*60}")
df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv(os.path.join(BASE, "part2_results.csv"), index=False)
print(f"\n✔ Results saved to part2_results.csv")