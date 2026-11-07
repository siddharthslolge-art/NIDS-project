# 3_train_model.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from importlib import import_module

# Import our preprocess module (handles the filename starting with digit)
preprocess = import_module("2_preprocess")
X, y_binary, y_multi, y_category = preprocess.load_and_clean()

# ============================================================
# MODEL 1: BINARY (normal vs attack)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 1: BINARY (0=normal, 1=attack)")
print("=" * 60)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"Training rows: {len(X_tr)}, Test rows: {len(X_te)}")

clf_binary = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced"
)
print("Training Random Forest...")
clf_binary.fit(X_tr, y_tr)
pred = clf_binary.predict(X_te)

print("\n✅ Accuracy:", round(accuracy_score(y_te, pred), 4))
print("\nClassification report:\n", classification_report(y_te, pred))
print("Confusion matrix:\n", confusion_matrix(y_te, pred))

# ============================================================
# MODEL 2: 5-CLASS (normal / DoS / Probe / R2L / U2R)
# ============================================================
print("\n" + "=" * 60)
print("MODEL 2: CATEGORY (normal / DoS / Probe / R2L / U2R)")
print("=" * 60)

counts = pd.Series(y_category).value_counts()
can_stratify = (counts >= 2).all()
print(f"Category counts: {counts.to_dict()}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_category, test_size=0.2, random_state=42,
    stratify=y_category if can_stratify else None
)

clf_cat = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced"
)
print("Training multi-class Random Forest...")
clf_cat.fit(X_tr, y_tr)
pred = clf_cat.predict(X_te)

print("\n✅ Accuracy:", round(accuracy_score(y_te, pred), 4))
print("\nClassification report:\n",
      classification_report(y_te, pred, zero_division=0))

# ============================================================
# TOP FEATURES (for your report)
# ============================================================
print("\n" + "=" * 60)
print("🔝 TOP 15 MOST IMPORTANT FEATURES")
print("=" * 60)
importance = pd.Series(
    clf_binary.feature_importances_, index=X.columns
).sort_values(ascending=False)
print(importance.head(15).to_string())

# ============================================================
# SAVE
# ============================================================
joblib.dump(
    {"model": clf_binary, "columns": X.columns.tolist(), "type": "binary"},
    "model_binary.pkl"
)
joblib.dump(
    {"model": clf_cat, "columns": X.columns.tolist(), "type": "category"},
    "model_category.pkl"
)
print("\n💾 Saved: model_binary.pkl, model_category.pkl")