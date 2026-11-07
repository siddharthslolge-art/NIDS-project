# part3_pytorch_nn.py
"""
Part 3 - PyTorch Neural Network.
Reuses the SHARED encoder + scaler bundle from Part 2.
"""
import os
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE, "data", "Train.csv")

# ---------- Load raw ----------
train_raw = pd.read_csv(TRAIN_PATH, engine="python", on_bad_lines="skip")
train_raw = train_raw.dropna(subset=["label", "attack_type"])
y_binary = train_raw["label"].astype(int).values
X = train_raw.drop(columns=["label", "attack_type"])

# ---------- Load shared bundle from Part 2 ----------
bundle = joblib.load(os.path.join(BASE, "preprocessing_bundle.pkl"))
encoder_maps = bundle["encoder_maps"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
cat_cols = bundle["cat_cols"]

# Apply shared encoder
for col in cat_cols:
    X[col] = X[col].map(encoder_maps[col]).fillna(-1).astype(int)
X = X.reindex(columns=feature_names, fill_value=0).fillna(0)

# ---------- Same 80/20 split as Part 2 ----------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ---------- Apply shared scaler ----------
X_tr_s = scaler.transform(X_tr).astype(np.float32)
X_te_s = scaler.transform(X_te).astype(np.float32)

print(f"Train: {X_tr_s.shape}  |  Eval: {X_te_s.shape}")
print(f"Attack ratio (train): {y_tr.mean():.2%}")

# ---------- Tensors ----------
X_tr_t = torch.from_numpy(X_tr_s)
y_tr_t = torch.from_numpy(y_tr.astype(np.int64))
X_te_t = torch.from_numpy(X_te_s)
y_te_t = torch.from_numpy(y_te.astype(np.int64))
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=512, shuffle=True)

# ---------- Model ----------
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

model = NIDS_MLP(X_tr_s.shape[1]).to(device)
print(f"\nModel:\n{model}")

n_normal = (y_tr == 0).sum()
n_attack = (y_tr == 1).sum()
w_n = len(y_tr) / (2 * n_normal)
w_a = len(y_tr) / (2 * n_attack)
class_weights = torch.tensor([w_n, w_a], dtype=torch.float32).to(device)
print(f"Class weights: normal={w_n:.3f}, attack={w_a:.3f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
print(f"\nTraining for {EPOCHS} epochs...\n")
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    print(f"  Epoch {epoch:2d}/{EPOCHS}  |  loss = {total_loss/len(X_tr_t):.4f}")

train_time = time.time() - t0
print(f"\nTraining finished in {train_time:.1f}s")

# ---------- Evaluate ----------
model.eval()
with torch.no_grad():
    y_pred = torch.argmax(model(X_te_t.to(device)), dim=1).cpu().numpy()

acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, zero_division=0)
rec  = recall_score(y_te, y_pred, zero_division=0)
f1   = f1_score(y_te, y_pred, zero_division=0)
cm   = confusion_matrix(y_te, y_pred)

print(f"\n{'='*60}")
print("PyTorch Neural Network (MLP 41→64→32→2)")
print(f"{'='*60}")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-score : {f1:.4f}")
print(f"  Confusion matrix:\n{cm}")
print(classification_report(y_te, y_pred, zero_division=0))

torch.save(model.state_dict(), os.path.join(BASE, "part3_nn.pt"))
print(f"\n💾 Saved part3_nn.pt")

pd.DataFrame([{
    "Model": "PyTorch Neural Network",
    "Accuracy": round(acc, 4), "Precision": round(prec, 4),
    "Recall": round(rec, 4), "F1": round(f1, 4),
    "TrainTime_s": round(train_time, 1),
}]).to_csv(os.path.join(BASE, "part3_results.csv"), index=False)
print(f"✔ Results saved to part3_results.csv")