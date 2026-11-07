# part5_client.py
"""
Part 5 - Socket-based IDS client.
Reads N sample rows from Test.csv, sends them one-by-one to the server,
and prints the classification returned by the server.
"""
import os
import json
import socket
import time
import joblib
import pandas as pd

HOST = "127.0.0.1"
PORT = 9999
N_SAMPLES = 20          # number of test records to send
BASE = os.path.dirname(os.path.abspath(__file__))

# ---------- Prepare test records using the SAME preprocessing as training ----------
bundle = joblib.load(os.path.join(BASE, "preprocessing_bundle.pkl"))
encoder_maps = bundle["encoder_maps"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
cat_cols = bundle["cat_cols"]

test_raw = pd.read_csv(os.path.join(BASE, "data", "Test.csv"),
                       engine="python", on_bad_lines="skip")
test_raw = test_raw.dropna(subset=["label", "attack_type"])

# Sample a mix — half normal, half attack — so the demo is interesting
normal = test_raw[test_raw["label"] == 0].sample(N_SAMPLES // 2, random_state=42)
attack = test_raw[test_raw["label"] == 1].sample(N_SAMPLES // 2, random_state=42)
sample = pd.concat([normal, attack]).sample(frac=1, random_state=42).reset_index(drop=True)

# Apply shared encoder
X_demo = sample.drop(columns=["label", "attack_type"]).copy()
for col in cat_cols:
    X_demo[col] = X_demo[col].map(encoder_maps[col]).fillna(-1).astype(int)
X_demo = X_demo.reindex(columns=feature_names, fill_value=0).fillna(0)
X_demo_s = scaler.transform(X_demo)

y_true = sample["label"].astype(int).values
features_list = X_demo_s.tolist()

print("=" * 60)
print("IDS CLIENT — Part 5")
print("=" * 60)
print(f"Sending {N_SAMPLES} records to {HOST}:{PORT}")

# ---------- Connect + send ----------
correct = 0
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Connected.\n")

    for i, (feat, yt) in enumerate(zip(features_list, y_true)):
        msg = {"features": feat, "true_label": int(yt)}
        s.sendall((json.dumps(msg) + "\n").encode("utf-8"))

        # Read one response line
        resp_data = b""
        while not resp_data.endswith(b"\n"):
            chunk = s.recv(1024)
            if not chunk:
                break
            resp_data += chunk

        resp = json.loads(resp_data.decode("utf-8"))
        pred = resp["prediction"]
        conf = resp["confidence"]
        label_str = "ATTACK 🚨" if pred == 1 else "NORMAL ✅"
        truth_str = "ATTACK" if yt == 1 else "NORMAL"
        marker = "✔" if pred == yt else "✘"
        if pred == yt:
            correct += 1

        print(f"[{i+1:2d}/{N_SAMPLES}]  pred={label_str:10s} "
              f"conf={conf:6.2%}  truth={truth_str:6s}  {marker}")
        time.sleep(0.1)

print(f"\nDemo complete: {correct}/{N_SAMPLES} correct")