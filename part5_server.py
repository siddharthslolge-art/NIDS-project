# part5_server.py
"""
Part 5 - Socket-based IDS server.
Listens for JSON network-traffic records, classifies with the trained
Random Forest (from Part 2), and returns the prediction.

Protocol:
  Client sends: JSON line = {"features": [41 numbers], "true_label": 0|1}
  Server sends: JSON line = {"prediction": 0|1, "confidence": float, "true_label": 0|1}
"""
import os
import json
import socket
import joblib
import numpy as np
import pandas as pd

HOST = "127.0.0.1"
PORT = 9999
BASE = os.path.dirname(os.path.abspath(__file__))

# ---------- Load model + preprocessing bundle ----------
rf_bundle = joblib.load(os.path.join(BASE, "part2_rf.pkl"))
model = rf_bundle["model"]

bundle = joblib.load(os.path.join(BASE, "preprocessing_bundle.pkl"))
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

print("=" * 60)
print("IDS SERVER — Part 5")
print("=" * 60)
print(f"Model: Random Forest ({len(feature_names)} features)")
print(f"Listening on {HOST}:{PORT}")

def classify(features):
    # features: list of 41 numbers, in feature_names order
    X = pd.DataFrame([features], columns=feature_names)
    X_s = scaler.transform(X)
    pred = int(model.predict(X_s)[0])
    prob = float(model.predict_proba(X_s)[0, pred])
    return pred, prob

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print("Waiting for client connection...")

    conn, addr = srv.accept()
    print(f"Client connected from {addr}")

    with conn:
        buffer = ""
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data.decode("utf-8")

            # Process each complete JSON line
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                    features = msg["features"]
                    true_label = msg.get("true_label", None)

                    pred, conf = classify(features)
                    label_str = "ATTACK 🚨" if pred == 1 else "NORMAL ✅"
                    truth_str = ("?"
                                 if true_label is None
                                 else "ATTACK" if true_label == 1 else "NORMAL")
                    match = "" if true_label is None else (
                        "✔" if pred == true_label else "�’")

                    print(f"  pred={label_str:10s} conf={conf:.2%} "
                          f"truth={truth_str:6s} {match}")

                    response = {
                        "prediction": pred,
                        "confidence": conf,
                        "true_label": true_label,
                    }
                    conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
                except Exception as e:
                    err = {"error": str(e)}
                    conn.sendall((json.dumps(err) + "\n").encode("utf-8"))

print("\nServer shutting down.")