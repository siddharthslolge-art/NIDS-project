# 4_predict.py
import joblib
import pandas as pd
import sys

def predict(csv_path, model_path="model_binary.pkl"):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    columns = bundle["columns"]

    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    # Remove label columns if present (we're testing)
    df = df.drop(
        columns=[c for c in ["label", "attack_type"] if c in df.columns],
        errors="ignore"
    )
    df = df.reindex(columns=columns, fill_value=0).fillna(0)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    print("=" * 60)
    print(f"Scanned {len(df)} rows from: {csv_path}")
    print("=" * 60)

    attack_count = 0
    for i, (p, prob) in enumerate(zip(preds, probs)):
        if p == 1:
            attack_count += 1
            print(f"🚨 Row {i:5d}: ATTACK  (confidence {prob:.1%})")

    normal_count = len(df) - attack_count
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   Total rows scanned : {len(df)}")
    print(f"   Predicted attacks  : {attack_count} ({attack_count/len(df):.1%})")
    print(f"   Predicted normal   : {normal_count} ({normal_count/len(df):.1%})")
    print("=" * 60)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Test.csv"
    predict(path)