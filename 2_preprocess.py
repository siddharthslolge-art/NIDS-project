# 2_preprocess.py
import pandas as pd

CSV_PATH = "data/Test.csv"

# NSL-KDD attack → category mapping
ATTACK_CATEGORIES = {
    # DoS
    "apache2": "DoS", "back": "DoS", "land": "DoS", "mailbomb": "DoS",
    "neptune": "DoS", "pod": "DoS", "processtable": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "mscan": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "saint": "Probe", "satan": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "httptunnel": "R2L",
    "imap": "R2L", "multihop": "R2L", "named": "R2L",
    "phf": "R2L", "sendmail": "R2L", "snmpgetattack": "R2L",
    "snmpguess": "R2L", "warezmaster": "R2L", "worm": "R2L",
    "xlock": "R2L", "xsnoop": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "ps": "U2R", "rootkit": "U2R", "sqlattack": "U2R", "xterm": "U2R",
    # normal
    "normal": "normal",
}

def load_and_clean(path=CSV_PATH):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["label", "attack_type"])

    # Labels
    y_binary = df["label"].astype(int).values

    attack_lower = df["attack_type"].astype(str).str.strip().str.lower()
    y_multi = attack_lower.values                            # exact attack name
    y_category = attack_lower.map(ATTACK_CATEGORIES)         # DoS/Probe/R2L/U2R/normal

    # Sanity check: any unknown attack?
    unknown = y_category[y_category.isna()].unique()
    if len(unknown) > 0:
        print("⚠️ UNMAPPED ATTACK TYPES:", unknown)
    else:
        print("✅ All attack types mapped correctly")

    # Features = everything except label columns
    X = df.drop(columns=["label", "attack_type"])

    # Ensure numeric
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes
    X = X.fillna(0)

    return X, y_binary, y_multi, y_category.values


if __name__ == "__main__":
    X, y_b, y_m, y_cat = load_and_clean()
    print(f"\nFeatures shape: {X.shape}")
    print(f"Feature columns: {X.columns.tolist()[:5]} ...")

    print("\nBinary label counts:")
    print(pd.Series(y_b).value_counts().to_dict())

    print("\nCategory counts:")
    print(pd.Series(y_cat).value_counts().to_dict())

    print("\nTop 10 attack names:")
    print(pd.Series(y_m).value_counts().head(10).to_dict())