import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load(X, y, test_size=0.2, scale=True, random_state=42):
    print("\n🔍 mlpilot: Loading your data...\n")

    if not isinstance(X, np.ndarray):
        X = np.array(X)
        print("  ✓ Converted X to numpy array")

    if not isinstance(y, np.ndarray):
        y = np.array(y)
        print("  ✓ Converted y to numpy array")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"\n✗ Mismatch: X has {X.shape[0]} rows but y has {y.shape[0]} values.\n"
            f"  → Make sure X and y have the same number of samples."
        )

    if X.shape[0] < 10:
        print("  ⚠ Warning: Less than 10 samples. ML works better with more data.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  ✓ Split: {len(X_train)} training samples, {len(X_test)} test samples")

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("  ✓ Features scaled with StandardScaler (mean=0, std=1)")

    print(f"\n  📦 Data shape: {X_train.shape[1]} features, {len(set(y))} classes")
    print("\n✅ Data ready! Pass X_train, X_test, y_train, y_test to ml.train()\n")

    return X_train, X_test, y_train, y_test