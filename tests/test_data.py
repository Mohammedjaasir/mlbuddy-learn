import numpy as np
import pytest
from mlpilot.auto.data import load


def test_load_returns_four_splits():
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    result = load(X, y)
    assert len(result) == 4


def test_load_correct_sizes():
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = load(X, y, test_size=0.2)
    assert len(X_train) == 80
    assert len(X_test) == 20


def test_load_raises_on_shape_mismatch():
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 90)   # wrong size — should raise error
    with pytest.raises(ValueError):
        load(X, y)


def _plot_feature_importance(importances, model_name):
    n = len(importances)
    features = [f"Feature {i+1}" for i in range(n)]
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 4))
    plt.bar(range(n), importances[indices], color="steelblue", alpha=0.8)
    plt.xticks(range(n), [features[i] for i in indices], rotation=45, ha="right")
    plt.title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.show()
    print("  ✓ Saved as 'feature_importance.png'")
    print("  → Features on the left matter most to your model.\n")


def _plot_coefficients(coefs, model_name):
    n = len(coefs)
    features = [f"Feature {i+1}" for i in range(n)]
    colors = ["crimson" if c < 0 else "steelblue" for c in coefs]

    plt.figure(figsize=(8, 4))
    plt.bar(range(n), coefs, color=colors, alpha=0.8)
    plt.xticks(range(n), features, rotation=45, ha="right")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title(f"Model Coefficients — {model_name}", fontsize=13, fontweight="bold")
    plt.ylabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig("coefficients.png", dpi=120)
    plt.show()
    print("  ✓ Saved as 'coefficients.png'")
    print("  → Blue = pushes toward class 1. Red = pushes toward class 0.\n")