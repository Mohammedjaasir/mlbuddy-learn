from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from ..guide.suggest import suggest


MODELS = {
    "logistic":      LogisticRegression(max_iter=1000),
    "tree":          DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "svm":           SVC(),
}


class GuidedModel:
    def __init__(self, model, X_test, y_test, model_name):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

        y_pred = model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.y_pred = y_pred

    def predict(self, X):
        return self.model.predict(X)

    def explain(self):
        from ..explain.visualizer import explain_model
        explain_model(self.model, self.X_test, self.y_test, self.model_name)

    def suggest(self):
        suggest(self)


def train(X_train, X_test, y_train, y_test, model="logistic"):
    if model not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(
            f"\n✗ Unknown model '{model}'.\n"
            f"  → Available models: {available}"
        )

    print(f"\n🤖 mlpilot: Training '{model}' model...\n")

    clf = MODELS[model]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"  ✓ Training complete!")
    print(f"  📊 Accuracy: {acc * 100:.1f}%")
    print()
    print(classification_report(y_test, y_pred))
    print("💡 Tip: Call model.explain() to see what the model learned.")
    print("💡 Tip: Call model.suggest() to get advice on improving it.\n")

    return GuidedModel(clf, X_test, y_test, model)