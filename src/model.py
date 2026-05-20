import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

def train_svm(X_train, y_train, C=1.0):
    base = LinearSVC(C=C, max_iter=3000, class_weight='balanced', random_state=42)
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train, C=1.0):
    model = LogisticRegression(
        C=C, max_iter=1000, class_weight='balanced',
        solver='lbfgs', random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent'])

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\nClassification Report:\n{report}")

    return {
        'name': model_name,
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'auc': auc,
        'confusion_matrix': cm, 'predictions': y_pred, 'probabilities': y_proba
    }

def plot_confusion_matrix(cm, title="Confusion Matrix", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

def plot_roc_curve(results_list, title="ROC Curves"):
    plt.figure(figsize=(8, 6))
    for res in results_list:
        plt.plot([], [], label=f"{res['name']} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_models(results_list):
    import pandas as pd
    rows = []
    for r in results_list:
        rows.append({
            'Model': r['name'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1']:.4f}",
            'ROC-AUC': f"{r['auc']:.4f}",
        })
    return pd.DataFrame(rows).set_index('Model')

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
