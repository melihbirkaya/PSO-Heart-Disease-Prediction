# ============================================================
#  evaluate.py  –  Model değerlendirme ve raporlama
# ============================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from config import MODEL, CV_FOLDS, RANDOM_SEED


def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test : np.ndarray, y_test : np.ndarray,
    X_full : np.ndarray, y_full : np.ndarray,
    mask   : np.ndarray | None = None,
    label  : str = "Model",
) -> dict:
    """
    Logistic Regression eğitir, test ve CV metriklerini döndürür.

    Parameters
    ----------
    mask : bool ndarray veya None
        None → tüm özellikler kullanılır.

    Returns
    -------
    dict ile şu anahtarlar:
        label, accuracy, roc_auc, cv_mean, cv_std,
        confusion_matrix, classification_report,
        n_features, selected_idx
    """
    if mask is not None:
        X_tr = X_train[:, mask]
        X_te = X_test [:, mask]
        X_fl = X_full [:, mask]
        selected_idx = np.where(mask)[0].tolist()
        n_features   = int(mask.sum())
    else:
        X_tr = X_train
        X_te = X_test
        X_fl = X_full
        selected_idx = list(range(X_train.shape[1]))
        n_features   = X_train.shape[1]

    model = LogisticRegression(**MODEL)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    cv      = cross_val_score(model, X_fl, y_full, cv=CV_FOLDS, scoring="accuracy")

    print(f"\n{'='*50}")
    print(f"  {label}  ({n_features} özellik)")
    print(f"{'='*50}")
    print(f"  Test Doğruluğu : %{acc*100:.2f}")
    print(f"  ROC-AUC        : {auc:.4f}")
    print(f"  {CV_FOLDS}-Fold CV       : %{cv.mean()*100:.2f} ± %{cv.std()*100:.2f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Hastalık Yok','Hastalık Var'])}")

    return {
        "label"                  : label,
        "model"                  : model,
        "accuracy"               : acc,
        "roc_auc"                : auc,
        "cv_mean"                : cv.mean(),
        "cv_std"                 : cv.std(),
        "confusion_matrix"       : confusion_matrix(y_test, y_pred),
        "classification_report"  : classification_report(
                                       y_test, y_pred,
                                       target_names=["Hastalık Yok","Hastalık Var"]
                                   ),
        "n_features"             : n_features,
        "selected_idx"           : selected_idx,
    }
