"""
Tien xu ly anh MNIST dang vector phang (N, 784) cho model co dien
(Logistic Regression, SVM, MLP, ...), kem K-Fold cross validation.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from image_preprocessing import normalize_images, validate_images

VALID_MODES = ("raw", "standard", "binary")
BINARY_THRESHOLD = 0.5


def flatten_images(X):
    """(N, 28, 28) hoac (N, 784) -> (N, 784)."""
    X = validate_images(X)
    return X.reshape(len(X), -1)


def _normalize_and_flatten(X):
    """Anh goc [0, 255] -> vector float32 trong [0, 1], shape (N, 784)."""
    X = normalize_images(X)
    return X.reshape(len(X), -1)


def preprocess_flat(X_train, X_val, X_test=None, mode="raw"):
    """
    Tien xu ly du lieu dang vector. Moi thong ke (pixel chet, scaler)
    chi hoc tren train.

    mode:
        "raw":      chia pixel cho 255 -> [0, 1].
        "standard": raw + StandardScaler (fit tren train).
        "binary":   raw + nhi phan hoa (pixel >= 0.5 -> 1, nguoc lai 0).

    X_test co the la None (vi du khi chay K-Fold).

    Tra ve: X_train, X_val, X_test, tools
        - X_test la None neu khong truyen vao.
        - tools chua "keep_pixels" (mask pixel giu lai) va "scaler"
          (chi co khi mode="standard").
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode phai la mot trong {VALID_MODES}, nhan duoc: {mode!r}")

    X_train = _normalize_and_flatten(X_train)
    X_val = _normalize_and_flatten(X_val)
    X_test = None if X_test is None else _normalize_and_flatten(X_test)

    # Bo pixel chet (luon bang 0 tren train)
    keep_pixels = X_train.max(axis=0) > 0
    X_train = X_train[:, keep_pixels]
    X_val = X_val[:, keep_pixels]
    if X_test is not None:
        X_test = X_test[:, keep_pixels]

    tools = {"keep_pixels": keep_pixels}

    if mode == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        if X_test is not None:
            X_test = scaler.transform(X_test)
        tools["scaler"] = scaler

    elif mode == "binary":
        X_train = X_train >= BINARY_THRESHOLD
        X_val = X_val >= BINARY_THRESHOLD
        if X_test is not None:
            X_test = X_test >= BINARY_THRESHOLD

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    if X_test is not None:
        X_test = X_test.astype(np.float32)

    return X_train, X_val, X_test, tools


def iter_kfold(X, y, n_splits=5, seed=42, mode="raw"):
    """
    Sinh tung fold (Stratified K-Fold) tu du lieu train goc.
    X phai la anh chua xu ly; viec tien xu ly duoc fit lai o moi fold
    de val fold khong bi ro ri vao train.

    Moi lan yield mot dict:
        fold, X_train, y_train, X_val, y_val, tools
    """
    X = np.asarray(X)
    y = np.asarray(y)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, X_val, _, tools = preprocess_flat(
            X[train_idx], X[val_idx], mode=mode
        )

        yield {
            "fold": fold,
            "X_train": X_train,
            "y_train": y[train_idx],
            "X_val": X_val,
            "y_val": y[val_idx],
            "tools": tools,
        }