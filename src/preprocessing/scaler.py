"""
scaler.py - Chia train/val và chuẩn hóa dữ liệu cho TẤT CẢ các mô hình,
gồm CNN, Naive Bayes, Perceptron, Logistic Regression, SVM, KNN, cây quyết định.

Mọi thống kê (mean, std, scaler, pixel chết) chỉ tính trên TRAIN,
rồi áp dụng cho val và test để không bị rò rỉ dữ liệu.

Cách dùng:
    from src.preprocessing.scaler import prepare
    data = prepare("standard", X_train, y_train, X_test, y_test)

kind:
    "cnn"      : ảnh (N,1,28,28) hoặc (N,28,28,1), chuẩn hóa (x-mean)/std
    "raw"      : vector [0,1]           -> GaussianNB, MultinomialNB, cây quyết định
    "binary"   : vector 0/1             -> BernoulliNB
    "standard" : vector mean 0, std 1   -> Perceptron, Logistic Regression, SVM tuyến tính
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing.image_preprocessing import augment_shift


def split_val(X, y, val_size=0.1, seed=42):
    """Tách validation từ train, giữ nguyên tỷ lệ nhãn (stratify)."""
    return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)


# ---------------------------------------------------------------- #
# CNN: giữ dạng ảnh
# ---------------------------------------------------------------- #
def scale_cnn(X_train, X_val, X_test, channels_last=True):
    """
    channels_last=True  -> (N, 28, 28, 1)  Keras / TensorFlow
    channels_last=False -> (N, 1, 28, 28)  PyTorch
    """
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    mean = X_train.mean()
    std = X_train.std()

    # Constant or invalid training data cannot be standardized.
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Training data contains invalid normalization statistics.")

    if std <= 0:
        raise ValueError("Training data must have a non-zero standard deviation.")
    
    out = []
    for X in (X_train, X_val, X_test):
        X = (X - mean) / std
        if channels_last:
            X = X.reshape(-1, 28, 28, 1)
        else:
            X = X.reshape(-1, 1, 28, 28)
        out.append(X.astype("float32"))

    return out[0], out[1], out[2], {"mean": mean, "std": std}


# ---------------------------------------------------------------- #
# Mô hình cổ điển: flatten thành vector
# ---------------------------------------------------------------- #
def scale_flat(X_train, X_val, X_test, mode="raw"):

    # Flat preprocessing does not support CNN image layouts.
    valid_modes = {"raw", "binary", "standard"}

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown mode: {mode!r}. Expected one of {sorted(valid_modes)}."
        )

    X_train = X_train.reshape(len(X_train), -1) / 255.0
    X_val = X_val.reshape(len(X_val), -1) / 255.0
    X_test = X_test.reshape(len(X_test), -1) / 255.0

    # Bỏ các pixel luôn bằng 0 trong train (pixel chết)
    # keep = X_train.max(axis=0) > 0
    # X_train = X_train[:, keep]
    # X_val = X_val[:, keep]
    # X_test = X_test[:, keep]

    # tools = {"keep_pixels": keep}
    
    tools = {}
    if mode == "binary":
        X_train = (X_train >= 0.5).astype("float32")
        X_val = (X_val >= 0.5).astype("float32")
        X_test = (X_test >= 0.5).astype("float32")

    elif mode == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        tools["scaler"] = scaler

    return (X_train.astype("float32"), X_val.astype("float32"),
            X_test.astype("float32"), tools)


# ---------------------------------------------------------------- #
# Cross-validation dùng cho tuning hyperparameter
# ---------------------------------------------------------------- #
def iter_kfold(X, y, n_splits=5, seed=42, mode="raw"):
    """Yield các fold train/val đã qua tiền xử lý kiểu ``mode``."""

    # Flat preprocessing does not support CNN image layouts.
    valid_modes = {"raw", "binary", "standard"}

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown mode: {mode!r}. Expected one of {sorted(valid_modes)}."
        )


    X = np.asarray(X)
    y = np.asarray(y)

    if n_splits < 2:
        raise ValueError("n_splits phải >= 2")
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        X_train_fold, X_val_fold, _, _ = scale_flat(
            X_train_fold, X_val_fold, X_val_fold, mode=mode
        )

        yield {
            "X_train": X_train_fold,
            "y_train": y_train_fold,
            "X_val": X_val_fold,
            "y_val": y_val_fold,
        }


# ---------------------------------------------------------------- #
# Hàm tổng hợp
# ---------------------------------------------------------------- #
def prepare(kind, X_train, y_train, X_test, y_test, val_size=0.1,
            augment=False, channels_last=True):
    """
    kind: "cnn", "raw", "binary", "standard"
    X_train, X_test: ảnh gốc (N, 28, 28) uint8.
    augment: chỉ dùng cho "cnn", tăng cường train bằng ảnh dịch.
    Trả về dict: X_train, y_train, X_val, y_val, X_test, y_test, tools.
    """

    # Reject unsupported preprocessing modes before splitting data.
    valid_kinds = {"cnn", "raw", "binary", "standard"}

    if kind not in valid_kinds:
        raise ValueError(
            f"Unknown kind: {kind!r}. Expected one of {sorted(valid_kinds)}."
        )

    X_train, X_val, y_train, y_val = split_val(X_train, y_train, val_size)

    if kind == "cnn":
        if augment:
            X_train, y_train = augment_shift(X_train, y_train)
        X_train, X_val, X_test, tools = scale_cnn(
            X_train, X_val, X_test, channels_last)
    else:
        X_train, X_val, X_test, tools = scale_flat(
            X_train, X_val, X_test, mode=kind)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "tools": tools,
    }
