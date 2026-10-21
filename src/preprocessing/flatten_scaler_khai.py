"""
Tien xu ly anh MNIST dang vector phang (N, 784) cho cac model co dien
(Logistic Regression, MLP, Naive Bayes, Random Forest, ...),
kem Stratified K-Fold cross validation.

Vi du su dung:
    # Chia train/val/test co dinh
    X_tr, X_va, X_te, tools = preprocess_flat(X_train, X_val, X_test, mode="standard")

    # K-Fold tren tap train goc
    for fold in iter_kfold(X_train, y_train, n_splits=5, mode="raw"):
        model.fit(fold["X_train"], fold["y_train"])
        score = model.score(fold["X_val"], fold["y_val"])
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Dung lai ham kiem tra o file kia de khong viet trung code.
# Neu da chuyen vao package (src/preprocessing/), doi thanh:
#     from .image_preprocessing import normalize_images, validate_images
from image_preprocessing import normalize_images, validate_images

# ----- Hang so -----
VALID_MODES = ("raw", "standard", "binary")  # cac che do tien xu ly hop le
BINARY_THRESHOLD = 0.5                       # nguong nhi phan hoa (tren thang [0, 1])


def flatten_images(X):
    """
    Trai phang anh: (N, 28, 28) hoac (N, 784) -> (N, 784).
    Co kiem tra hop le (shape, NaN, khoang [0, 255]) nhung KHONG chia 255.
    """
    X = validate_images(X)
    return X.reshape(len(X), -1)


def _normalize_and_flatten(X):
    """Anh goc [0, 255] -> vector float32 trong [0, 1], shape (N, 784)."""
    X = normalize_images(X)
    return X.reshape(len(X), -1)


def preprocess_flat(X_train, X_val, X_test=None, mode="raw"):
    """
    Tien xu ly du lieu dang vector.

    Nguyen tac chung: moi thong ke (pixel chet, scaler) CHI hoc tren train,
    roi ap dung cho val va test. Nhu vay val/test khong anh huong den
    qua trinh huan luyen (tranh data leakage).

    mode:
        "raw":      chia pixel cho 255 -> [0, 1].
                    Phu hop: Random Forest, MultinomialNB, MLP don gian.
        "standard": raw + StandardScaler (moi pixel co mean 0, std 1, fit tren train).
                    Phu hop: Logistic Regression, MLP, SVM, GaussianNB.
        "binary":   raw + nhi phan hoa (pixel >= 0.5 -> 1, nguoc lai -> 0).
                    Phu hop: BernoulliNB.

    X_test co the la None (vi du khi chay K-Fold, chi co train va val).

    Tra ve: X_train, X_val, X_test, tools
        - X_test la None neu khong truyen vao.
        - tools["keep_pixels"]: mask boolean (784,) cho biet pixel nao duoc giu.
          Can dung lai khi du doan anh moi.
        - tools["scaler"]: StandardScaler da fit (chi co khi mode="standard").
    """
    # Kiem tra mode ngay tu dau de bao loi som, truoc khi xu ly du lieu
    if mode not in VALID_MODES:
        raise ValueError(f"mode phai la mot trong {VALID_MODES}, nhan duoc: {mode!r}")

    # Buoc 1: kiem tra + chia 255 + trai phang
    X_train = _normalize_and_flatten(X_train)
    X_val = _normalize_and_flatten(X_val)
    X_test = None if X_test is None else _normalize_and_flatten(X_test)

    # Buoc 2: bo pixel "chet" (luon bang 0 tren train, vi du cac pixel o vien anh).
    # Chi xac dinh tren train de khong dung thong tin tu val/test.
    keep_pixels = X_train.max(axis=0) > 0
    X_train = X_train[:, keep_pixels]
    X_val = X_val[:, keep_pixels]
    if X_test is not None:
        X_test = X_test[:, keep_pixels]

    tools = {"keep_pixels": keep_pixels}

    # Buoc 3: bien doi theo mode
    if mode == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)  # hoc mean/std tren train
        X_val = scaler.transform(X_val)          # chi ap dung, khong hoc lai
        if X_test is not None:
            X_test = scaler.transform(X_test)
        tools["scaler"] = scaler

    elif mode == "binary":
        X_train = X_train >= BINARY_THRESHOLD
        X_val = X_val >= BINARY_THRESHOLD
        if X_test is not None:
            X_test = X_test >= BINARY_THRESHOLD

    # mode == "raw": khong can lam gi them

    # Buoc 4: dua ve float32 de dong nhat kieu du lieu dau ra
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    if X_test is not None:
        X_test = X_test.astype(np.float32)

    return X_train, X_val, X_test, tools


def iter_kfold(X, y, n_splits=5, seed=42, mode="raw"):
    """
    Sinh tung fold (Stratified K-Fold) tu du lieu train goc.

    - Stratified: ti le cac chu so 0-9 trong moi fold giong tap goc.
    - X phai la anh CHUA xu ly. Tien xu ly duoc fit lai o MOI fold
      (chi tren phan train cua fold do), neu fit truoc khi chia fold
      thi val fold se bi ro ri vao train.
    - Khong co tap test trong K-Fold; test nen de rieng, chi dung o cuoi.

    Tham so:
        n_splits: so fold.
        seed:     random_state, giup chia fold tai lap duoc.
        mode:     xem preprocess_flat.

    Moi lan yield mot dict:
        {"fold": so thu tu fold (tu 1), "X_train", "y_train",
         "X_val", "y_val", "tools"}
    """
    X = np.asarray(X)
    y = np.asarray(y)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        # Tien xu ly rieng cho tung fold (khong truyen test)
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
