"""
Tien xu ly anh MNIST cho CNN.

Luong xu ly:
    anh goc [0, 255]
      -> kiem tra hop le (shape, NaN, khoang gia tri)
      -> chia 255 ve [0, 1]
      -> chuan hoa (x - mean) / std, mean/std tinh tren train
      -> them channel dimension

Vi du su dung:
    X_tr, X_va, X_te, tools = preprocess_cnn(X_train, X_val, X_test)
    # tools = {"mean": ..., "std": ...} -> luu lai de dung khi du doan anh moi
"""

import numpy as np

# ----- Hang so -----
IMAGE_SIZE = 28                       # MNIST la anh 28 x 28
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE  # 784 pixel khi trai phang
PIXEL_MAX = 255.0                     # gia tri pixel lon nhat cua anh goc


def _to_image_shape(X):
    """
    Dua du lieu ve shape (N, 28, 28).
    Chi kiem tra shape, KHONG kiem tra gia tri pixel
    (de dung duoc ca cho anh da chuan hoa, co gia tri am).

    Chap nhan:
        (N, 28, 28)  -> giu nguyen
        (N, 784)     -> reshape thanh (N, 28, 28)
    """
    X = np.asarray(X)

    # Dang vector phang -> doi lai thanh anh 2D
    if X.ndim == 2 and X.shape[1] == NUM_PIXELS:
        X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

    # Sau buoc tren, du lieu hop le chi co the la (N, 28, 28)
    if X.ndim != 3 or X.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("X phai co shape (N, 28, 28) hoac (N, 784)")

    # Mang rong se lam X.min() / X.max() bi loi o buoc sau
    if X.size == 0:
        raise ValueError("X khong duoc rong")

    return X


def validate_images(X):
    """
    Kiem tra anh MNIST GOC (pixel nam trong [0, 255]).

    Input:  (N, 28, 28) hoac (N, 784)
    Output: (N, 28, 28)

    Luu y: chi goi ham nay voi anh chua chuan hoa.
    Anh da chuan hoa (co gia tri am) se bi bao loi o buoc kiem tra khoang.
    """
    X = _to_image_shape(X)

    # NaN / inf thuong den tu loi doc file hoac tien xu ly truoc do
    if not np.isfinite(X).all():
        raise ValueError("Du lieu chua NaN hoac infinity")

    if X.min() < 0 or X.max() > PIXEL_MAX:
        raise ValueError("Gia tri pixel phai nam trong [0, 255]")

    return X


def normalize_images(X):
    """
    Kiem tra anh, roi dua pixel tu [0, 255] ve [0, 1] (float32).
    Day la buoc "min-max" don gian, khong hoc thong ke nao tu du lieu.
    """
    return validate_images(X).astype(np.float32) / PIXEL_MAX


def standardize_images(X_train, X_val, X_test):
    """
    Chuan hoa (x - mean) / std.

    QUAN TRONG: mean va std CHI tinh tren tap train, sau do dung chung
    cho val va test. Neu tinh tren ca val/test thi thong tin cua chung
    "ro ri" vao qua trinh train (data leakage) va ket qua danh gia
    se cao hon thuc te.

    Day la chuan hoa TOAN CUC (1 mean, 1 std cho ca anh), khong phai
    theo tung pixel. Cach nay pho bien cho CNN tren MNIST.

    Input:  3 tap anh goc [0, 255]
    Output: (X_train, X_val, X_test, {"mean": ..., "std": ...}), float32
    """
    # Buoc 1: kiem tra + dua ve [0, 1]
    X_train = normalize_images(X_train)
    X_val = normalize_images(X_val)
    X_test = normalize_images(X_test)

    # Buoc 2: hoc thong ke tren train
    mean = float(X_train.mean())
    std = float(X_train.std())

    # Tranh chia cho 0 (vi du khi toan bo anh train giong het nhau)
    if std == 0:
        std = 1.0

    # Buoc 3: ap dung cung mot cong thuc cho ca 3 tap
    def apply(X):
        return ((X - mean) / std).astype(np.float32)

    return apply(X_train), apply(X_val), apply(X_test), {"mean": mean, "std": std}


def add_channel_dimension(X, channels_last=True):
    """
    Them chieu kenh (channel) vi CNN yeu cau input co kenh.
    MNIST la anh xam nen chi co 1 kenh.

        channels_last=True  -> (N, 28, 28, 1)  # TensorFlow / Keras
        channels_last=False -> (N, 1, 28, 28)  # PyTorch

    Luu y: ham nay nhan ca anh da chuan hoa (co gia tri am),
    nen chi kiem tra shape, khong kiem tra khoang gia tri.
    """
    X = _to_image_shape(X)

    if channels_last:
        return X[..., np.newaxis]        # them truc cuoi
    return X[:, np.newaxis, :, :]        # them truc thu 2 (sau N)


def preprocess_cnn(X_train, X_val, X_test, channels_last=True):
    """
    Pipeline day du cho CNN: chuan hoa pixel + them channel dimension.

    Tham so:
        channels_last: True cho TensorFlow/Keras, False cho PyTorch.

    Tra ve:
        X_train, X_val, X_test: float32, shape (N, 28, 28, 1) hoac (N, 1, 28, 28)
        tools: {"mean", "std"} - can luu lai de xu ly anh moi khi du doan
    """
    X_train, X_val, X_test, tools = standardize_images(X_train, X_val, X_test)

    X_train = add_channel_dimension(X_train, channels_last)
    X_val = add_channel_dimension(X_val, channels_last)
    X_test = add_channel_dimension(X_test, channels_last)

    return X_train, X_val, X_test, tools
