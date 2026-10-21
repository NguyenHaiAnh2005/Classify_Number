"""
Tien xu ly anh MNIST cho CNN.

Luong xu ly:
    anh [0, 255] -> chia 255 -> chuan hoa theo train -> them channel
"""

import numpy as np

IMAGE_SIZE = 28
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE  # 784
PIXEL_MAX = 255.0


def _to_image_shape(X):
    """
    Dua du lieu ve shape (N, 28, 28). Chi kiem tra shape,
    khong kiem tra gia tri pixel.
    """
    X = np.asarray(X)

    if X.ndim == 2 and X.shape[1] == NUM_PIXELS:
        X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

    if X.ndim != 3 or X.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("X phai co shape (N, 28, 28) hoac (N, 784)")

    if X.size == 0:
        raise ValueError("X khong duoc rong")

    return X


def validate_images(X):
    """
    Kiem tra anh MNIST goc (pixel trong [0, 255]).
    Input:  (N, 28, 28) hoac (N, 784)
    Output: (N, 28, 28)
    """
    X = _to_image_shape(X)

    if not np.isfinite(X).all():
        raise ValueError("Du lieu chua NaN hoac infinity")

    if X.min() < 0 or X.max() > PIXEL_MAX:
        raise ValueError("Gia tri pixel phai nam trong [0, 255]")

    return X


def normalize_images(X):
    """Kiem tra anh, roi dua pixel tu [0, 255] ve [0, 1]."""
    return validate_images(X).astype(np.float32) / PIXEL_MAX


def standardize_images(X_train, X_val, X_test):
    """
    Chuan hoa (x - mean) / std.
    mean va std chi tinh tren train, sau do dung chung cho val va test
    (tranh ro ri thong tin tu val/test).

    Input la anh goc [0, 255]. Output la anh float32 da chuan hoa.
    """
    X_train = normalize_images(X_train)
    X_val = normalize_images(X_val)
    X_test = normalize_images(X_test)

    mean = float(X_train.mean())
    std = float(X_train.std())
    if std == 0:
        std = 1.0

    def apply(X):
        return ((X - mean) / std).astype(np.float32)

    return apply(X_train), apply(X_val), apply(X_test), {"mean": mean, "std": std}


def add_channel_dimension(X, channels_last=True):
    """
    channels_last=True  -> (N, 28, 28, 1)  # TensorFlow / Keras
    channels_last=False -> (N, 1, 28, 28)  # PyTorch

    Luu y: ham nay nhan ca anh da chuan hoa (co gia tri am),
    nen chi kiem tra shape.
    """
    X = _to_image_shape(X)

    if channels_last:
        return X[..., np.newaxis]
    return X[:, np.newaxis, :, :]


def preprocess_cnn(X_train, X_val, X_test, channels_last=True):
    """
    Pipeline day du cho CNN: chuan hoa pixel + them channel dimension.
    Tra ve (X_train, X_val, X_test, tools), tools = {"mean", "std"}.
    """
    X_train, X_val, X_test, tools = standardize_images(X_train, X_val, X_test)

    X_train = add_channel_dimension(X_train, channels_last)
    X_val = add_channel_dimension(X_val, channels_last)
    X_test = add_channel_dimension(X_test, channels_last)

    return X_train, X_val, X_test, tools