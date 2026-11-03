"""
image_preprocessing.py - Xử lý ảnh MNIST: làm sạch và tăng cường dữ liệu.

Lưu ý: chỉ làm sạch tập TRAIN (trước khi chia validation).
Không đụng vào tập test để kết quả đánh giá không bị thay đổi.
"""
import numpy as np
from PIL import Image, ImageOps





def remove_duplicates(X, y):
    """
    Bỏ các ảnh trùng lặp hoàn toàn (giống nhau từng pixel), giữ lại bản đầu tiên.
    Làm TRƯỚC khi chia train/val để cùng một ảnh không rơi vào cả hai tập.
    """
    seen = set()
    keep = []
    for i in range(len(X)):
        key = X[i].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return X[keep], y[keep]


def remove_blank_images(X, y):
    if len(X) == 0:
        print('len(X) = 0 khong hop le')
        return None
    
    """Bỏ các ảnh trống (toàn pixel 0)."""
    keep = X.reshape(len(X), -1).max(axis=1) > 0
    return X[keep], y[keep]


# call two function above
def clean_images(X, y):
    """Làm sạch tập train: bỏ ảnh trống rồi bỏ ảnh trùng. In ra số ảnh đã bỏ."""
    n0 = len(X)
    X, y = remove_blank_images(X, y)
    n1 = len(X)
    X, y = remove_duplicates(X, y)
    n2 = len(X)
    print(f"Ảnh trống đã bỏ: {n0 - n1} | Ảnh trùng đã bỏ: {n1 - n2} | Còn lại: {n2}")
    return X, y


