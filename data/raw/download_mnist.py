"""
Tải MNIST và lưu tập train/test dưới dạng .npz.
Mỗi file chứa ảnh X có shape (N, 28, 28), pixel 0->255
và nhãn y tương ứng. Lưu tại thư mục làm việc hiện tại.
"""

from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

output_dir = Path(".")

output_dir.mkdir(parents=True, exist_ok=True)

for is_train, filename in [
    (True, "mnist_train.npz"),
    (False, "mnist_test.npz"),
]:
    dataset = MNIST(
        root=str(output_dir),
        train=is_train,
        download=True,
    )

    # Giữ nguyên ảnh: (N, 28, 28), pixel từ 0 đến 255
    images = dataset.data.numpy()
    labels = dataset.targets.numpy()

    path = output_dir / filename
    np.savez_compressed(path, X=images, y=labels)

    print(f"Đã lưu {path}")
    print(f"  Shape ảnh: {images.shape}")
    print(f"  Shape nhãn: {labels.shape}")

print("Hoàn tất tải và lưu MNIST ở dạng ảnh (28 x 28).")