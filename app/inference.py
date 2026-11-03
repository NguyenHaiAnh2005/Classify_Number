"""Nạp mô hình đã train và chuẩn bị đầu vào theo định dạng MNIST."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps

from src.models.cnn import CNN
from src.models.mlp import MLP

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    """Thông tin về một mô hình đã được lưu."""

    key: str
    name: str
    files: tuple[str, ...]
    description: str


MODEL_SPECS = (
    ModelSpec(
        "cnn",
        "CNN",
        ("mnist_cnn.pth",),
        "Mạng tích chập nhận dạng đặc trưng không gian của nét viết.",
    ),
    ModelSpec(
        "mlp",
        "MLP",
        ("mlp_model.pth",),
        "Mạng neural 784 → 500 → 10, lan truyền ngược tự cài đặt.",
    ),
    ModelSpec(
        "logistic",
        "Logistic Regression",
        ("logistic_regression.joblib",),
        "Bộ phân loại tuyến tính đa lớp, triển khai bằng NumPy.",
    ),
    ModelSpec(
        "naive_bayes",
        "Naive Bayes",
        ("naive_bayes.joblib",),
        "Gaussian Naive Bayes, triển khai bằng NumPy.",
    ),
    ModelSpec(
        "random_forest",
        "Random Forest",
        ("ensemble_v1/rf_part.joblib",),
        "Thành viên Random Forest đã huấn luyện của ensemble.",
    ),
    ModelSpec(
        "knn",
        "K-Nearest Neighbors",
        ("ensemble_v1/knn_part.joblib",),
        "Thành viên KNN đã huấn luyện của ensemble.",
    ),
    ModelSpec(
        "ensemble",
        "Ensemble · MLP + RF + KNN",
        (
            "ensemble_v1/mlp_part.pth",
            "ensemble_v1/rf_part.joblib",
            "ensemble_v1/knn_part.joblib",
            "ensemble_v1/config.json",
        ),
        "Soft voting theo trọng số trong config.json.",
    ),
)

MODELS = {spec.key: spec for spec in MODEL_SPECS}


def model_revision(
    spec: ModelSpec,
) -> tuple[tuple[int, int], ...]:
    """Kiểm tra artifact và tạo khóa cache theo phiên bản file."""
    revision = []

    for filename in spec.files:
        path = MODELS_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(f"Thiếu models/{filename}.")

        with path.open("rb") as artifact:
            if artifact.read(80).startswith(b"version https://git-lfs"):
                raise ValueError(
                    f"models/{filename} là con trỏ Git LFS. "
                    "Hãy chạy git lfs pull tại thư mục gốc dự án."
                )

        stat = path.stat()
        revision.append((stat.st_mtime_ns, stat.st_size))

    return tuple(revision)

def load_mlp(path: Path) -> MLP:
    """Restore MLP architecture and trained weights."""

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    # Chỉ lấy tham số dùng để khởi tạo kiến trúc MLP
    config = checkpoint["config"]

    model_config = {
        "sizes": config["sizes"],
        "p_drop": config["p_drop"],
        "activation": config["activation"],
        "init": config["init"],
        "seed": config["seed"],
    }

    # Recreate architecture
    model = MLP(**model_config)

    # Restore trained weights
    model.model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.model.eval()

    return model


def class_probabilities(
    model: object,
    features: np.ndarray,
) -> np.ndarray:
    """Sắp xếp các cột xác suất theo nhãn chữ số 0 đến 9."""
    values = np.asarray(
        model.predict_proba(features),
        dtype=np.float64,
    )
    classes = np.asarray(
        getattr(model, "classes_", np.arange(10))
    )

    if (
        values.shape != (len(features), len(classes))
        or not np.isin(classes, np.arange(10)).all()
        or len(np.unique(classes)) != len(classes)
    ):
        raise ValueError("Các lớp đầu ra của mô hình không hợp lệ.")

    aligned = np.zeros((len(features), 10), dtype=np.float64)
    aligned[:, classes.astype(int)] = values

    return aligned


@dataclass
class SoftVotingModel:
    """Kết hợp ba mô hình thành viên đã train bằng soft voting."""

    mlp: MLP
    random_forest: object
    knn: object
    weights: np.ndarray

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Tính trung bình xác suất theo trọng số đã lưu."""
        probabilities = [
            self.mlp.predict_proba(features).cpu().numpy(),
            class_probabilities(self.random_forest, features),
            class_probabilities(self.knn, features),
        ]

        return np.average(
            probabilities,
            axis=0,
            weights=self.weights,
        )


def load_model(key: str) -> object:
    """Nạp các artifact đã biết trong models/ của dự án."""
    spec = MODELS[key]
    model_revision(spec)
    path = MODELS_DIR / spec.files[0]

    if key == "cnn":
        model = CNN()
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        return model

    if key == "mlp":
        return load_mlp(path)

    if key == "ensemble":
        directory = MODELS_DIR / "ensemble_v1"
        config = json.loads(
            (directory / "config.json").read_text(encoding="utf-8")
        )
        weights = np.asarray(
            config["weights"],
            dtype=np.float64,
        )

        if (
            weights.shape != (3,)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "Ensemble cần 3 trọng số không âm và có tổng lớn hơn 0."
            )

        return SoftVotingModel(
            mlp=load_mlp(directory / "mlp_part.pth"),
            random_forest=joblib.load(directory / "rf_part.joblib"),
            knn=joblib.load(directory / "knn_part.joblib"),
            weights=weights,
        )

    return joblib.load(path)


@torch.inference_mode()
def predict_probabilities(
    model: object,
    pixels: np.ndarray,
) -> np.ndarray:
    """Đưa pixel 0–255 về đúng scale và shape dùng khi train.

    Các mô hình vector dùng scale_flat(mode="raw").
    CNN dùng transforms.ToTensor().
    Cả hai đều chỉ chia pixel cho 255.
    """
    pixels = np.asarray(pixels, dtype=np.float32)

    if pixels.shape != (28, 28) or not np.isfinite(pixels).all():
        raise ValueError("Đầu vào phải là ảnh 28 × 28 hợp lệ.")

    if pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("Pixel phải nằm trong khoảng 0–255.")

    features = (pixels / 255.0).reshape(1, 784)

    if isinstance(model, CNN):
        tensor = torch.from_numpy(
            features.reshape(1, 1, 28, 28)
        )
        values = model(tensor).softmax(dim=1).cpu().numpy()

    elif isinstance(model, MLP):
        values = model.predict_proba(features).cpu().numpy()

    else:
        values = class_probabilities(model, features)

    probabilities = np.asarray(values[0], dtype=np.float64)

    if (
        probabilities.shape != (10,)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or probabilities.sum() <= 0
    ):
        raise ValueError("Mô hình trả về xác suất không hợp lệ.")

    return probabilities / probabilities.sum()


def prepare_image(
    image: Image.Image,
    background: str = "auto",
    preserve_layout: bool = False,
) -> Image.Image:
    """Chuẩn bị ảnh MNIST, giữ tỷ lệ nét và căn tâm theo cường độ.

    Ảnh vẽ được cắt vào khung tối đa 20 x 20, sau đó đặt lên
    nền 28 x 28. Ảnh mẫu MNIST được giữ nguyên bố cục.
    """
    if background not in {"auto", "light", "dark"}:
        raise ValueError("Kiểu nền không hợp lệ.")

    image = ImageOps.exif_transpose(image).convert("RGBA")
    image.thumbnail((2048, 2048))

    color = "black" if background == "dark" else "white"
    layer = Image.new("RGBA", image.size, color)
    gray = Image.alpha_composite(layer, image).convert("L")
    pixels = np.asarray(gray, dtype=np.float32)

    border = np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1])
    )

    light_background = background == "light" or (
        background == "auto" and np.median(border) > 127
    )

    if light_background:
        pixels = 255.0 - pixels

    if float(np.ptp(pixels)) < 15:
        raise ValueError(
            "Chưa thấy nét viết. Hãy nhập một chữ số rõ nét."
        )

    if preserve_layout:
        if gray.size != (28, 28):
            raise ValueError(
                "Chỉ giữ bố cục cho ảnh MNIST 28 × 28."
            )
        return Image.fromarray(pixels.astype(np.uint8))

    # Loại mức sáng nền và nhiễu mờ.
    border_level = (
        255 - np.median(border)
        if light_background
        else np.median(border)
    )
    pixels = np.clip(pixels - border_level, 0, 255)
    foreground = pixels > max(20, float(pixels.max()) * 0.15)

    if np.count_nonzero(foreground) < 4:
        raise ValueError(
            "Nét viết quá nhỏ. Hãy vẽ lớn hơn."
        )

    rows, columns = np.where(foreground)
    pixels[~foreground] = 0

    crop = Image.fromarray(pixels.astype(np.uint8)).crop(
        (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        )
    )

    scale = 20 / max(crop.size)
    size = tuple(
        max(1, round(side * scale))
        for side in crop.size
    )
    crop = crop.resize(size, Image.Resampling.LANCZOS)

    centered = Image.new("L", (28, 28))
    centered.paste(
        crop,
        ((28 - size[0]) // 2, (28 - size[1]) // 2),
    )

    pixels = np.asarray(centered, dtype=np.float32)
    grid_y, grid_x = np.indices(pixels.shape)
    mass = pixels.sum()

    offset_x = round(
        13.5 - float((grid_x * pixels).sum() / mass)
    )
    offset_y = round(
        13.5 - float((grid_y * pixels).sum() / mass)
    )

    result = Image.new("L", (28, 28))
    result.paste(centered, (offset_x, offset_y))

    return result