"""Nạp các trọng số đã lưu và suy luận chữ số MNIST."""

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
# Thống kê của tập train 54.000 ảnh:
# split phân tầng, val_size=0.1, seed=42.
TRAIN_MEAN = 0.13073687891823396
TRAIN_STD = 0.3081904023322827

@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    files: tuple[str, ...]
    description: str


MODEL_SPECS = (
    ModelSpec(
        "cnn",
        "CNN",
        ("mnist_cnn.pth",),
        "Mạng tích chập học từ ảnh MNIST gốc.",
    ),
    ModelSpec(
        "cnn_rotated",
        "CNN Rotated",
        ("mnist_cnn_rotated.pth",),
        "CNN học từ ảnh tăng cường bằng phép xoay ngẫu nhiên ±30°.",
    ),
    ModelSpec(
        "mlp",
        "MLP",
        ("mlp_model.pth",),
        "Mạng neural nhiều lớp, lan truyền ngược tự cài đặt.",
    ),
    ModelSpec(
        "logistic",
        "Logistic Regression",
        ("logistic_regression.joblib",),
        "Bộ phân loại tuyến tính đa lớp bằng NumPy.",
    ),
    ModelSpec(
        "naive_bayes",
        "Naive Bayes",
        ("naive_bayes.joblib",),
        "Gaussian Naive Bayes bằng NumPy.",
    ),
    ModelSpec(
        "random_forest",
        "Random Forest",
        ("ensemble_v1/rf_part.joblib",),
        "Mô hình Random Forest thành viên của ensemble.",
    ),
    ModelSpec(
        "knn",
        "K-Nearest Neighbors",
        ("ensemble_v1/knn_part.joblib",),
        "Mô hình KNN thành viên của ensemble.",
    ),
    ModelSpec(
        "ensemble",
        "Ensemble",
        (
            "ensemble_v1/mlp_part.pth",
            "ensemble_v1/rf_part.joblib",
            "ensemble_v1/knn_part.joblib",
            "ensemble_v1/config.json",
        ),
        "Soft voting có trọng số: MLP + Random Forest + KNN.",
    ),
)

MODELS = {spec.key: spec for spec in MODEL_SPECS}


def model_revision(spec: ModelSpec) -> tuple:
    """Kiểm tra trọng số và tạo phiên bản cho cache."""
    revision = []

    for filename in spec.files:
        path = MODELS_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(f"Thiếu models/{filename}.")

        with path.open("rb") as artifact:
            if artifact.read(80).startswith(b"version https://git-lfs"):
                raise ValueError(
                    f"models/{filename} chưa được tải. Chạy git lfs pull."
                )

        stat = path.stat()
        revision.append((stat.st_mtime_ns, stat.st_size))

    return tuple(revision)


def load_cnn(path: Path) -> CNN:
    """Khởi tạo đúng kiến trúc theo trọng số CNN đã lưu."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint.get("model_state_dict", checkpoint)

    # Hai checkpoint hiện tại đều dùng c1=32, c2=64.
    c1 = int(state["layers.0.weight"].shape[0])
    c2 = int(state["layers.3.weight"].shape[0])

    model = CNN(c1=c1, c2=c2)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def load_mlp(path: Path) -> MLP:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    model = MLP(**{
        key: config[key]
        for key in ("sizes", "p_drop", "activation", "init", "seed")
    })
    model.model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.model.eval()

    return model


def class_probabilities(
    model: object,
    features: np.ndarray,
) -> np.ndarray:
    """Sắp xếp xác suất theo đúng nhãn 0 đến 9."""
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
    mlp: MLP
    random_forest: object
    knn: object
    weights: np.ndarray

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = [
            self.mlp.predict_proba(features).cpu().numpy(),
            class_probabilities(self.random_forest, features),
            class_probabilities(self.knn, features),
        ]
        return np.average(
            values,
            axis=0,
            weights=self.weights,
        )


def load_model(key: str) -> object:
    spec = MODELS[key]
    model_revision(spec)

    path = MODELS_DIR / spec.files[0]

    if key in {"cnn", "cnn_rotated"}:
        return load_cnn(path)

    if key == "mlp":
        return load_mlp(path)

    if key == "ensemble":
        directory = MODELS_DIR / "ensemble_v1"
        config = json.loads(
            (directory / "config.json").read_text(encoding="utf-8")
        )
        weights = np.asarray(config["weights"], dtype=np.float64)

        if (
            weights.shape != (3,)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "Ensemble cần 3 trọng số không âm, tổng lớn hơn 0."
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
    pixels = np.asarray(pixels, dtype=np.float32)

    if pixels.shape != (28, 28) or not np.isfinite(pixels).all():
        raise ValueError("Đầu vào phải là ảnh 28 × 28 hợp lệ.")

    if pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("Pixel phải nằm trong khoảng 0–255.")

    unit_pixels = pixels / 255.0

    if isinstance(model, CNN):
        # CNN và CNN Rotated: chỉ chia 255.
        tensor = torch.from_numpy(
            unit_pixels.reshape(1, 1, 28, 28)
        )
        values = model(tensor).softmax(dim=1).cpu().numpy()

    else:
        # MLP, LR, GNB, RF, KNN và Ensemble:
        # chuẩn hóa bằng thống kê của tập train.
        features = (
            (unit_pixels - TRAIN_MEAN) / TRAIN_STD
        ).reshape(1, 784).astype(np.float32)

        if isinstance(model, MLP):
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
    """Đưa ảnh về MNIST: nét sáng, nền đen, kích thước 28 × 28."""
    if background not in {"auto", "light", "dark"}:
        raise ValueError("Kiểu nền không hợp lệ.")

    image = ImageOps.exif_transpose(image).convert("RGBA")
    image.thumbnail((2048, 2048))

    layer = Image.new(
        "RGBA",
        image.size,
        "black" if background == "dark" else "white",
    )
    gray = Image.alpha_composite(layer, image).convert("L")
    pixels = np.asarray(gray, dtype=np.float32)

    border = np.concatenate((
        pixels[0],
        pixels[-1],
        pixels[:, 0],
        pixels[:, -1],
    ))
    light = background == "light" or (
        background == "auto" and np.median(border) > 127
    )

    if light:
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

    border_level = (
        255 - np.median(border)
        if light
        else np.median(border)
    )
    pixels = np.clip(pixels - border_level, 0, 255)
    foreground = pixels > max(20, float(pixels.max()) * 0.15)

    if np.count_nonzero(foreground) < 4:
        raise ValueError("Nét viết quá nhỏ. Hãy vẽ lớn hơn.")

    rows, columns = np.where(foreground)
    pixels[~foreground] = 0

    crop = Image.fromarray(pixels.astype(np.uint8)).crop((
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ))

    scale = 20 / max(crop.size)
    size = tuple(max(1, round(side * scale)) for side in crop.size)
    crop = crop.resize(size, Image.Resampling.LANCZOS)

    centered = Image.new("L", (28, 28))
    centered.paste(
        crop,
        ((28 - size[0]) // 2, (28 - size[1]) // 2),
    )

    pixels = np.asarray(centered, dtype=np.float32)
    grid_y, grid_x = np.indices(pixels.shape)

    offset_x = round(
        13.5 - float((grid_x * pixels).sum() / pixels.sum())
    )
    offset_y = round(
        13.5 - float((grid_y * pixels).sum() / pixels.sum())
    )

    result = Image.new("L", (28, 28))
    result.paste(centered, (offset_x, offset_y))

    return result