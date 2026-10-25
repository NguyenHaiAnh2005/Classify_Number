
from pathlib import Path
import sys
import joblib
import numpy as np
import torch
import streamlit as st

from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ==================================================
# 1. Đường dẫn project
# ==================================================
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.logistic_regression import LogisticRegression
from src.models.mlp import MLP

WEIGHTS = ROOT / "src" / "models" / "weights"

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Vẽ một chữ số từ 0 đến 9 và chọn mô hình để dự đoán.")


# ==================================================
# 2. Nạp Logistic Regression NumPy
# ==================================================
@st.cache_resource
def load_logistic_numpy():
    path = WEIGHTS / "logistic_regression.npz"

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    model = LogisticRegression()

    with np.load(path) as saved:
        model.W_ = saved["W"].copy()

    return model


# ==================================================
# 3. Nạp MLP PyTorch tự cài đặt
# ==================================================
@st.cache_resource
def load_mlp_torch():
    path = WEIGHTS / "mlp.pt"

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model = MLP(
        sizes=checkpoint["sizes"],
        activation="relu",
        p_drop=0.0,
        init="he"
    )

    model.model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.model.eval()

    return model


# ==================================================
# 4. Nạp Logistic Regression Scikit-learn
# ==================================================
@st.cache_resource
def load_logistic_sklearn():
    path = WEIGHTS / "logistic_regression_sklearn.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    return joblib.load(path)


# ==================================================
# 5. Nạp MLP Scikit-learn
# ==================================================
@st.cache_resource
def load_mlp_sklearn():
    path = WEIGHTS / "mlp_sklearn.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    return joblib.load(path)


# ==================================================
# 6. Nạp scaler dành cho Scikit-learn
# ==================================================
@st.cache_resource
def load_scaler_sklearn():
    path = WEIGHTS / "scaler_sklearn.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    return joblib.load(path)


# ==================================================
# 7. Tiền xử lý ảnh vẽ
# ==================================================
def preprocess_image(image):
    """
    Chuyển ảnh canvas thành vector (1, 784).

    Lưu ý:
    Đầu ra ban đầu là pixel [0, 1].
    Việc chuẩn hóa tiếp theo phụ thuộc vào mô hình.
    """

    image = image.convert("RGBA")

    # Ghép ảnh lên nền đen
    background = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 255)
    )
    image = Image.alpha_composite(background, image)

    # Chuyển sang grayscale và resize
    image = image.convert("L")
    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    X = np.asarray(image, dtype=np.float32) / 255.0
    X = X.reshape(1, 784)

    return X, image


# ==================================================
# 8. Chọn mô hình
# ==================================================
model_name = st.selectbox(
    "Chọn mô hình",
    [
        "Logistic Regression (NumPy)",
        "Logistic Regression (Scikit-learn)",
        "MLP (PyTorch tự cài đặt)",
        "MLP (Scikit-learn)"
    ]
)


# ==================================================
# 9. Canvas và khu vực kết quả
# ==================================================
col_draw, col_result = st.columns(2)

with col_draw:
    st.subheader("Vẽ chữ số")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        return_image_data=True
    )

with col_result:
    st.subheader("Kết quả dự đoán")

    if st.button("Dự đoán", type="primary"):

        if canvas_result.image_data is None:
            st.warning("Vui lòng vẽ một chữ số trước.")

        else:
            image_array = canvas_result.image_data.astype(np.uint8)

            # Kiểm tra canvas có nét vẽ không
            if image_array[..., :3].max() == 0:
                st.warning("Canvas đang trống. Hãy vẽ một chữ số.")

            else:
                try:
                    image = Image.fromarray(image_array).convert("RGBA")
                    X_raw, image_28 = preprocess_image(image)

                    # ------------------------------------------
                    # 10. Dự đoán theo mô hình được chọn
                    # ------------------------------------------
                    if model_name == "Logistic Regression (NumPy)":
                        model = load_logistic_numpy()

                        # Giả định mô hình NumPy được train với pixel [0, 1]
                        probabilities = model.predict_proba(X_raw)[0]

                    elif model_name == "Logistic Regression (Scikit-learn)":
                        model = load_logistic_sklearn()
                        scaler = load_scaler_sklearn()

                        X = scaler.transform(X_raw).astype(np.float32)
                        probabilities = model.predict_proba(X)[0]

                    elif model_name == "MLP (PyTorch tự cài đặt)":
                        model = load_mlp_torch()

                        # Giả định MLP PyTorch được train với pixel [0, 1]
                        X_tensor = torch.tensor(
                            X_raw,
                            dtype=torch.float32
                        )

                        with torch.no_grad():
                            probabilities = (
                                model.predict_proba(X_tensor)[0]
                                .detach()
                                .cpu()
                                .numpy()
                            )

                    else:
                        model = load_mlp_sklearn()
                        scaler = load_scaler_sklearn()

                        X = scaler.transform(X_raw).astype(np.float32)
                        probabilities = model.predict_proba(X)[0]

                    # ------------------------------------------
                    # 11. Hiển thị kết quả
                    # ------------------------------------------
                    prediction = int(np.argmax(probabilities))
                    confidence = float(probabilities[prediction])

                    st.success(f"Chữ số dự đoán: {prediction}")

                    st.metric(
                        "Xác suất của lớp dự đoán",
                        f"{confidence * 100:.2f}%"
                    )

                    st.write("Xác suất của từng chữ số")

                    st.bar_chart({
                        "Xác suất (%)": probabilities * 100
                    })

                    with st.expander("Xem ảnh sau tiền xử lý"):
                        st.image(
                            image_28,
                            caption="Ảnh grayscale 28 × 28",
                            width=196
                        )

                except FileNotFoundError as error:
                    st.error(f"Không tìm thấy file cần thiết: {error}")

                except Exception as error:
                    st.error(f"Lỗi khi dự đoán: {error}")
