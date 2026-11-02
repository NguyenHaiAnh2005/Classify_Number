"""Giao diện Webify Dark cho các mô hình MNIST trong models/."""

import hashlib
import io
import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import (  # noqa: E402
    MODEL_SPECS,
    MODELS,
    load_model,
    model_revision,
    predict_probabilities,
    prepare_image,
)

LOGGER = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False, max_entries=14)
def get_model(key: str, revision: tuple) -> object:
    """Nạp lại mô hình khi phiên bản file thay đổi."""
    return load_model(key)


@st.cache_data(show_spinner=False)
def load_samples() -> tuple[np.ndarray, np.ndarray]:
    """Đọc ảnh MNIST; nhãn thật không được đưa vào mô hình."""
    path = ROOT / "data" / "raw" / "mnist_test.npz"

    with np.load(path) as data:
        return data["X"], data["y"]


def clear_drawing() -> None:
    """Tạo canvas mới và xóa kết quả của ảnh trước."""
    st.session_state.canvas_version += 1
    st.session_state.pop("prediction", None)


def next_sample() -> None:
    """Chọn ảnh mẫu khác của chữ số hiện tại."""
    st.session_state.sample_offset += 1


def render_header(available_count: int) -> None:
    """Hiển thị tên ứng dụng và số mô hình có sẵn."""
    st.html(
        '<div class="topbar">'
        '<div class="brand">DIGIT RECOGNITION</div>'
        f'<span class="status-pill"><i></i>{available_count} mô hình '
        'có sẵn</span></div>'
    )


def render_drawing() -> Image.Image | None:
    """Hiển thị canvas và đọc ảnh RGBA từ nét viết."""
    controls, action = st.columns(
        [2, 1],
        vertical_alignment="bottom",
    )

    with controls:
        stroke_width = st.slider(
            "Độ dày nét",
            min_value=8,
            max_value=32,
            value=18,
            step=2,
        )

    with action:
        st.button(
            "Xóa nét vẽ",
            on_click=clear_drawing,
            width="stretch",
        )

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key=f"digit_canvas_{st.session_state.canvas_version}",
    )

    st.caption(
        "Vẽ một chữ số 0–9 ở giữa khung. "
        "Dùng nét liền, rõ và đủ lớn."
    )

    if canvas.image_data is None:
        return None

    return Image.fromarray(
        canvas.image_data.astype(np.uint8)
    )


def render_sample() -> Image.Image | None:
    """Chọn ảnh MNIST có sẵn để thử ứng dụng."""
    try:
        images, labels = load_samples()
    except (OSError, KeyError, ValueError):
        st.info(
            "Chưa có data/raw/mnist_test.npz. "
            "Bạn vẫn có thể vẽ chữ số."
        )
        return None

    digit = st.select_slider(
        "Chữ số mẫu",
        options=list(range(10)),
        value=7,
    )

    indices = np.flatnonzero(labels == digit)

    if not len(indices):
        st.warning("Tập dữ liệu chưa có ảnh của chữ số này.")
        return None

    index = indices[
        st.session_state.sample_offset % len(indices)
    ]
    image = Image.fromarray(
        images[index].astype(np.uint8)
    )

    st.image(
        image,
        width=280,
        caption=f"Ảnh MNIST · nhãn thật: {digit}",
    )
    st.button("Đổi ảnh mẫu", on_click=next_sample)
    st.caption(
        "Nhãn thật chỉ để đối chiếu; "
        "mô hình chỉ nhận pixel của ảnh."
    )

    return image


def run_predictions(
    pixels: np.ndarray,
    keys: list[str],
) -> dict:
    """Chạy suy luận và ghi nhận lỗi riêng của từng mô hình."""
    results, errors = {}, {}

    for key in keys:
        try:
            model = get_model(
                key,
                model_revision(MODELS[key]),
            )

            start = perf_counter()
            probabilities = predict_probabilities(model, pixels)
            elapsed_ms = (perf_counter() - start) * 1000

            results[key] = {
                "probabilities": probabilities,
                "digit": int(probabilities.argmax()),
                "confidence": float(probabilities.max()),
                "elapsed_ms": elapsed_ms,
            }

        except Exception as error:
            LOGGER.exception("Không dự đoán được bằng %s", key)
            errors[key] = str(error)

    return {"results": results, "errors": errors}


def render_probabilities(probabilities: np.ndarray) -> None:
    """Hiển thị phân phối xác suất của đủ 10 chữ số."""
    winner = int(probabilities.argmax())
    bars = []

    for digit, probability in enumerate(probabilities):
        percent = float(probability) * 100
        active = " active" if digit == winner else ""

        bars.append(
            f'<div class="probability-row{active}">'
            f'<span class="digit-label">{digit}</span>'
            '<div class="probability-track" role="meter" '
            f'aria-label="Xác suất chữ số {digit}" '
            'aria-valuemin="0" aria-valuemax="100" '
            f'aria-valuenow="{percent:.2f}">'
            f'<div style="width:{percent:.4f}%"></div></div>'
            f'<span class="probability-value">{percent:.1f}%</span>'
            '</div>'
        )

    st.html(
        '<div class="probability-chart">'
        + "".join(bars)
        + "</div>"
    )


def render_result(
    prediction: dict | None,
    selected_key: str,
) -> None:
    """Hiển thị kết quả hiện tại và bảng so sánh mô hình."""
    st.html(
        '<div class="eyebrow panel-label">'
        '02 / KẾT QUẢ NHẬN DẠNG</div>'
    )

    if prediction is None:
        st.html(
            '<div class="empty-result">'
            '<div class="empty-digit">?</div>'
            '<h3>Mỗi nét vẽ, một khả năng.</h3>'
            '<p>Thêm một chữ số ở bên trái rồi nhấn<br>'
            '<strong>Nhận dạng chữ số</strong> để xem kết quả.</p>'
            '<span class="small-pill">10 lớp · từ 0 đến 9</span>'
            '</div>'
        )
        return

    result = prediction["results"].get(selected_key)

    if result is not None:
        st.html(
            '<div class="prediction-hero"><div>'
            '<span class="muted">CHỮ SỐ DỰ ĐOÁN</span>'
            f'<div class="prediction-digit">{result["digit"]}</div>'
            '</div><div class="prediction-summary">'
            f'<span class="confidence">{result["confidence"]:.1%}</span>'
            '<span class="muted">Xác suất dự đoán</span>'
            f'<span class="small-pill">{result["elapsed_ms"]:.0f} ms'
            '</span></div></div>'
        )

        st.caption(MODELS[selected_key].name)
        st.markdown("**Phân phối xác suất**")
        render_probabilities(result["probabilities"])
        st.caption(
            "Xác suất do mô hình ước lượng, "
            "không phải độ chính xác thực tế."
        )

    for key, error in prediction["errors"].items():
        st.error(f"Không chạy được {MODELS[key].name}.")

        with st.expander("Chi tiết lỗi mô hình"):
            st.code(error, language=None)

    if len(prediction["results"]) > 1:
        with st.expander(
            "Đối chiếu các mô hình",
            expanded=True,
        ):
            rows = [
                {
                    "Mô hình": MODELS[key].name,
                    "Dự đoán": item["digit"],
                    "Xác suất": f'{item["confidence"]:.1%}',
                    "Thời gian": f'{item["elapsed_ms"]:.0f} ms',
                }
                for key, item in prediction["results"].items()
            ]

            st.dataframe(
                rows,
                hide_index=True,
                width="stretch",
            )


def render_catalog(unavailable: dict[str, str]) -> None:
    """Liệt kê các file mô hình được ứng dụng sử dụng."""
    with st.expander(
        "Kho mô hình · xem các mô hình đã huấn luyện"
    ):
        for spec in MODEL_SPECS:
            status = (
                "Chưa sẵn sàng"
                if spec.key in unavailable
                else "Có sẵn"
            )

            st.markdown(f"**{spec.name}** · {status}")
            st.caption(spec.description)
            st.code(
                "\n".join(
                    f"models/{filename}"
                    for filename in spec.files
                )
            )

            if spec.key in unavailable:
                st.warning(unavailable[spec.key])

        st.caption(
            "Random Forest và KNN là các thành viên đã train "
            "của ensemble. Ứng dụng không huấn luyện lại."
        )


def render_processed_image(image: Image.Image) -> None:
    """Hiển thị và cho phép tải ảnh sau tiền xử lý."""
    with st.expander("Xem ảnh sau tiền xử lý"):
        st.image(
            image,
            width=140,
            caption="Ảnh đầu vào 28 × 28",
        )
        st.caption(
            "Nét sáng trên nền đen · pixel chia 255 về [0, 1]. "
            "CNN: (1, 1, 28, 28). "
            "Các mô hình còn lại: (1, 784)."
        )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        st.download_button(
            "Tải ảnh 28 × 28",
            data=buffer.getvalue(),
            file_name="digit_28x28.png",
            mime="image/png",
        )


def main() -> None:
    """Tổ chức giao diện, kiểm tra đầu vào và chạy suy luận."""
    st.set_page_config(
        page_title="DIGIT RECOGNITION",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    stylesheet = (APP_DIR / "styles.css").read_text(
        encoding="utf-8"
    )
    st.html("<style>" + stylesheet + "</style>")

    st.session_state.setdefault("canvas_version", 0)
    st.session_state.setdefault("sample_offset", 0)

    revisions, unavailable = {}, {}

    for spec in MODEL_SPECS:
        try:
            revisions[spec.key] = model_revision(spec)
        except (OSError, ValueError) as error:
            unavailable[spec.key] = str(error)

    render_header(len(revisions))

    if not revisions:
        st.error(
            "Chưa tìm thấy mô hình hợp lệ trong thư mục models."
        )
        render_catalog(unavailable)
        return

    with st.container(border=True, key="model_toolbar"):
        selector, comparison = st.columns(
            [3, 2],
            vertical_alignment="bottom",
        )

        with selector:
            selected_key = st.selectbox(
                "Mô hình nhận dạng",
                options=list(revisions),
                format_func=lambda key: MODELS[key].name,
            )

        with comparison:
            compare = st.checkbox(
                "So sánh tất cả mô hình",
                value=False,
            )

        st.caption(MODELS[selected_key].description)

    left, right = st.columns([1.05, 1], gap="large")

    with left, st.container(
        border=True,
        key="input_panel",
    ):
        st.html(
            '<div class="eyebrow panel-label">'
            '01 / CHỮ SỐ CỦA BẠN</div>'
        )

        source = st.radio(
            "Cách nhập",
            options=["Vẽ tay", "Ảnh mẫu MNIST"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if source == "Vẽ tay":
            image = render_drawing()
        else:
            image = render_sample()

        processed, input_error = None, None

        if image is not None:
            try:
                processed = prepare_image(
                    image,
                    background="dark",
                    preserve_layout=source == "Ảnh mẫu MNIST",
                )
            except ValueError as error:
                input_error = str(error)

        keys = list(revisions) if compare else [selected_key]
        token = None

        if processed is not None:
            digest = hashlib.sha256(
                processed.tobytes()
            ).hexdigest()

            token = (
                digest,
                selected_key,
                source,
                tuple((key, revisions[key]) for key in keys),
            )

        previous = st.session_state.get("prediction")

        if token is None or (
            previous and previous["token"] != token
        ):
            st.session_state.pop("prediction", None)

        if st.button(
            "Nhận dạng chữ số →",
            type="primary",
            width="stretch",
        ):
            if processed is None:
                st.warning(
                    input_error
                    or "Hãy vẽ hoặc chọn một ảnh trước."
                )
            else:
                with st.spinner("Đang đọc nét viết của bạn…"):
                    prediction = run_predictions(
                        np.asarray(processed),
                        keys,
                    )

                st.session_state.prediction = {
                    "token": token,
                    **prediction,
                }

        if processed is not None:
            render_processed_image(processed)

    with right, st.container(
        border=True,
        key="result_panel",
    ):
        render_result(
            st.session_state.get("prediction"),
            selected_key,
        )

    st.html(
        '<div class="workflow">'
        '<div><span>01</span><strong>Một chữ số</strong>'
        '<p>Vẽ rõ nét hoặc chọn một ảnh mẫu MNIST.</p></div>'
        '<div><span>02</span><strong>Đúng định dạng</strong>'
        '<p>Căn giữa ảnh, chuẩn hóa về 28 × 28.</p></div>'
        '<div><span>03</span><strong>Nhiều góc nhìn</strong>'
        '<p>Đối chiếu dự đoán giữa các mô hình.</p></div>'
        '</div>'
    )

    render_catalog(unavailable)

    st.html(
        '<div class="footer">'
        '<strong>DIGIT RECOGNITION</strong>'
        '<span>Classify_Number · Đồ án Nhận dạng mẫu</span>'
        '<span>Built with Python & Streamlit</span>'
        '</div>'
    )


if __name__ == "__main__":
    main()