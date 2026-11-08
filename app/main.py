"""Ứng dụng MNIST với giao diện tối giản và điểm nhấn xanh ngọc."""

import hashlib
import io
import logging
import sys
from html import escape
from pathlib import Path
from time import perf_counter

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import (
    MODEL_SPECS,
    MODELS,
    load_model,
    model_revision,
    predict_probabilities,
    prepare_image,
)

LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False, max_entries=16)
def get_model(key: str, revision: tuple) -> object:
    return load_model(key)


@st.cache_data(show_spinner=False)
def load_samples() -> tuple[np.ndarray, np.ndarray]:
    with np.load(ROOT / "data/raw/mnist_test.npz") as data:
        return data["X"], data["y"]


def clear_drawing() -> None:
    st.session_state.canvas_version += 1
    st.session_state.pop("prediction", None)


def next_sample() -> None:
    st.session_state.sample_offset += 1
    st.session_state.pop("prediction", None)


def render_input(source: str) -> tuple[Image.Image | None, str]:
    if source == "Vẽ tay":
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=st.session_state.stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=256,
            height=256,
            drawing_mode="freedraw",
            return_image_data=True,
            key=f"canvas_{st.session_state.canvas_version}",
        )

        controls, clear = st.columns(
            [2, 1],
            vertical_alignment="bottom",
        )

        with controls:
            st.slider(
                "Độ dày nét",
                8,
                32,
                step=2,
                key="stroke_width",
            )

        with clear:
            st.button(
                "Xóa nét",
                on_click=clear_drawing,
                width="stretch",
            )

        st.caption(
            "Vẽ một chữ số ở giữa khung, nét rõ và đủ lớn."
        )

        image = (
            None
            if canvas.image_data is None
            else Image.fromarray(canvas.image_data.astype(np.uint8))
        )
        return image, "dark"

    if source == "Ảnh MNIST":
        try:
            images, labels = load_samples()
        except (OSError, KeyError, ValueError):
            st.info(
                "Chưa có dữ liệu mẫu. Bạn vẫn có thể vẽ hoặc tải ảnh."
            )
            return None, "dark"

        digit = st.select_slider(
            "Chữ số mẫu",
            list(range(10)),
            value=7,
        )
        indices = np.flatnonzero(labels == digit)

        if not len(indices):
            st.warning("Không có ảnh mẫu cho chữ số này.")
            return None, "dark"

        index = indices[
            st.session_state.sample_offset % len(indices)
        ]
        image = Image.fromarray(images[index].astype(np.uint8))

        st.image(
            image,
            width=256,
            caption=f"MNIST · nhãn thật: {digit}",
        )
        st.button(
            "Đổi ảnh mẫu",
            on_click=next_sample,
            width="stretch",
        )

        return image, "dark"

    uploaded = st.file_uploader(
        "Chọn ảnh chữ số",
        type=["png", "jpg", "jpeg", "webp"],
    )
    background = st.selectbox(
        "Nền ảnh",
        ["Tự động", "Nền sáng", "Nền tối"],
    )

    if uploaded is None:
        st.caption(
            "Dùng ảnh chứa một chữ số, nền đơn giản và tương phản rõ."
        )
        return None, "auto"

    try:
        with Image.open(uploaded) as opened:
            image = opened.copy()

        st.image(image, width=256)

        return image, {
            "Tự động": "auto",
            "Nền sáng": "light",
            "Nền tối": "dark",
        }[background]

    except (OSError, ValueError, Image.DecompressionBombError):
        st.warning(
            "Không đọc được ảnh. Hãy thử một tệp PNG hoặc JPG khác."
        )
        return None, "auto"


def run_predictions(
    pixels: np.ndarray,
    keys: list[str],
    revisions: dict,
) -> dict:
    results, errors = {}, {}

    for key in keys:
        try:
            model = get_model(key, revisions[key])

            start = perf_counter()
            probabilities = predict_probabilities(model, pixels)

            results[key] = {
                "probabilities": probabilities,
                "digit": int(probabilities.argmax()),
                "confidence": float(probabilities.max()),
                "elapsed_ms": (perf_counter() - start) * 1000,
            }

        except Exception as error:
            LOGGER.exception("Không chạy được mô hình %s", key)
            errors[key] = str(error)

    return {"results": results, "errors": errors}


def render_result(
    prediction: dict | None,
    selected: str,
) -> None:
    st.html(
        '<div class="section-label">'
        '02 / KẾT QUẢ NHẬN DẠNG</div>'
    )

    if prediction is None:
        st.html("""
            <div class="empty-result">
                <div class="empty-ring">?</div>
                <h3>Chờ một nét vẽ.</h3>
                <p>
                    Thêm ảnh đầu vào và nhấn nhận dạng<br>
                    để xem kết quả tại đây.
                </p>
                <span class="pill muted-pill">
                    0 — 9 · Handwritten digits
                </span>
            </div>
        """)
        return

    result = prediction["results"].get(selected)

    if result is None:
        st.error(f"Không chạy được {MODELS[selected].name}.")

        with st.expander("Chi tiết lỗi"):
            st.code(
                prediction["errors"].get(
                    selected,
                    "Chưa có kết quả.",
                ),
                language=None,
            )
        return

    confidence = result["confidence"] * 100

    st.html(f"""
        <div class="result-summary">
            <div>
                <div class="overline">CHỮ SỐ DỰ ĐOÁN</div>
                <div class="predicted-digit">{result['digit']}</div>
            </div>
            <div
                class="confidence-ring"
                style="--angle:{confidence * 3.6:.3f}deg"
            >
                <div>
                    <strong>{confidence:.1f}<small>%</small></strong>
                    <span>XÁC SUẤT</span>
                </div>
            </div>
        </div>
        <div class="result-meta">
            <span>{escape(MODELS[selected].name)}</span>
            <span>{result['elapsed_ms']:.1f} ms</span>
        </div>
    """)

    st.html(
        '<div class="section-label chart-label">'
        'PHÂN PHỐI XÁC SUẤT</div>'
    )

    bars = []

    for digit, probability in enumerate(result["probabilities"]):
        percent = float(probability) * 100
        active = " active" if digit == result["digit"] else ""

        bars.append(f"""
            <div class="probability-row{active}">
                <span>{digit}</span>
                <div
                    class="bar-track"
                    role="meter"
                    aria-label="Xác suất chữ số {digit}"
                    aria-valuemin="0"
                    aria-valuemax="100"
                    aria-valuenow="{percent:.2f}"
                >
                    <i style="width:{percent:.4f}%"></i>
                </div>
                <b>{percent:.1f}%</b>
            </div>
        """)

    st.html(
        '<div class="probability-chart">'
        + "".join(bars)
        + "</div>"
    )
    st.caption(
        "Xác suất của dự đoán này, "
        "không phải độ chính xác trên tập kiểm thử."
    )


def render_comparison(
    prediction: dict,
    selected: str,
) -> None:
    reference = prediction["results"].get(selected, {}).get("digit")

    with st.container(border=True, key="comparison"):
        st.html(
            '<div class="section-label">'
            '03 / ĐỐI CHIẾU MÔ HÌNH</div>'
        )

        rows = []

        for spec in MODEL_SPECS:
            key = spec.key
            item = prediction["results"].get(key)

            if item is None and key not in prediction["errors"]:
                continue

            if item is None:
                rows.append(f"""
                    <div class="model-row">
                        <span class="status-icon error">×</span>
                        <div>
                            <strong>{escape(spec.name)}</strong>
                            <small>Không chạy được</small>
                        </div>
                        <span class="row-value">—</span>
                        <span></span>
                    </div>
                """)
                continue

            agree = (
                reference is not None
                and item["digit"] == reference
            )
            state = (
                "agree"
                if agree
                else "different"
                if reference is not None
                else "neutral"
            )
            symbol = (
                "✓"
                if agree
                else "≠"
                if reference is not None
                else "•"
            )
            active = " selected" if key == selected else ""
            percent = item["confidence"] * 100

            rows.append(f"""
                <div class="model-row{active}">
                    <span class="status-icon {state}">{symbol}</span>
                    <div>
                        <strong>{escape(spec.name)}</strong>
                        <small>
                            Dự đoán {item['digit']}
                            · {item['elapsed_ms']:.1f} ms
                        </small>
                    </div>
                    <div class="bar-track">
                        <i style="width:{percent:.4f}%"></i>
                    </div>
                    <span class="row-value">{percent:.1f}%</span>
                </div>
            """)

        st.html(
            '<div class="model-list">'
            + "".join(rows)
            + "</div>"
        )
        st.caption(
            "✓ Cùng dự đoán với mô hình đang chọn · "
            "≠ Khác dự đoán. Không phải đánh giá đúng/sai."
        )

        if prediction["errors"]:
            with st.expander("Chi tiết các mô hình gặp lỗi"):
                for key, error in prediction["errors"].items():
                    st.markdown(f"**{MODELS[key].name}**")
                    st.code(error, language=None)


def render_catalog(unavailable: dict) -> None:
    with st.expander("Kho mô hình & trọng số"):
        for spec in MODEL_SPECS:
            status = (
                "Thiếu trọng số"
                if spec.key in unavailable
                else "Có trọng số"
            )
            st.markdown(f"**{spec.name}** · {status}")
            st.caption(spec.description)
            st.code(
                "\n".join(
                    f"models/{name}" for name in spec.files
                ),
                language=None,
            )

            if spec.key in unavailable:
                st.warning(unavailable[spec.key])


def main() -> None:
    st.set_page_config(
        page_title="Digit Studio",
        page_icon="✳",
        layout="wide",
    )

    css = (ROOT / "app/styles.css").read_text(encoding="utf-8")
    st.html("<style>" + css + "</style>")

    for key, default in {
        "canvas_version": 0,
        "sample_offset": 0,
        "stroke_width": 18,
    }.items():
        st.session_state.setdefault(key, default)

    revisions, unavailable = {}, {}

    for spec in MODEL_SPECS:
        try:
            revisions[spec.key] = model_revision(spec)
        except (OSError, ValueError) as error:
            unavailable[spec.key] = str(error)

    st.html(f"""
        <div class="topbar">
            <div class="brand">
                <span class="brand-mark">✳</span>
                digit<span>studio</span>
            </div>
            <span class="pill">
                <i></i>
                {len(revisions)}/{len(MODEL_SPECS)} bộ trọng số
            </span>
        </div>
        <div class="intro">
            <div class="overline">
                HANDWRITTEN DIGIT RECOGNITION
            </div>
            <h1>Một nét vẽ. <span>Một con số.</span></h1>
            <p>
                Vẽ, thử ảnh và khám phá cách các mô hình
                nhận dạng chữ số.
            </p>
        </div>
    """)

    if not revisions:
        st.error(
            "Chưa có trọng số hợp lệ. "
            "Kiểm tra thư mục models hoặc chạy git lfs pull."
        )
        render_catalog(unavailable)
        return

    with st.container(border=True, key="workspace"):
        selector, comparison = st.columns(
            [2, 1],
            vertical_alignment="bottom",
        )
        
        with selector:
            model_options = list(revisions)

            # Đưa CNN Rotated lên đầu nếu có trọng số hợp lệ.
            if "cnn_rotated" in model_options:
                model_options.remove("cnn_rotated")
                model_options.insert(0, "cnn_rotated")

            selected = st.selectbox(
                "Mô hình nhận dạng",
                model_options,
                index=0,
                format_func=lambda key: MODELS[key].name,
            )

        with comparison:
            compare = st.toggle("So sánh tất cả", value=True)

        st.caption(MODELS[selected].description)
        st.divider()

        left, right = st.columns([1, 1], gap="large")

        with left, st.container(key="input_panel"):
            st.html(
                '<div class="section-label">'
                '01 / ẢNH ĐẦU VÀO</div>'
            )

            source = st.radio(
                "Cách nhập",
                ["Vẽ tay", "Ảnh MNIST", "Tải ảnh"],
                horizontal=True,
                label_visibility="collapsed",
            )

            image, background = render_input(source)
            processed, input_error = None, None

            with st.expander("Ảnh xử lý & góc xoay"):
                angle = st.slider(
                    "Góc xoay thử nghiệm",
                    -30,
                    30,
                    0,
                    5,
                    format="%d°",
                    help=(
                        "Áp dụng cùng góc xoay cho mọi mô hình. "
                        "Góc dương xoay ngược chiều kim đồng hồ."
                    ),
                )

                if image is not None:
                    try:
                        processed = prepare_image(
                            image,
                            background,
                            source == "Ảnh MNIST",
                        )

                        if angle:
                            processed = processed.rotate(
                                angle,
                                resample=Image.Resampling.BILINEAR,
                                fillcolor=0,
                            )

                        st.image(
                            processed,
                            width=112,
                            caption=(
                                "Ảnh đưa vào mọi mô hình · 28 × 28"
                            ),
                        )

                        buffer = io.BytesIO()
                        processed.save(buffer, format="PNG")

                        st.download_button(
                            "Tải ảnh đã xử lý",
                            buffer.getvalue(),
                            file_name="digit_28x28.png",
                            mime="image/png",
                        )

                    except ValueError as error:
                        input_error = str(error)

                else:
                    st.caption(
                        "Thêm ảnh để xem bản xử lý và thử CNN Rotated."
                    )

            keys = list(revisions) if compare else [selected]

            token = (
                None
                if processed is None
                else (
                    source,
                    angle,
                    hashlib.sha256(processed.tobytes()).hexdigest(),
                    tuple((key, revisions[key]) for key in keys),
                )
            )

            previous = st.session_state.get("prediction")

            if token is None or (
                previous and previous["token"] != token
            ):
                st.session_state.pop("prediction", None)

            if input_error:
                st.caption(input_error)

            if st.button(
                "Nhận dạng chữ số →",
                type="primary",
                width="stretch",
                disabled=processed is None,
            ):
                with st.spinner("Đang nhận dạng…"):
                    st.session_state.prediction = {
                        "token": token,
                        **run_predictions(
                            np.asarray(processed),
                            keys,
                            revisions,
                        ),
                    }

        with right, st.container(key="result_panel"):
            render_result(
                st.session_state.get("prediction"),
                selected,
            )

    prediction = st.session_state.get("prediction")

    if prediction and compare:
        render_comparison(prediction, selected)

    render_catalog(unavailable)

    st.html(
        '<div class="footer">'
        '<span>Digit Studio · Classify_Number</span>'
        '<span>MNIST · 0–9</span>'
        "</div>"
    )


if __name__ == "__main__":
    main()