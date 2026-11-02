# Classify_Number

Dự án nhận dạng chữ số viết tay từ **0 đến 9** trên tập dữ liệu MNIST, nhằm mục đích so sánh dộ hiệu quả giữa các mô hình học máy.

Giao diện Streamlit cho phép vẽ chữ số, chọn ảnh mẫu và so sánh kết quả từ các mô hình đã huấn luyện.

## Yêu cầu

- **Python 3.14** — môi trường phát triển sử dụng **3.14.7**.
- [Git](https://git-scm.com/downloads).
- [Git LFS](https://git-lfs.com/) để tải các file mô hình.
- Các thư viện trong `requirements.txt`.

## Cài đặt trên Windows

Cài [Python 3.14](https://www.python.org/downloads/windows/), Git và Git LFS.

Mở PowerShell và chạy:

```powershell
git lfs install
git clone https://github.com/NguyenHaiAnh2005/Classify_Number.git
cd Classify_Number
git lfs pull

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m streamlit run app/main.py
```

Nếu PowerShell không cho kích hoạt môi trường, có thể dùng Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Sau đó tiếp tục các lệnh cài thư viện và chạy ứng dụng.

## Cài đặt trên Linux

Cài Python 3.14 theo bản phân phối Linux đang sử dụng.

Với Ubuntu/Debian, cài Git và Git LFS:

```bash
sudo apt update
sudo apt install -y git git-lfs
```

Tải dự án và cài môi trường:

```bash
git lfs install
git clone https://github.com/NguyenHaiAnh2005/Classify_Number.git
cd Classify_Number
git lfs pull

python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m streamlit run app/main.py
```

## Sử dụng

Sau khi chạy, mở địa chỉ **Local URL** được hiển thị trong terminal, thường là:

```text
http://localhost:8501
```

1. Chọn mô hình nhận dạng.
2. Vẽ một chữ số hoặc chọn ảnh mẫu MNIST.
3. Nhấn **Nhận dạng chữ số**.
4. Bật **So sánh tất cả mô hình** nếu muốn đối chiếu kết quả.

Nhấn `Ctrl + C` trong terminal để dừng ứng dụng.

**Lưu ý:** Giữ nguyên các thư mục `models/`, `src/` và `data/`. Chạy lệnh Streamlit tại thư mục gốc `Classify_Number`. Không cần huấn luyện lại khi đã có đầy đủ các file mô hình.
