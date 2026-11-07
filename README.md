# Classify_Number

Dự án nhận dạng chữ số viết tay từ **0 đến 9** trên tập dữ liệu MNIST, nhằm so sánh hiệu quả giữa các mô hình học máy.

Giao diện Streamlit cho phép vẽ chữ số, chọn ảnh mẫu và so sánh kết quả từ các mô hình đã huấn luyện.

## Yêu cầu

- **Python 3.14** — môi trường phát triển sử dụng **3.14.7**.
- [Git](https://git-scm.com/downloads).
- [Git LFS](https://git-lfs.com/) để tải các file mô hình.
- Các thư viện trong `requirements.txt`.

> Có thể vào trang web đã được public với link: https://classifynumber.streamlit.app/
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

# Để run app chạy lệnh này 
python -m streamlit run app/main.py
```

Nếu PowerShell không cho kích hoạt môi trường, mở Command Prompt tại thư mục `Classify_Number` và chạy:

```cmd
.venv\Scripts\activate.bat
```

Sau đó tiếp tục các lệnh cài thư viện và chạy ứng dụng:

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m streamlit run app/main.py
```

## Cài đặt trên macOS

Cài [Homebrew](https://brew.sh/) theo hướng dẫn trên trang chủ. Sau khi cài, thực hiện các bước cấu hình PATH được trình cài đặt hiển thị trong phần **Next steps**.

Mở Terminal và cài Python 3.14, Git và Git LFS:

```bash
brew install python@3.14 git git-lfs
```

Kiểm tra phiên bản Python:

```bash
python3.14 --version
```

Tải dự án và tạo môi trường ảo:

```bash
git lfs install
git clone https://github.com/NguyenHaiAnh2005/Classify_Number.git
cd Classify_Number
git lfs pull

python3.14 -m venv .venv
source .venv/bin/activate
```

Cài thư viện và chạy ứng dụng:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m streamlit run app/main.py
```

**Lưu ý:** Homebrew cài bản cập nhật hiện có của Python 3.14, không nhất thiết đúng phiên bản 3.14.7.

Nếu Terminal báo `brew: command not found`, kiểm tra lại bước cấu hình PATH trong phần **Next steps** khi cài Homebrew, sau đó mở lại Terminal.

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