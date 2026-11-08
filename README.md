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


## Code Structure

```text
Classify_Number/
├── src/                                  # Mã nguồn xử lý dữ liệu, mô hình và đánh giá
│   ├── models/
│   │   ├── naive_bayes.py                # Gaussian Naive Bayes tự cài đặt bằng NumPy
│   │   ├── logistic_regression.py        # Logistic Regression đa lớp, Gradient Descent và L2
│   │   ├── mlp.py                        # MLP với backpropagation và momentum tự cài đặt
│   │   ├── cnn.py                        # Kiến trúc CNN và bộ huấn luyện bằng PyTorch
│   │   └── test_model.ipynb              # Kiểm tra các thành phần của MLP
│   ├── preprocessing/
│   │   ├── image_preprocessing.py        # Loại ảnh trống, ảnh trùng và tăng cường bằng phép dịch
│   │   └── scaler.py                     # Chia train/validation, chuẩn hóa và định dạng đầu vào
│   └── evaluation/
│       ├── metrics.py                    # Confusion matrix, Accuracy, Precision, Recall và F1
│       └── test_metrics.ipynb            # Kiểm tra các công thức đánh giá
│
├── app/                                  # Ứng dụng nhận dạng chữ số trên Streamlit
│   ├── main.py                           # Giao diện vẽ tay, chọn ảnh và so sánh mô hình
│   ├── inference.py                      # Nạp trọng số, xử lý ảnh và tính xác suất dự đoán
│   └── styles.css                        # Kiểu hiển thị, màu sắc và bố cục giao diện
│
├── notebooks/                            # Notebook khám phá dữ liệu và thực nghiệm
│   ├── 01_eda.ipynb                      # Khám phá MNIST, kiểm tra dữ liệu và phân bố nhãn
│   ├── experiment_GNB.ipynb              # Huấn luyện GNB và khảo sát var_smoothing
│   ├── experiments_LR.ipynb              # Khảo sát learning rate và regularization L2 của LR
│   ├── experiment_mlp.ipynb              # Khảo sát kiến trúc, optimizer và regularization MLP
│   ├── ensemble_other_train_save.ipynb   # Xây dựng, đánh giá và lưu Soft Voting MLP + RF + KNN
│   ├── train_cnn_rotation.ipynb          # Huấn luyện CNN gốc và CNN tăng cường ảnh xoay
│   ├── compare_model.ipynb               # So sánh các mô hình, độ đo và thời gian thực hiện
│   ├── report_best_models_vs_sklearn.ipynb
│   │                                     # Tổng hợp kết quả và đối chiếu bổ sung với scikit-learn
│   └── data/MNIST/raw/                   # Dữ liệu MNIST tải từ đường dẫn tương đối của notebook
│
├── data/                                 # Dữ liệu dùng cho huấn luyện và kiểm thử
│   ├── MNIST/raw/                        # Dữ liệu MNIST định dạng IDX và tệp nén
│   └── raw/
│       ├── MNIST/raw/                    # Dữ liệu MNIST tải tại đường dẫn data/raw
│       ├── download_mnist.py             # Tải và chuẩn bị dữ liệu MNIST
│       ├── mnist_train.npz               # 60.000 ảnh huấn luyện chính thức và nhãn
│       └── mnist_test.npz                # 10.000 ảnh kiểm thử chính thức và nhãn
│
├── models/                               # Mô hình và trọng số đã huấn luyện
│   ├── logistic_regression.joblib        # Mô hình Logistic Regression
│   ├── naive_bayes.joblib                # Mô hình Gaussian Naive Bayes
│   ├── mlp_model.pth                     # Trọng số, cấu hình và lịch sử huấn luyện MLP
│   ├── mnist_cnn.pth                     # Trọng số CNN huấn luyện trên ảnh MNIST gốc
│   ├── mnist_cnn_rotated.pth             # Trọng số CNN huấn luyện với ảnh xoay ngẫu nhiên
│   └── ensemble_v1/
│       ├── config.json                  # Trọng số voting, thứ tự mô hình và danh sách nhãn
│       ├── mlp_part.pth                 # MLP thành viên của Ensemble
│       ├── rf_part.joblib               # Random Forest thành viên, sử dụng scikit-learn
│       └── knn_part.joblib              # KNN thành viên, sử dụng scikit-learn
│
├── .streamlit/
│   └── config.toml                       # Cấu hình theme và hiển thị của Streamlit
├── .gitattributes                        # Cấu hình Git LFS cho các tệp mô hình
├── .gitignore                            # Các tệp và thư mục được loại khỏi Git
├── requirements.txt                      # Thư viện và phiên bản dùng trong dự án
└── README.md                             # Giới thiệu, cấu trúc và hướng dẫn chạy dự án
```