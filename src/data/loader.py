# File phụ trách 
class MNISTLoader:
    def __init__(
        self,
        data_dir="data/raw",
        validation_ratio=0.1,
        random_state=42,
    ):
        """
        Lưu đường dẫn dữ liệu, tỷ lệ validation và seed chia tập.
        Ví dụ: tách 60.000 ảnh thành 54.000 train + 6.000 validation.
        """

    def load_npz(self, file_path):
        """
        Đọc một file .npz có hai khóa X và y.
        Trả về X: (N, 28, 28), y: (N,).
        """

    def _validate_data(self, X, y):
        """
        Kiểm tra số ảnh khớp số nhãn, kích thước ảnh và nhãn hợp lệ.
        Báo lỗi khi dữ liệu không đúng định dạng MNIST.
        """

    def split_train_validation(self, X, y):
        """
        Chia dữ liệu thành train và validation theo tỷ lệ đã cấu hình.
        Chia phân tầng theo nhãn để giữ tỷ lệ các chữ số gần tương đương.
        Trả về:
        {
            "train": (X_train, y_train),
            "validation": (X_val, y_val),
        }
        """

    def load(self):
        """
        Đọc mnist_train.npz và mnist_test.npz.
        Chỉ chia validation từ tập train gốc; giữ riêng tập test.
        Trả về dictionary có ba khóa train, validation và test.
        """