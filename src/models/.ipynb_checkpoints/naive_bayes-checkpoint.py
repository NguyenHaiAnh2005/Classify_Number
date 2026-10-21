import numpy as np

class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        pass
    def fit(self, X, y):
        """
        Xác định các lớp và tính xác suất tiên nghiệm mỗi lớp.
        Tính trung bình, phương sai từng đặc trưng theo từng lớp.
        Trả về self.
        """

    def _joint_log_likelihood(self, X):
        """
        Tính log P(y=k) + tổng log P(x_j | y=k) theo phân phối Gaussian.
        Giả thiết các đặc trưng độc lập có điều kiện theo lớp.
        Trả về ma trận điểm log có dạng (N, K).
        """

    def predict_log_proba(self, X):
        """
        Chuẩn hóa các điểm log thành log xác suất hậu nghiệm.
        Dùng phép log-sum-exp để hạn chế lỗi số học.
        """

    def predict(self, X):
        """Chọn lớp có điểm log cao nhất và trả về nhãn gốc."""