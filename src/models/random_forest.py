import numpy as np

class TreeNode:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        probabilities=None,
    ):
        """
        Nút trong lưu đặc trưng, ngưỡng chia và hai nút con.
        Nút lá lưu phân phối xác suất của các lớp.
        """

    def is_leaf(self):
        """Cho biết nút hiện tại có phải nút lá hay không."""


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        seed=42,
    ):
        """Lưu điều kiện dừng cây và số đặc trưng xét ngẫu nhiên mỗi nút."""

    def _gini(self, y_indices):
        """Tính độ hỗn tạp Gini từ tỷ lệ các lớp trong một nút."""

    def _find_best_split(self, X, y_indices):
        """
        Chọn ngẫu nhiên tập đặc trưng ứng viên tại nút hiện tại.
        Tìm đặc trưng và ngưỡng làm giảm Gini có trọng số nhiều nhất.
        Tuân thủ min_samples_leaf; trả None nếu không có phép chia hợp lệ.
        """

    def _build_tree(self, X, y_indices, depth):
        """
        Xây cây đệ quy.
        Dừng khi nút thuần nhất, đạt max_depth, thiếu mẫu
        hoặc không tìm được phép chia cải thiện Gini.
        Trả về TreeNode.
        """

    def fit(self, X, y, classes=None):
        """
        Xây cây và lưu root_, classes_.
        classes cho phép nhận thứ tự lớp chung từ Random Forest.
        Trả về self.
        """

    def _predict_one(self, x, node):
        """Duyệt từ nút hiện tại đến lá và trả phân phối xác suất tại lá."""

    def predict_proba(self, X):
        """Trả xác suất tại nút lá ứng với từng mẫu."""

    def predict(self, X):
        """Chọn nhãn có xác suất lớn nhất tại lá."""


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        seed=42,
    ):
        """Lưu số cây và cấu hình chung dùng để tạo các cây."""
        ...

    def _bootstrap_indices(self, n_samples):
        """
        Lấy n_samples chỉ số có hoàn lại.
        Dùng để tạo tập bootstrap riêng cho mỗi cây.
        """

    def fit(self, X, y):
        """
        Xác định classes_ chung, tạo các mẫu bootstrap.
        Huấn luyện các cây với seed riêng và lưu vào estimators_.
        Truyền classes_ chung cho mỗi cây. Trả về self.
        """

    def predict_proba(self, X):
        """Lấy trung bình xác suất dự đoán từ tất cả các cây."""

    def predict(self, X):
        """Chọn lớp có xác suất trung bình lớn nhất."""
