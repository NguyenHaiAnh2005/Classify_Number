import numpy as np


# Danh sach var_smoothing de thu khi tinh chinh (thang log)
DEFAULT_VAR_SMOOTHING_GRID = (1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


class GaussianNaiveBayes:

    def __init__(self, var_smoothing=1e-9, batch_size=2000):
        self.var_smoothing = var_smoothing
        self.batch_size = batch_size

    @staticmethod
    def gaussian_log_pdf(X, mean, var):
        """log N(X; mean, var), tinh tung phan tu (broadcast)."""
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * (X - mean) ** 2 / var

    # ------------------------------------------------------------------
    # Huan luyen
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X phai co shape (N, d)")
        if len(X) != len(y):
            raise ValueError("X va y phai co cung so mau")

        # y_idx: chi so lop (0..K-1) cua tung mau
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes, n_features = len(self.classes_), X.shape[1]

        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.prior_ = np.zeros(n_classes)

        # Hoc mean, var, prior cho tung lop
        for k in range(n_classes):
            X_k = X[y_idx == k]
            self.mean_[k] = X_k.mean(axis=0)
            self.var_[k] = X_k.var(axis=0)
            self.prior_[k] = X_k.shape[0] / X.shape[0]

        # Lam muot phuong sai (giong sklearn) de tranh chia cho 0
        self.var_ += self.var_smoothing * X.var(axis=0).max()
        return self

    # ------------------------------------------------------------------
    # Du doan
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not hasattr(self, "classes_"):
            raise RuntimeError("Model chua duoc fit. Hay goi fit(X, y) truoc.")

    def _joint_log_likelihood(self, X):
        """
        log P(c) + log P(x | c) cho moi (anh, lop). Shape (N, K).
        Xu ly theo tung lo de dinh bo nho chi phu thuoc batch_size.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), len(self.classes_)))

        for i in range(0, len(X), self.batch_size):
            Xb = X[i:i + self.batch_size]
            # Broadcast: (batch, 1, d) voi (K, d) -> (batch, K, d), cong theo pixel
            log_lik = self.gaussian_log_pdf(
                Xb[:, None, :], self.mean_, self.var_
            ).sum(axis=2)
            out[i:i + self.batch_size] = log_lik + np.log(self.prior_)

        return out

    def predict(self, X):
        """Nhan du doan (lop co log-xac-suat lon nhat). Shape (N,)."""
        jll = self._joint_log_likelihood(X)   # co kiem tra da fit ben trong
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        """log P(c | x), da chuan hoa bang logsumexp. Shape (N, K)."""
        jll = self._joint_log_likelihood(X)
        m = jll.max(axis=1, keepdims=True)   # tru max de on dinh so hoc
        log_norm = m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True))
        return jll - log_norm

    def predict_proba(self, X):
        """P(c | x). Shape (N, K), moi hang co tong bang 1."""
        return np.exp(self.predict_log_proba(X))

    def score(self, X, y):
        """Do chinh xac (accuracy)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


#GaussianNB = GaussNB

# ----------------------------------------------------------------------
# Giao dien chung cho cac file trong models/
# ----------------------------------------------------------------------
def build_model(var_smoothing=1e-9, batch_size=2000):
    """Tao model chua huan luyen."""
    return GaussianNaiveBayes(var_smoothing=var_smoothing, batch_size=batch_size)


def train(model, X_train, y_train):
    """Huan luyen model. Tra ve chinh model da fit."""
    return model.fit(X_train, y_train)


def predict(model, X):
    """Du doan nhan 0-9. Shape (N,)."""
    return model.predict(X)


def predict_proba(model, X):
    """Xac suat tung lop. Shape (N, 10)."""
    return model.predict_proba(X)


def evaluate(model, X, y):
    """Accuracy tren (X, y)."""
    return model.score(X, y)

