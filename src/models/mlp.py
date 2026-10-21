
import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e_z = np.exp(z)
    return e_z / np.sum(e_z, axis=1, keepdims=True)


def cross_entropy(logits, y):
    N = len(y)

    z = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(
        np.sum(np.exp(z), axis=1, keepdims=True)
    )
    log_probs = z - log_sum_exp

    return -np.mean(log_probs[np.arange(N), y])


# =========================
# 3. MLP
# =========================

class MLP:
    def __init__(
        self,
        input_size=784,
        hidden_size=128,
        output_size=10,
        seed=42
    ):
        rng = np.random.default_rng(seed)

        # Layer 1: input -> hidden
        self.W1 = rng.normal(
            0,
            np.sqrt(2.0 / input_size),
            size=(input_size, hidden_size)
        )
        self.b1 = np.zeros((1, hidden_size))

        # Layer 2: hidden -> output
        self.W2 = rng.normal(
            0,
            np.sqrt(2.0 / hidden_size),
            size=(hidden_size, output_size)
        )
        self.b2 = np.zeros((1, output_size))

    # =========================
    # 3.1 Forward propagation
    # =========================

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)

        Z2 = A1 @ self.W2 + self.b2
        cache = (X, Z1, A1, Z2)

        return Z2, cache

    # =========================
    # 3.2 Backpropagation
    # =========================

    def backward(self, cache, y):
        X, Z1, A1, Z2 = cache
        N = X.shape[0]

        # Gradient tại tầng đầu ra:
        # softmax + cross entropy
        P = softmax(Z2)

        dZ2 = P.copy()
        dZ2[np.arange(N), y] -= 1
        dZ2 /= N

        # Gradient của tầng output
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Truyền gradient về tầng ẩn
        dA1 = dZ2 @ self.W2.T

        # Đạo hàm ReLU
        dZ1 = dA1 * (Z1 > 0)

        # Gradient của tầng hidden
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def fit(self, 
        X, y,
        X_val=None, y_val=None,
        epochs=20,
        batch_size=128,
        learning_rate=0.1,
        seed=42
    ):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        rng = np.random.default_rng(seed)
        N = X.shape[0]

        self.history_ = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(epochs):
            # 1. Xáo trộn dữ liệu
            indices = rng.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # 2. Chia thành các mini-batch
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # 3. Forward propagation
                logits, cache = self.forward(X_batch)

                # 4. Backpropagation
                dW1, db1, dW2, db2 = self.backward(
                    cache, y_batch
                )

                # 5. Cập nhật trọng số bằng Gradient Descent
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2

            # 6. Đánh giá trên tập train sau mỗi epoch
            train_logits, _ = self.forward(X)
            train_loss = cross_entropy(train_logits, y)
            train_acc = np.mean(
                np.argmax(train_logits, axis=1) == y
            )

            self.history_["train_loss"].append(train_loss)
            self.history_["train_acc"].append(train_acc)

            message = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_acc:.2%}"
            )

            # 7. Đánh giá trên tập validation nếu có
            if X_val is not None and y_val is not None:
                X_val = np.asarray(X_val, dtype=np.float64)
                y_val = np.asarray(y_val, dtype=np.int64)

                val_logits, _ = self.forward(X_val)
                val_loss = cross_entropy(val_logits, y_val)
                val_acc = np.mean(
                    np.argmax(val_logits, axis=1) == y_val
                )

                self.history_["val_loss"].append(val_loss)
                self.history_["val_acc"].append(val_acc)

                message += (
                    f" | Val Loss: {val_loss:.4f}"
                    f" | Val Accuracy: {val_acc:.2%}"
                )

            print(message)

        return self

    # =========================
    # 3.3 Prediction
    # =========================

    def predict_proba(self, X):
        logits, _ = self.forward(X)
        return softmax(logits)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
