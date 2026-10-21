import numpy as np


def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

def softmax(z):
    z = z - np.max(z, axis = 1, keepdims=True)
    e_z = np.exp(z)
    return e_z / np.sum(e_z, axis = 1, keepdims=True)

def cross_entropy(z, y):
    z_shifted = z - np.max(z, axis=1, keepdims=True)

    log_sum_exp = np.log(
        np.sum(np.exp(z_shifted), axis=1, keepdims=True)
    )

    log_probs = z_shifted - log_sum_exp
    return -np.mean(log_probs[np.arange(len(y)), y])


class LogisticRegression:
    def __init__(self, eta=0.1, lam=1e-3, max_epochs=100, batch_size=128, seed=42, tol=1e-7):
        self.eta = eta
        self.lam = lam
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.tol = tol

    def fit(self, X, y, X_val=None, y_val=None):
        Xb = add_bias(X)
        y = np.asarray(y, dtype=int)

        N, d = Xb.shape
        n_classes = 10

        rng = np.random.default_rng(self.seed)
        self.W_ = np.zeros((d, n_classes))

        mask = np.ones((d, n_classes))
        mask[-1, :] = 0.0

        self.history_ = {
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(self.max_epochs):
            indices = rng.permutation(N)

            for start in range(0, N, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                X_batch = Xb[batch_idx]
                y_batch = y[batch_idx]

                Y_batch = np.zeros((len(y_batch), n_classes))
                Y_batch[np.arange(len(y_batch)), y_batch] = 1.0

                P = softmax(X_batch @ self.W_)

                grad = X_batch.T @ (P - Y_batch) / len(y_batch)
                grad += self.lam * mask * self.W_

                self.W_ -= self.eta * grad

        # Loss sau khi cập nhật hết các mini-batch
            train_logits = Xb @ self.W_
            train_loss = cross_entropy(train_logits, y)
            train_loss += 0.5 * self.lam * np.sum((mask * self.W_) ** 2)

            self.history_["train_loss"].append(train_loss)

            if X_val is not None and y_val is not None:
                val_logits = self.decision_function(X_val)
                val_loss = cross_entropy(val_logits, y_val)
                val_loss += 0.5 * self.lam * np.sum(
                    (mask * self.W_) ** 2
                )
                self.history_["val_loss"].append(val_loss)

            if epoch > 0:
                previous = self.history_["train_loss"][-2]
                current = self.history_["train_loss"][-1]

                if abs(previous - current) < self.tol:
                    break

        return self

    def decision_function(self, X):
        return add_bias(X) @ self.W_

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)





