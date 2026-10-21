import copy
import time
import torch
import torch.nn as nn

# Tắt autograd: mọi gradient đều do backward tự cài đặt tính ra.
torch.set_grad_enabled(False)

SEED = 42
ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}

# =========================================================
# 1. XÂY DỰNG MẠNG
# =========================================================

def build_mlp(sizes, activation="relu", p_drop=0.0, init="he", const=0.01, seed=SEED):
    """sizes = [784, k_1, ..., 10]. Linear -> Act -> (Dropout) -> ... -> Linear."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation phải thuộc {list(ACTIVATIONS)}")
    if not 0 <= p_drop < 1:
        raise ValueError("p_drop phải thuộc [0, 1).")

    torch.manual_seed(seed)
    layers = []
    L = len(sizes) - 1
    for l in range(L):
        lin = nn.Linear(sizes[l], sizes[l + 1])
        if init == "he":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        elif init == "xavier":
            nn.init.xavier_uniform_(lin.weight)
        elif init == "const":
            nn.init.constant_(lin.weight, const)
        else:
            raise ValueError(init)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if l < L - 1:
            layers.append(ACTIVATIONS[activation]())
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))
    return nn.Sequential(*layers)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# =========================================================
# 2. FORWARD CÓ CACHE
# =========================================================

def forward_with_cache(model, X, train=True):
    """Trả về (output, cache). cache[i] = (layer, a_in, a_out, extra)."""
    cache = []
    a = X
    for layer in model:
        a_in = a
        extra = None
        if isinstance(layer, nn.Dropout):
            if train and layer.p > 0:
                M = (torch.rand_like(a_in) > layer.p).to(a_in.dtype)   # P(M=1) = 1-p
                a = a_in * M / (1 - layer.p)                           # inverted dropout
                extra = M
            else:
                a = a_in
        else:
            a = layer(a_in)
        cache.append((layer, a_in, a, extra))
    return a, cache


# =========================================================
# 3. LOSS
# =========================================================

def softmax_cross_entropy(logits, y):
    """Trả về (loss trung bình, delta = dJ/dlogits)."""
    N = logits.shape[0]
    z = logits - logits.max(dim=1, keepdim=True).values            # trừ max tránh tràn số
    log_probs = z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))
    loss = -log_probs[torch.arange(N), y].mean()
    Y = torch.zeros_like(logits)
    Y[torch.arange(N), y] = 1.0
    delta = (torch.exp(log_probs) - Y) / N                         # BP1
    return loss, delta


# =========================================================
# 4. BACKWARD
# =========================================================

def linear_backward(layer, a_in, delta):
    dW = delta.T @ a_in                  # (k_l, k_{l-1})
    db = delta.sum(dim=0)                # (k_l,)
    grad_in = delta @ layer.weight       # (N, k_{l-1})
    return dW, db, grad_in


def relu_backward(a_in, grad):
    return grad * (a_in > 0).to(grad.dtype)


def sigmoid_backward(a_out, grad):
    return grad * a_out * (1.0 - a_out)


def tanh_backward(a_out, grad):
    return grad * (1.0 - a_out ** 2)


def dropout_backward(mask, p, grad):
    if mask is None:
        return grad
    return grad * mask / (1 - p)


def backward(cache, grad_out, weight_decay=0.0):
    """Gán .grad cho mọi tham số nn.Linear; trả về dJ/dX."""
    grad = grad_out
    for layer, a_in, a_out, extra in reversed(cache):
        if isinstance(layer, nn.Linear):
            dW, db, grad = linear_backward(layer, a_in, grad)
            layer.weight.grad = dW + weight_decay * layer.weight   # L2 chỉ cho W
            layer.bias.grad = db
        elif isinstance(layer, nn.ReLU):
            grad = relu_backward(a_in, grad)
        elif isinstance(layer, nn.Sigmoid):
            grad = sigmoid_backward(a_out, grad)
        elif isinstance(layer, nn.Tanh):
            grad = tanh_backward(a_out, grad)
        elif isinstance(layer, nn.Dropout):
            grad = dropout_backward(extra, layer.p, grad)
    return grad


# =========================================================
# 5. OPTIMIZER
# =========================================================

class SGDMomentum:
    """Delta_w = alpha * Delta_w - gamma * g ;  w += Delta_w."""

    def __init__(self, params, lr, momentum=0.0):
        self.params = list(params)
        self.lr, self.momentum = lr, momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            v.mul_(self.momentum).sub_(self.lr * p.grad)
            p.add_(v)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# =========================================================
# 6. LỚP MLP
# =========================================================

class MLP:
    def __init__(self, sizes, activation="relu", p_drop=0.0, init="he",
                 const=0.01, seed=SEED):
        self.sizes = list(sizes)
        self.model = build_mlp(sizes, activation, p_drop, init, const, seed)
        self.history_ = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ---------- tiện ích ----------
    def parameters(self):
        return self.model.parameters()

    def n_params(self):
        return n_params(self.model)

    def forward_with_cache(self, X, train=True):
        return forward_with_cache(self.model, X, train)

    def backward(self, cache, grad_out, weight_decay=0.0):
        return backward(cache, grad_out, weight_decay)

    # ---------- huấn luyện / đánh giá ----------
    def train_one_epoch(self, X, y, optimizer, batch_size=128, weight_decay=0.0):
        N = X.shape[0]
        perm = torch.randperm(N)
        total, count = 0.0, 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits, cache = self.forward_with_cache(X[idx], train=True)
            loss, delta = softmax_cross_entropy(logits, y[idx])
            self.backward(cache, delta, weight_decay)
            optimizer.step()
            total += loss.item() * len(idx)
            count += len(idx)
        return total / count

    def evaluate(self, X, y, batch_size=5000):
        """(cross-entropy trung bình, accuracy), dropout tắt."""
        total, correct = 0.0, 0
        for s in range(0, len(X), batch_size):
            logits, _ = self.forward_with_cache(X[s:s + batch_size], train=False)
            loss, _ = softmax_cross_entropy(logits, y[s:s + batch_size])
            total += loss.item() * len(logits)
            correct += (logits.argmax(dim=1) == y[s:s + batch_size]).sum().item()
        return total / len(X), correct / len(X)

    def _check_data(self, X, y):
        X, y = X.float(), y.long()
        if X.ndim != 2 or X.shape[1] != self.sizes[0]:
            raise ValueError("Kích thước X không khớp với input layer.")
        if len(X) != len(y) or len(y) == 0:
            raise ValueError("X và y phải cùng số mẫu và không rỗng.")
        if y.min() < 0 or y.max() >= self.sizes[-1]:
            raise ValueError("Nhãn y nằm ngoài phạm vi output layer.")
        return X, y

    def fit(self, X, y, epochs=10, batch_size=128, lr=0.05, momentum=0.9,
            weight_decay=0.0, validation_data=None, patience=None,
            seed=SEED, verbose=True):

        if batch_size <= 0 or epochs <= 0 or lr <= 0:
            raise ValueError("batch_size, epochs và lr phải dương.")
        if patience is not None and validation_data is None:
            raise ValueError("Early stopping cần validation_data.")

        X, y = self._check_data(X, y)
        if validation_data is not None:
            X_val, y_val = self._check_data(*validation_data)

        torch.manual_seed(seed)
        opt = SGDMomentum(self.parameters(), lr, momentum)
        hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "time"]}

        best_loss, best_state, bad = float("inf"), None, 0
        self.best_epoch_ = None

        for ep in range(1, epochs + 1):
            t0 = time.time()
            self.train_one_epoch(X, y, opt, batch_size, weight_decay)
            dt = time.time() - t0

            # đo lại với dropout tắt để so sánh công bằng với validation
            tl, ta = self.evaluate(X, y)
            hist["train_loss"].append(tl)
            hist["train_acc"].append(ta)
            hist["time"].append(dt)
            msg = f"Epoch {ep:02d}/{epochs:02d} | {dt:5.1f}s | train loss {tl:.4f} acc {ta:.4f}"

            if validation_data is not None:
                vl, va = self.evaluate(X_val, y_val)
                hist["val_loss"].append(vl)
                hist["val_acc"].append(va)
                msg += f" | val loss {vl:.4f} acc {va:.4f}"

            if verbose:
                print(msg)

            if not torch.isfinite(torch.tensor(tl)):
                if verbose:
                    print("Phân kỳ (loss = nan/inf), dừng sớm.")
                break

            if patience is not None:
                if vl < best_loss:
                    best_loss, self.best_epoch_, bad = vl, ep, 0
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    bad += 1
                    if bad >= patience:
                        if verbose:
                            print(f"Early stopping ở epoch {ep} (tốt nhất: epoch {self.best_epoch_})")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        for k in self.history_:
            self.history_[k].extend(hist[k])
        return hist

    # ---------- dự đoán ----------
    def predict_logits(self, X, batch_size=5000):
        X = X.float()
        out = [self.forward_with_cache(X[s:s + batch_size], train=False)[0]
               for s in range(0, len(X), batch_size)]
        return torch.cat(out, dim=0)

    def predict_proba(self, X, batch_size=5000):
        return torch.softmax(self.predict_logits(X, batch_size), dim=1)

    def predict(self, X, batch_size=5000):
        return self.predict_logits(X, batch_size).argmax(dim=1)

    


