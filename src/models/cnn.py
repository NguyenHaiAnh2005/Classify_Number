"""CNN from scratch using only torch.Tensor for math.

Input: grayscale images (1, 28, 28), labels 0..9.
No torch.nn, torch.optim, or autograd in model/training.
Dataset is provided by the caller through fit().
"""

"""
References:
- https://princeton-introml.github.io/files/ch12.pdf
- https://cs231n.github.io/convolutional-networks/
- Dive into Deep Learning (Book) - https://d2l.ai/
"""
import torch
from torch.utils.data import DataLoader


class Parameter:
    """A tensor we want to learn, plus its manually computed gradient."""
    def __init__(self, data):
        self.data = data.detach()
        self.grad = None


class Conv2D:
    """Convolution with stride=1 and padding=0."""
    def __init__(self, in_channels, out_channels, kernel_size):
        self.k = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        std = (2 / (in_channels * kernel_size**2))**0.5
        self.weight = Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
        self.bias = Parameter(torch.zeros(out_channels))

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        n, c, h, w = x.shape
        k = self.k
        patches = x.unfold(2, k, 1).unfold(3, k, 1)
        ho, wo = patches.shape[2:4]

        # Every k*k patch is a row: convolution becomes matrix multiplication.
        cols = patches.permute(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, -1)
        y = cols @ self.weight.data.reshape(self.out_channels, -1).T + self.bias.data

        self.input_shape = x.shape
        self.cols = cols
        self.out_hw = (ho, wo)
        return y.reshape(n, ho, wo, self.out_channels).permute(0, 3, 1, 2).contiguous()

    def backward(self, grad):
        n, c, h, w = self.input_shape
        ho, wo = self.out_hw
        k = self.k
        g = grad.permute(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.weight.grad = (g.T @ self.cols).reshape_as(self.weight.data)
        self.bias.grad = g.sum(dim=0)

        dcols = g @ self.weight.data.reshape(self.out_channels, -1)
        dpatches = dcols.reshape(n, ho, wo, c, k, k).permute(0, 3, 1, 2, 4, 5)
        dx = torch.zeros((n, c, h, w), dtype=grad.dtype, device=grad.device)
        for i in range(k):
            for j in range(k):
                dx[:, :, i:i + ho, j:j + wo] += dpatches[..., i, j]
        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x.clamp_min(0)

    def backward(self, grad):
        return grad * self.mask


class MaxPool2D:
    """Non-overlapping max pooling with 2x2 windows."""
    def forward(self, x):
        n, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        flat = patches.contiguous().reshape(n, c, h // 2, w // 2, 4)
        y, self.indices = flat.max(dim=-1)
        self.input_shape = x.shape
        return y

    def backward(self, grad):
        n, c, h, w = self.input_shape
        dx = torch.zeros((n, c, h, w), dtype=grad.dtype, device=grad.device)
        for i in range(2):
            for j in range(2):
                dx[:, :, i::2, j::2] = grad * (self.indices == i * 2 + j)
        return dx


class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Parameter(torch.randn(in_features, out_features) * (2 / in_features)**0.5)
        self.bias = Parameter(torch.zeros(out_features))

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        self.x = x
        return x @ self.weight.data + self.bias.data

    def backward(self, grad):
        self.weight.grad = self.x.T @ grad
        self.bias.grad = grad.sum(dim=0)
        return grad @ self.weight.data.T


class Dropout:
    def __init__(self, p=0.2):
        if not 0 <= p < 1:
            raise ValueError('dropout p must be in [0, 1)')
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            self.mask = None
            return x
        self.mask = (torch.rand_like(x) >= self.p).to(x.dtype) / (1 - self.p)
        return x * self.mask

    def backward(self, grad):
        return grad if self.mask is None else grad * self.mask


class CrossEntropy:
    """Stable softmax cross-entropy for labels as class indices."""
    def forward(self, logits, labels):
        z = logits - logits.max(dim=1, keepdim=True).values
        exp_z = z.exp()
        denominator = exp_z.sum(dim=1, keepdim=True)
        self.probs = exp_z / denominator
        self.labels = labels.long()
        log_probs = z - denominator.log()
        n = labels.size(0)
        return -log_probs[torch.arange(n, device=labels.device), self.labels].mean()

    def backward(self):
        grad = self.probs.clone()
        n = len(self.labels)
        grad[torch.arange(n, device=grad.device), self.labels] -= 1
        return grad / n


class CNN:
    """CNN for 28x28 grayscale images and 10 classes."""
    def __init__(self, dropout=0.2):
        self.layers = [
            Conv2D(1, 8, 5), ReLU(), MaxPool2D(),
            Conv2D(8, 16, 5), ReLU(), MaxPool2D(),
            Flatten(), Linear(16 * 4 * 4, 64), ReLU(),
            Dropout(dropout), Linear(64, 10),
        ]
        self.device = torch.device('cpu')
        self.train()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def train(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = False

    def to(self, device):
        self.device = torch.device(device)
        for p in self.parameters():
            p.data = p.data.to(self.device)
        return self
    
    def predict_proba(self, x):
        
        dtype = self.parameters()[0].data.dtype
        x = torch.as_tensor(x, dtype=dtype, device=self.device)

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)

        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28):
            raise ValueError("Invalid image shape")

        self.eval()

        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)

        return probs


    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.argmax(dim=1)


class Adam:
    """Manual Adam optimizer; no torch.optim."""
    def __init__(self, parameters, lr=0.001):
        self.params = list(parameters)
        self.lr = lr
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.m[i] = 0.9 * self.m[i] + 0.1 * p.grad
                self.v[i] = 0.999 * self.v[i] + 0.001 * p.grad.square()
                m_hat = self.m[i] / (1 - 0.9**self.step_count)
                v_hat = self.v[i] / (1 - 0.999**self.step_count)
                p.data -= self.lr * m_hat / (v_hat.sqrt() + 1e-8)


def evaluate(model, dataset, batch_size=64):
    """Compute loss and accuracy; do not update model."""
    if len(dataset) == 0:
        raise ValueError('dataset cannot be empty')
    model.eval()  # Disable dropout.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = CrossEntropy()
    total_loss = correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            logits = model.forward(images)
            loss = criterion.forward(logits, labels)

            n = labels.size(0)
            total_loss += loss.item() * n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += n
    return total_loss / total, correct / total


def fit(model, train_dataset, val_dataset=None,
        epochs=10, batch_size=64, lr=0.001, patience=4):
    """Train CNN with datasets passed from OUTSIDE.

    Every dataset item must be (image, label):
      image: float tensor (1, 28, 28)
      label: integer 0..9

    If val_dataset is given, enable early stopping and restore best weights.
    """
    if len(train_dataset) == 0:
        raise ValueError('train_dataset cannot be empty')
    if val_dataset is not None and len(val_dataset) == 0:
        raise ValueError('val_dataset cannot be empty')
    if epochs < 1 or batch_size < 1 or lr <= 0 or patience < 1:
        raise ValueError('epochs, batch_size, lr and patience must be positive')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = CrossEntropy()
    optimizer = Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_weights = None
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()  # Enable dropout.
        total_loss = correct = total = 0

        with torch.no_grad():  # Manual backward; do not build an autograd graph.
            for images, labels in loader:
                images = images.to(model.device)
                labels = labels.to(model.device)

                logits = model.forward(images)
                loss = criterion.forward(logits, labels)

                grad = criterion.backward()   # dLoss / dLogits
                model.backward(grad)          # dLoss / dWeights in each layer
                optimizer.step()              # Update model weights

                n = labels.size(0)
                total_loss += loss.item() * n
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += n

        message = (f'Epoch {epoch}/{epochs} | '
                   f'train loss={total_loss / total:.4f} | '
                   f'train acc={correct / total:.2%}')

        if val_dataset is not None:
            val_loss, val_acc = evaluate(model, val_dataset, batch_size)
            message += f' | val loss={val_loss:.4f} | val acc={val_acc:.2%}'

            if val_loss < best_loss - 0.001:
                best_loss = val_loss
                best_weights = [p.data.clone() for p in model.parameters()]
                no_improvement = 0
            else:
                no_improvement += 1

        print(message, flush=True)

        if val_dataset is not None and no_improvement >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    if best_weights is not None:
        with torch.no_grad():
            for p, best in zip(model.parameters(), best_weights):
                p.data.copy_(best)
        model.eval()
    return model
