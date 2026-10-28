"""
Cài đặt: pip install torch torchvision matplotlib numpy
Chạy:    python mnist_cnn.py
Trong Colab: %run mnist_cnn.py

Đổi epochs, batch_size, lr ngay trong main().
Dùng autograd của PyTorch, không dùng backward thủ công như bài MLP.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                           # (N, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (N, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                           # (N, 64, 7, 7)
            nn.Flatten(),                              # (N, 3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),                        # 10 logits
        )

    def forward(self, x):
        return self.layers(x)


class Trainer:
    def __init__(self, model, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self, loader):
        self.model.train()
        with torch.enable_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader, collect=False):
        self.model.eval()
        loss_sum = correct = count = 0
        all_labels, all_probs = [], []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss_sum += self.loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
            if collect:
                all_labels.append(labels.cpu())
                all_probs.append(logits.softmax(1).cpu())

        result = {'loss': loss_sum / count, 'accuracy': correct / count}
        if collect:
            result['labels'] = torch.cat(all_labels)
            result['probs'] = torch.cat(all_probs)
        return result

    def fit(self, train_loader, val_loader, epochs=5):
        history = {key: [] for key in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}
        for epoch in range(epochs):
            self.train_epoch(train_loader)
            train = self.evaluate(train_loader)
            val = self.evaluate(val_loader)
            history['train_loss'].append(train['loss'])
            history['val_loss'].append(val['loss'])
            history['train_acc'].append(train['accuracy'])
            history['val_acc'].append(val['accuracy'])
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train loss: {train['loss']:.4f}, acc: {train['accuracy']:.4f} | "
                  f"Val loss: {val['loss']:.4f}, acc: {val['accuracy']:.4f}")
        return history


def show_images(images, titles, title='MNIST'):
    """Lưới ảnh nền trắng, chữ số tối giống file mẫu. Ảnh có giá trị [0, 1]."""
    if len(images) == 0:
        print('Không có ảnh để hiển thị.')
        return
    rows = (len(images) + 9) // 10
    fig, axes = plt.subplots(rows, 10, figsize=(14, rows * 1.8), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, image, text in zip(axes.flat, images, titles):
        ax.imshow(image.squeeze(), cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(text, fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    for ax, metric, label in zip(axes, ['loss', 'acc'], ['Cross-entropy', 'Accuracy']):
        ax.plot(epochs, history[f'train_{metric}'], '-', color='tab:blue', label='Train')
        ax.plot(epochs, history[f'val_{metric}'], '--', color='tab:blue', label='Validation')
        ax.set(xlabel='Epoch', ylabel=label)
        ax.legend()
    fig.suptitle('CNN MNIST — liền: train, đứt: validation')
    plt.tight_layout()
    plt.show()


def plot_results(test_data, result):
    labels, probs = result['labels'], result['probs']
    predictions = probs.argmax(1)
    matrix = torch.bincount(10 * labels + predictions, minlength=100).reshape(10, 10)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = np.log1p(matrix.numpy())
    ax.imshow(colors, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(matrix[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if colors[i, j] > colors.max() / 2 else 'black')
    ax.set(xticks=range(10), yticks=range(10), xlabel='Dự đoán', ylabel='Nhãn thật',
           title='Ma trận nhầm lẫn — màu theo log(1 + số lượng)')
    plt.tight_layout()
    plt.show()

    show_images(test_data.data[:20].float() / 255,
                [f'Thật {labels[k]}, đoán {predictions[k]}' for k in range(20)],
                'Kết quả dự đoán')
    # 20 ảnh dự đoán sai có độ tự tin cao nhất
    wrong = torch.where(labels != predictions)[0]
    confidence = probs.max(1).values
    top = wrong[confidence[wrong].argsort(descending=True)[:20]]
    show_images(test_data.data[top].float() / 255,
                [f'Thật {labels[k]}, đoán {predictions[k]}\np={confidence[k]:.2f}' for k in top],
                'Các ảnh dự đoán sai')

def save_weights(model, path='mnist_cnn.pth'):
    """Lưu trọng số mô hình."""
    torch.save(model.state_dict(), path)
    print(f'Đã lưu trọng số vào: {path}')

@torch.no_grad()
def predict(model, image):
    """Nhận tensor ảnh [0, 1], shape (28, 28) hoặc (1, 28, 28).

    Trả về (chữ số dự đoán, xác suất softmax của chữ số đó).
    Ví dụ: digit, confidence = predict(model, test_data[0][0])
    """
    model.eval()
    device = next(model.parameters()).device
    if image.ndim == 2:
        image = image.unsqueeze(0)  # Thêm chiều kênh màu.
    if tuple(image.shape) != (1, 28, 28):
        raise ValueError('Ảnh phải có kích thước (28, 28) hoặc (1, 28, 28).')
    image = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    probabilities = model(image).softmax(dim=1)
    confidence, predicted = probabilities.max(dim=1)
    return predicted.tolist(), confidence.tolist()

def read_dataset():
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

    return {'train_data' : train_data, 'test_data' : test_data}

def main():
    torch.manual_seed(42)
    epochs, batch_size, lr = 5, 128, 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    dataset = read_dataset()
    train_data, test_data = dataset['train_data'], dataset['test_data']

    train_data, val_data = random_split(
        train_data, [50000, 10000], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # lay cac images, labels tuong ung trong batch dau tien
    images, labels = next(iter(train_loader))
    show_images(images[:20], [f'y = {int(y)}' for y in labels[:20]], 'Ảnh huấn luyện')

    model = CNN()
    print(model)
    trainer = Trainer(model, device, lr)

    # do thoi gian train cnn
    start_time = time.perf_counter()

    history = trainer.fit(train_loader, val_loader, epochs)

    end_time = time.perf_counter()
    time_cnn_train = end_time - start_time

    save_weights(model) # save model
    plot_history(history)

    # Dùng mô hình ở epoch cuối; chỉ đánh giá test sau khi huấn luyện xong.
    result = trainer.evaluate(test_loader, collect=True)
    print(f"Test loss: {result['loss']:.4f} | Test accuracy: {result['accuracy']:.4f}")
    plot_results(test_data, result)

    return time_cnn_train
if __name__ == '__main__':
    main()
