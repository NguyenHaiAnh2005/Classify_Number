import torch
from torch import nn
import copy

class CNN(nn.Module):
    """CNN phân loại ảnh MNIST có kích thước 28 x 28."""

    def __init__(self, c1=16, c2=32):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (N, c1, 14, 14)

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (N, c2, 7, 7)

            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.layers(images)


class Trainer:
    """Huấn luyện và đánh giá mô hình phân loại."""

    def __init__(self, model, device, lr=0.001):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

    def train_epoch(self, loader):
        """Huấn luyện mô hình trong một epoch."""
        self.model.train()

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)

            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader, collect=False):
        """Tính loss, accuracy và tùy chọn thu thập dự đoán."""
        self.model.eval()

        loss_sum = 0.0
        correct = 0
        count = 0
        all_labels = []
        all_probs = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            count += batch_size

            if collect:
                all_labels.append(labels.cpu())
                all_probs.append(logits.softmax(dim=1).cpu())

        if count == 0:
            raise ValueError("DataLoader không có dữ liệu.")

        result = {
            "loss": loss_sum / count,
            "accuracy": correct / count,
        }

        if collect:
            result["labels"] = torch.cat(all_labels)
            result["probs"] = torch.cat(all_probs)

        return result

    def fit(self, train_loader, val_loader, epochs=30, patience=4):
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.best_epoch = None

        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader)

            train_result = self.evaluate(train_loader)
            val_result = self.evaluate(val_loader)

            history["train_loss"].append(train_result["loss"])
            history["val_loss"].append(val_result["loss"])
            history["train_acc"].append(train_result["accuracy"])
            history["val_acc"].append(val_result["accuracy"])

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train loss: {train_result['loss']:.4f}, "
                f"acc: {train_result['accuracy']:.4f} | "
                f"Val loss: {val_result['loss']:.4f}, "
                f"acc: {val_result['accuracy']:.4f}"
            )

            if val_result["loss"] < best_val_loss:
                best_val_loss = val_result["loss"]
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping tại epoch {epoch}; "
                    f"val_loss tốt nhất ở epoch {self.best_epoch}."
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history


@torch.no_grad()
def predict(model, image):
    """Dự đoán một ảnh tensor [0, 1].

    Ảnh có shape (28, 28) hoặc (1, 28, 28).
    Trả về chữ số dự đoán (int) và xác suất softmax (float).
    """
    model.eval()
    device = next(model.parameters()).device

    if image.ndim == 2:
        image = image.unsqueeze(0)

    if tuple(image.shape) != (1, 28, 28):
        raise ValueError(
            "Ảnh phải có kích thước (28, 28) hoặc (1, 28, 28)."
        )

    image = image.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )

    probabilities = model(image).softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)

    return prediction.item(), confidence.item()





