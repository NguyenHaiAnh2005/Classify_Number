import torch

from src.models.mlp import MLP

class MLPEnsemble:
    """Train independent CPU MLPs and average their class probabilities."""

    def __init__(
        self,
        sizes=(784, 128, 10),
        seeds=(42, 123, 2026),
        activation="relu",
        p_drop=0.0,
        init="he",
    ):
        self.sizes = tuple(sizes)
        self.seeds = tuple(seeds)

        if len(self.sizes) < 2 or any(s <= 0 for s in self.sizes):
            raise ValueError("sizes must contain positive input and output sizes.")

        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide at least two distinct seeds.")

        self.activation = activation
        self.p_drop = p_drop
        self.init = init

        # These lists are populated after training completes.
        self.models_ = []
        self.histories_ = []

    def _check_data(self, X, y=None):
        """Accept NumPy arrays or tensors; validate data and use the CPU."""
        X = torch.as_tensor(X, dtype=torch.float32, device="cpu")

        if X.ndim != 2 or X.shape[1] != self.sizes[0] or len(X) == 0:
            raise ValueError(f"X must have shape (N, {self.sizes[0]}) with N > 0.")

        if not torch.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

        if y is None:
            return X, None

        y = torch.as_tensor(y, device="cpu")

        if y.ndim != 1 or len(y) != len(X): 
            raise ValueError("y must have shape (N,) and match X.")

        if not torch.isfinite(y).all() or not torch.equal(y, y.long()):
            raise ValueError("y must contain finite integer labels.")

        if (y < 0).any() or (y >= self.sizes[-1]).any():
            raise ValueError("Labels must be in [0, number_of_classes - 1].")

        return X, y.long()

    @torch.no_grad()
    def fit(
        self,
        X,
        y,
        epochs=30,
        batch_size=128,
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0,
        validation_data=None,
        patience=None,
        verbose=True,
    ):
        """
        Train fresh members on the same training split.

        Each member uses a different seed for initialization,
        mini-batch shuffling, and dropout. No bootstrap sampling is used.

        Gradient calculation and parameter updates are handled manually
        by the existing MLP implementation.

        Returns
        -------
        self
            Training histories are available in self.histories_.
        """
        X, y = self._check_data(X, y)

        if patience is not None:
            if (
                validation_data is None
                or not isinstance(patience, int)
                or patience <= 0
            ):
                raise ValueError(
                    "patience requires validation_data "
                    "and a positive integer."
                )

        if validation_data is not None:
            validation_data = self._check_data(*validation_data)

        models, histories = [], []

        for index, seed in enumerate(self.seeds, start=1):
            if verbose:
                print(
                    f"\nTraining member {index}/{len(self.seeds)} "
                    f"(seed={seed})"
                )

            # Initialize a separate network for each member.
            model = MLP(
                list(self.sizes),
                activation=self.activation,
                p_drop=self.p_drop,
                init=self.init,
                seed=seed,
            )

            # Pass the member's seed again because MLP.fit()
            # resets the random seed before training.
            history = model.fit(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                validation_data=validation_data,
                patience=patience,
                seed=seed,
                verbose=verbose,
            )

            models.append(model)
            histories.append(history)

        # Replace the previous ensemble only after all members finish.
        # Calling fit() again trains new networks from scratch.
        self.models_ = models
        self.histories_ = histories

        return self

    @torch.no_grad()
    def predict_proba(self, X, batch_size=5000):
        """
        Return equal-weight soft voting probabilities.

        Output shape: (number_of_samples, number_of_classes).
        """
        if not self.models_:
            raise RuntimeError("Call fit() before prediction.")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        X, _ = self._check_data(X)

        probabilities = torch.zeros(
            (len(X), self.sizes[-1]),
            dtype=torch.float32,
        )

        for model in self.models_:
            # Average probabilities, not logits, weights, or labels.
            # Accumulation probabilities.
            probabilities.add_(
                # call the predict_proba of MLP class
                model.predict_proba(X, batch_size=batch_size)
            )

        return probabilities / len(self.models_)

    def predict(self, X, batch_size=5000):
        """Return the class with the largest averaged probability."""
        probabilities = self.predict_proba(X, batch_size=batch_size)
        return probabilities.argmax(dim=1)

    @torch.no_grad()
    def evaluate(self, X, y, batch_size=5000):
        """Return ensemble cross-entropy and accuracy, like MLP.evaluate()."""
        X, y = self._check_data(X, y)
        probabilities = self.predict_proba(X, batch_size=batch_size)

        # Select the probability assigned to each sample's true class.
        true_probabilities = probabilities[torch.arange(len(y)), y]

        # Prevent log(0) when a probability underflows to zero.
        minimum = torch.finfo(probabilities.dtype).tiny
        loss = -true_probabilities.clamp_min(minimum).log().mean()

        predictions = probabilities.argmax(dim=1)
        accuracy = (predictions == y).float().mean()

        return loss.item(), accuracy.item()

    @torch.no_grad()
    def evaluate_members(self, X, y, batch_size=5000):
        """Compare each trained member against the ensemble."""

        # call the self.evaluate() function of MLPEnsemble class, defined above
        loss, accuracy = self.evaluate(X, y, batch_size=batch_size)
        X, y = self._check_data(X, y)

        results = []

        for index, (seed, model) in enumerate(zip(self.seeds, self.models_), start=1):

            # call the model.evaluate() function of MLP class
            member_loss, member_accuracy = model.evaluate(X, y, batch_size=batch_size)

            # save each model
            results.append({
                "model": f"MLP {index}",
                "seed": seed,
                "loss": member_loss,
                "accuracy": member_accuracy,
            })

        results.append({
            "model": "Ensemble",
            "seed": None,
            "loss": loss,
            "accuracy": accuracy,
        })

        return results



import numpy as np
import torch

from sklearn.model_selection import StratifiedKFold, train_test_split

from src.models.mlp import MLP
from src.preprocessing.scaler import scale_flat
from src.evaluation.metrics import ClassificationMetrics


class KFoldMLPEnsemble:
    """Train one MLP per fold and combine them with equal soft voting."""

    def __init__(self, n_splits=5, mode="raw", seed=42):
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2.")

        if mode not in {"raw", "standard"}:
            raise ValueError("mode must be 'raw' or 'standard'.")

        self.n_splits = n_splits
        self.mode = mode
        self.seed = seed

        self.models_ = []
        self.tools_ = []
        self.histories_ = []
        self.fold_metrics_ = []
        self.oof_proba_ = None

    @staticmethod
    def _check_images(X):
        """Expect raw MNIST images with pixel values from 0 to 255."""
        X = np.asarray(X)

        if X.ndim != 3 or X.shape[1:] != (28, 28) or len(X) == 0:
            raise ValueError(
                "X must have shape (N, 28, 28), with N > 0."
            )

        if not np.isfinite(X).all() or X.min() < 0 or X.max() > 255:
            raise ValueError(
                "Pixels must be finite and between 0 and 255."
            )

        return X

    @torch.no_grad()
    def fit(
        self,
        X,
        y,
        epochs=30,
        batch_size=128,
        lr=0.05,
        momentum=0.9,
        weight_decay=1e-4,
        patience=5,
        early_stop_size=0.1,
        verbose=True,
    ):
        """Use separate training, early-stopping and held-out fold sets."""
        X = self._check_images(X)
        y = np.asarray(y)

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have shape (N,) and match X.")

        if (
            not np.isfinite(y).all()
            or not np.equal(y, y.astype(np.int64)).all()
        ):
            raise ValueError("Labels must be finite integers.")

        if y.min() < 0 or y.max() > 9:
            raise ValueError("Labels must be between 0 and 9.")

        if not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a positive integer.")

        y = y.astype(np.int64)

        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )

        models = []
        tools_list = []
        histories = []
        scores = []

        # Each sample receives exactly one held-out prediction.
        oof = torch.zeros((len(X), 10), dtype=torch.float32)

        for fold, (train_pool, held_out) in enumerate(
            splitter.split(X, y),
            start=1,
        ):
            fold_seed = self.seed + fold

            # Split only the training pool.
            # Never use the held-out fold for early stopping.
            train_idx, stop_idx = train_test_split(
                train_pool,
                test_size=early_stop_size,
                stratify=y[train_pool],
                random_state=fold_seed,
            )

            # Fit preprocessing on actual training data only.
            # The third output is the held-out fold's transformed data.
            X_train, X_stop, X_hold, tools = scale_flat(
                X[train_idx],
                X[stop_idx],
                X[held_out],
                mode=self.mode,
            )

            # All folds use exactly the same architecture.
            model = MLP(
                [784, 128, 10],
                activation="relu",
                p_drop=0.0,
                init="he",
                seed=fold_seed,
            )

            # Reuse the existing manual-gradient training implementation.
            history = model.fit(
                torch.from_numpy(X_train),
                torch.from_numpy(y[train_idx]),
                validation_data=(
                    torch.from_numpy(X_stop),
                    torch.from_numpy(y[stop_idx]),
                ),
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                patience=patience,
                seed=fold_seed,
                verbose=False,
            )

            # Predict samples excluded from both training and early stopping.
            probabilities = model.predict_proba(
                torch.from_numpy(X_hold)
            )
            oof[held_out] = probabilities

            metrics = ClassificationMetrics(
                y[held_out],
                probabilities.argmax(dim=1).numpy(),
                labels=np.arange(10),
            ).evaluate()

            scores.append({
                "fold": fold,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            })

            models.append(model)
            tools_list.append(tools)
            histories.append(history)

            if verbose:
                print(
                    f"Fold {fold}/{self.n_splits} | "
                    f"accuracy={metrics['accuracy']:.4f} | "
                    f"macro recall={metrics['macro_recall']:.4f} | "
                    f"macro F1={metrics['macro_f1']:.4f}"
                )

        self.models_ = models
        self.tools_ = tools_list
        self.histories_ = histories
        self.fold_metrics_ = scores
        self.oof_proba_ = oof

        return self

    @torch.no_grad()
    def predict_proba(self, X, batch_size=5000):
        """Apply each fold's preprocessing and average probabilities."""
        if not self.models_:
            raise RuntimeError("Call fit() before prediction.")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        X = self._check_images(X)

        # This fixed rescaling does not learn statistics from new data.
        flat = X.reshape(len(X), 784).astype(np.float32) / 255.0

        probabilities = torch.zeros(
            (len(X), 10),
            dtype=torch.float32,
        )

        for model, tools in zip(self.models_, self.tools_):
            # Standard mode requires the scaler fitted for this fold.
            # Never fit a new scaler during prediction.
            if self.mode == "standard":
                features = tools["scaler"].transform(flat)
            else:
                features = flat

            features = np.asarray(features, dtype=np.float32)

            probabilities += model.predict_proba(
                torch.from_numpy(features),
                batch_size=batch_size,
            )

        # Equal-weight soft voting across all fold models.
        return probabilities / len(self.models_)

    def predict(self, X, batch_size=5000):
        """Return the class with the largest averaged probability."""
        return self.predict_proba(
            X,
            batch_size=batch_size,
        ).argmax(dim=1)