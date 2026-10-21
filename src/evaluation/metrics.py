import numpy as np

class ClassificationMetrics:
    def __init__(self, y_true, y_pred, labels=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if (
            y_true.ndim != 1
            or y_pred.ndim != 1
            or y_true.size == 0
            or y_true.shape != y_pred.shape
        ):
            raise ValueError("Hai mảng nhãn phải 1D, cùng độ dài và không rỗng.")

        if labels is None:
            """
            Gộp cả nhãn thật và nhãn dự đoán để không bỏ sót lớp nào
            vd: tap y_test = [0, 1,2 ], y_pred = [0, 1] thì labels = [0, 1, 2]
            phòng trường hợp trong tập test không đủ full nhãn ta tận dụng tập y_pred đã học đc
            ở tập train, cách này chưa giải quyết triệt để
            """
            self.labels = np.unique(np.concatenate([y_true, y_pred]))
        else:
            self.labels = np.asarray(labels)


        if (
            self.labels.ndim != 1
            or np.unique(self.labels).size != self.labels.size
            or not np.all(np.isin(y_true, self.labels))
            or not np.all(np.isin(y_pred, self.labels))
        ):
            raise ValueError("labels phải gồm các class duy nhất và chứa mọi nhãn.")

        # Confusion matrix: hàng = nhãn thật, cột = nhãn dự đoán.
        n_classes = len(self.labels)
        self.cm = np.zeros((n_classes, n_classes), dtype=int)

        for i, actual in enumerate(self.labels):
            for j, predicted in enumerate(self.labels):
                self.cm[i, j] = np.sum(
                    (y_true == actual) & (y_pred == predicted)
                )

    def evaluate(self):
        tp = np.diag(self.cm)
        support = self.cm.sum(axis=1)    # Số mẫu thật của mỗi class
        predicted = self.cm.sum(axis=0)  # Số mẫu dự đoán là mỗi class

        fp = predicted - tp
        fn = support - tp

        # Precision = TP / (TP + FP)
        # Nếu mẫu số = 0, trả về 0.
        precision = np.divide(
            tp, tp + fp, # tp / tp + fp cho mọi lớp
            out=np.zeros(len(tp), dtype=float),
            where=(tp + fp) != 0,
        )

        # Recall = TP / (TP + FN)
        recall = np.divide(
            tp, tp + fn,
            out=np.zeros(len(tp), dtype=float),
            where=(tp + fn) != 0,
        )

        # F1 = 2TP / (2TP + FP + FN)
        f1 = np.divide(
            2 * tp, 2 * tp + fp + fn,
            out=np.zeros(len(tp), dtype=float),
            where=(2 * tp + fp + fn) != 0,
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": tp.sum() / self.cm.sum(),
            # Macro: trung bình đều trên các class.
            "macro_precision": precision.mean(),
            "macro_recall": recall.mean(),
            "macro_f1": f1.mean(),
        }

    def report(self):
        m = self.evaluate()

        print(f"{'Class':>10} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>10} {'Support':>10}")

        for i, label in enumerate(self.labels):
            print(
                f"{str(label):>10} "
                f"{m['precision'][i]:>10.3f} "
                f"{m['recall'][i]:>10.3f} "
                f"{m['f1'][i]:>10.3f} "
            )

        print(
            f"{'Macro avg':>10} "
            f"{m['macro_precision']:>10.3f} "
            f"{m['macro_recall']:>10.3f} "
            f"{m['macro_f1']:>10.3f} "
            f"{m['support'].sum():>10d}"
        )
        print(f"\nAccuracy: {m['accuracy']:.3f}")