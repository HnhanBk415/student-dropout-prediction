"""Hàm tính metric dùng chung cho cả bốn nhánh mô hình của bộ dữ liệu UCI 697."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from common.preprocessing import LABELS


# Lớp cần nhận diện trong bài toán cảnh báo sớm.
DROPOUT_LABEL = LABELS[0]

# F2 đặt recall nặng hơn precision, dùng cho lớp Dropout.
DROPOUT_BETA = 2

# Các cột của bảng so sánh giữa Dummy, Logistic, Random Forest và MLP.
SUMMARY_COLUMNS = (
    "accuracy",
    "macro_f1",
    "dropout_recall",
    "dropout_precision",
    "dropout_f2",
)

__all__ = [
    "DROPOUT_LABEL",
    "LABELS",
    "SUMMARY_COLUMNS",
    "confusion_frame",
    "evaluate",
    "per_class_frame",
    "summary_frame",
]


# Nhãn có thể là chuỗi "Dropout" hoặc mã lớp 0/1/2 do MLP trả về.
# Hàm đưa cả hai dạng về cùng chuỗi nhãn để các nhánh so kết quả với nhau được.
def _to_labels(y, name):
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError(
            f"{name} phải là một dãy nhãn. Nếu đang có xác suất ba lớp, "
            "lấy argmax theo trục lớp trước khi gọi."
        )
    if values.size == 0:
        raise ValueError(f"{name} không có mẫu nào.")

    # Mã lớp được tra theo thứ tự LABELS, nên 0 là Dropout.
    if values.dtype.kind in "iu":
        if not np.isin(values, np.arange(len(LABELS))).all():
            raise ValueError(f"{name} có mã lớp ngoài khoảng 0..{len(LABELS) - 1}.")
        return np.asarray(LABELS, dtype=object)[values]

    labels = values.astype(object)
    unknown = sorted({str(label) for label in labels} - set(LABELS))
    if unknown:
        raise ValueError(f"{name} có nhãn lạ: {unknown}. Nhãn hợp lệ: {list(LABELS)}.")
    return labels


# Tính toàn bộ metric của một lần dự đoán.
# y_true là nhãn thật, y_pred là nhãn mô hình chọn cho cùng các mẫu đó.
# Thứ tự nhãn luôn là Dropout -> Enrolled -> Graduate ở mọi bảng và confusion matrix.
def evaluate(y_true, y_pred):
    true_labels = _to_labels(y_true, "y_true")
    predicted_labels = _to_labels(y_pred, "y_pred")
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"y_true có {len(true_labels)} mẫu nhưng y_pred có {len(predicted_labels)} mẫu."
        )

    label_order = list(LABELS)
    # zero_division=0 để Dummy chỉ dự đoán một lớp vẫn chấm được: lớp không được
    # dự đoán lần nào có precision 0 thay vì cảnh báo chia cho 0.
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=label_order, zero_division=0,
    )
    # Gọi lại với beta=2 để lấy F2 của từng lớp; chỉ F2 của Dropout được báo cáo.
    _, _, f2, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=label_order,
        beta=DROPOUT_BETA, zero_division=0,
    )

    per_class = {
        label: {
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
            "f2": float(f2[position]),
            "support": int(support[position]),
        }
        for position, label in enumerate(label_order)
    }
    dropout = per_class[DROPOUT_LABEL]

    return {
        "labels": tuple(LABELS),
        "n_samples": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        # Macro-F1 là metric chính: trung bình đều F1 của ba lớp.
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "dropout_recall": dropout["recall"],
        "dropout_precision": dropout["precision"],
        "dropout_f2": dropout["f2"],
        # Dòng là nhãn thật, cột là nhãn dự đoán, cùng thứ tự LABELS.
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=label_order,
        ),
    }


# Bảng điểm từng lớp của một kết quả evaluate(), dùng để display trong notebook.
def per_class_frame(result):
    return pd.DataFrame.from_dict(result["per_class"], orient="index").reindex(LABELS)


# Confusion matrix đã gắn tên nhãn: dòng là nhãn thật, cột là nhãn dự đoán.
def confusion_frame(result):
    return pd.DataFrame(
        result["confusion_matrix"],
        index=pd.Index(LABELS, name="thực tế"),
        columns=pd.Index(LABELS, name="dự đoán"),
    )


# Gộp nhiều kết quả thành một dòng mỗi mô hình để so sánh các nhánh.
# results là dict dạng {"Dummy": result_dummy, "Logistic": result_logistic, ...}.
def summary_frame(results):
    return pd.DataFrame(
        [{column: result[column] for column in SUMMARY_COLUMNS} for result in results.values()],
        index=pd.Index(list(results), name="model"),
    )
