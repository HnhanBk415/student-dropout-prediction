"""Danh sách nhóm cột và hàm tạo bộ tiền xử lý cho bộ dữ liệu UCI 697."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Target"
LABELS = ("Dropout", "Enrolled", "Graduate")

# Application order là thứ tự nguyện vọng, nên nhóm này giữ thông tin thứ tự.
NUMERIC_COLUMNS = (
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
)

# Các số này là mã nhóm, ví dụ mã ngành. One-hot tạo một cột 0/1 cho mỗi nhóm.
CATEGORICAL_COLUMNS = (
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
)

# Các cột này dùng hai giá trị 0 và 1.
BINARY_COLUMNS = (
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
)

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS


def numeric_quality_report(frame):
    """Kiểm tra giá trị sai nghĩa, trả bảng số vi phạm và giữ nguyên dữ liệu đầu vào."""
    records = []
    numeric = {}

    def record(column, rule, invalid):
        records.append({
            "column": column,
            "rule": rule,
            "invalid_count": int(invalid.sum()),
        })

    for column in NUMERIC_COLUMNS:
        original = frame[column]
        values = pd.to_numeric(original, errors="coerce")
        record(column, "non_numeric", original.notna() & values.isna())
        record(column, "infinite", pd.Series(
            np.isinf(values.to_numpy(dtype=float, na_value=np.nan)), index=frame.index,
        ))
        # Ô thiếu vẫn được xử lý bởi imputer khi fit; ô vô hạn đã được ghi nhận riêng.
        numeric[column] = values.where(np.isfinite(values))

    # Số học phần và số lượt đánh giá là số đếm nguyên, không âm.
    count_columns = [
        column for column in NUMERIC_COLUMNS
        if column.startswith("Curricular units ") and "(grade)" not in column
    ]
    for column in count_columns:
        values = numeric[column]
        record(column, "negative_count", values.lt(0))
        record(column, "non_integer_count", values.notna() & values.mod(1).ne(0))

    age = numeric["Age at enrollment"]
    record("Age at enrollment", "non_positive_age", age.le(0))
    record("Age at enrollment", "non_integer_age", age.notna() & age.mod(1).ne(0))

    # Các khoảng điểm và thứ tự nguyện vọng lấy từ mô tả biến của bộ dữ liệu.
    ranges = {
        "Application order": (0, 9),
        "Previous qualification (grade)": (0, 200),
        "Admission grade": (0, 200),
        "Curricular units 1st sem (grade)": (0, 20),
        "Curricular units 2nd sem (grade)": (0, 20),
    }
    for column, (lower, upper) in ranges.items():
        values = numeric[column]
        record(column, f"outside_{lower}_{upper}", values.lt(lower) | values.gt(upper))
    order = numeric["Application order"]
    record("Application order", "non_integer_order", order.notna() & order.mod(1).ne(0))

    for semester in ("1st", "2nd"):
        approved = f"Curricular units {semester} sem (approved)"
        enrolled = f"Curricular units {semester} sem (enrolled)"
        record(approved, "approved_exceeds_enrolled", numeric[approved].gt(numeric[enrolled]))

    # GDP và lạm phát có thể âm. Các cột này chỉ kiểm tra kiểu số và giá trị vô hạn.
    return pd.DataFrame(records, columns=["column", "rule", "invalid_count"])


# Tạo bộ xử lý. Notebook huấn luyện sẽ fit bằng phần dữ liệu dùng để học.
# Logistic và MLP dùng scale_numeric=True để chuẩn hóa cột số.
# Random Forest có thể dùng False để giữ thang đo gốc.
# X dùng tên cột CSV đã bỏ khoảng trắng ở đầu và cuối.
# EDA giữ giá trị hiếm hợp lệ và số 0 có ý nghĩa; hàm này chuẩn hóa và mã hóa dữ liệu.
# StandardScaler học thang đo từ phần fit, các điểm nằm xa trung bình vẫn được giữ.
def make_preprocessor(*, scale_numeric=True):

    numeric_steps = [("fill_missing", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    # Điền mã phổ biến nếu thiếu. Mã chưa gặp sẽ có toàn 0 ở các cột one-hot tương ứng.
    categorical = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Điền giá trị phổ biến nếu thiếu và giữ mỗi biến nhị phân thành một cột 0/1.
    binary = SimpleImputer(strategy="most_frequent")

    # Xử lý riêng từng nhóm cột rồi ghép các cột đã biến đổi thành đầu vào X.
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), list(NUMERIC_COLUMNS)),
        ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
        ("binary", binary, list(BINARY_COLUMNS)),
    ])
