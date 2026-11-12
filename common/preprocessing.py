"""Danh sách nhóm cột và hàm tạo bộ tiền xử lý cho bộ dữ liệu UCI 697."""

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


# Tạo bộ xử lý. Notebook huấn luyện sẽ fit bằng phần dữ liệu dùng để học.
# Logistic và MLP dùng scale_numeric=True để chuẩn hóa cột số.
# Random Forest có thể dùng False để giữ thang đo gốc.
# X dùng tên cột CSV đã bỏ khoảng trắng ở đầu và cuối.
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
