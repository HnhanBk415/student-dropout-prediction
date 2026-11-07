# README — Code dữ liệu, EDA, Dummy và tích hợp

Phần code của Người 1 trong project dự đoán **Dropout / Enrolled / Graduate**. README này hướng dẫn chạy các file Python, giải thích đầu vào, đầu ra và những gì cần bàn giao cho nhóm.

## 1. Cấu trúc code

```text
student-dropout-prediction/
├── data/data.csv                 # CSV đầu vào, cột nhãn Target
├── common/workflow.py            # đọc dữ liệu, chia tập, tiền xử lý, metrics
├── prepare.py                    # chia tập, tạo 5 fold, EDA trên Train
├── models.py                     # Dummy, Logistic Regression, RF, MLP
├── run_model.py                  # train trên Train, đánh giá Validation
├── compare.py                    # so sánh kết quả Validation
├── final_test.py                 # đánh giá Test sau khi chốt mô hình
├── requirements-workflow.txt     # thư viện cần cài
└── HUONG_DAN.md                  # hướng dẫn workflow đầy đủ
```

## 2. Chuẩn bị

Mở PowerShell ở thư mục gốc repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-workflow.txt
```

Đặt dataset ở `data/data.csv`. Code đọc CSV bằng dấu `;`; cột `Target` phải gồm các nhãn `Dropout`, `Enrolled`, `Graduate`.

## 3. Chia dữ liệu và chạy EDA

```powershell
python prepare.py --data data/data.csv --seed 42 --test-size 0.15 --validation-size 0.15
```

`prepare.py` kiểm tra dữ liệu, chia phân tầng Train / Validation / Test và tạo **5 fold trên Train**. Tỉ lệ 70% / 15% / 15% trong lệnh là ví dụ; nhóm cần chốt tỉ lệ trước lần chạy chính thức. Seed 42 giúp tái lập phép chia khi dùng cùng CSV và cùng quy trình.

| Đầu ra | Nội dung |
| --- | --- |
| `artifacts/splits.json` | Chỉ số dòng của ba tập và 5 fold, seed, SHA-256 của CSV. |
| `artifacts/eda_train/overview.json` | Kích thước, số dòng trùng, nhóm cột số / phân loại / nhị phân. |
| `artifacts/eda_train/missing.csv` | Số giá trị thiếu theo cột trên Train. |
| `artifacts/eda_train/target_counts.csv` | Số mẫu mỗi nhãn trên Train. |
| `artifacts/eda_train/class_proportions.csv` | Tỉ lệ nhãn trên Train. |
| `artifacts/eda_train/summary.csv` | Thống kê mô tả trên Train. |

Các thành viên dùng chung `splits.json` và **đúng phiên bản CSV** đã tạo ra nó. Code kiểm tra SHA-256 khi nạp file chia. Nếu file chia đã có, `prepare.py` dừng để tránh ghi đè; chỉ chạy lại với `--overwrite-splits` khi cả nhóm thống nhất và chấp nhận chạy lại toàn bộ Validation.

## 4. Dummy làm mốc

```powershell
python run_model.py --data data/data.csv --model dummy
```

Dummy dùng `DummyClassifier(strategy='most_frequent')`, dự đoán lớp phổ biến nhất trong Train. Các file được tạo:

```text
artifacts/validation/dummy/
├── predictions.csv    # row_index, actual, predicted, prob_* cho ba lớp
├── metrics.json       # Accuracy, Macro-F1, Dropout-F2, từng lớp, confusion matrix
└── config.json        # model, seed, mã kiểm tra splits.json
```

## 5. Tích hợp kết quả của các mô hình

Các thành viên chạy mô hình của mình trên **cùng phép chia**:

```powershell
python run_model.py --data data/data.csv --model logistic
python run_model.py --data data/data.csv --model random_forest
python run_model.py --data data/data.csv --model mlp
python compare.py
```

Logistic Regression và Random Forest tìm tham số bằng 5 fold Train, tối ưu Macro-F1. Pipeline fit phép điền thiếu, chuẩn hóa và one-hot riêng trong từng fold. `compare.py` kiểm tra các kết quả dùng cùng `splits.json`, in bảng so sánh Validation và tạo `artifacts/validation_comparison.csv`.

## 6. Test sau khi chốt lựa chọn

Ví dụ khi nhóm đã chọn Logistic Regression và khóa cấu hình:

```powershell
python final_test.py --data data/data.csv --model logistic --selection-locked
```

Code fit lại trên Train + Validation, đánh giá Test và lưu tại `artifacts/test/`. Không dùng Test để chọn giữa các mô hình.

## Trạng thái bàn giao

Mã nguồn đã định nghĩa quy trình và đường dẫn đầu ra. README **chưa ghi điểm số thực nghiệm** vì phải chạy trên đúng dataset mới có kết quả. Người 1 bàn giao `splits.json`, các file EDA, kết quả Dummy và bảng so sánh Validation sau khi chạy; những người phụ trách mô hình tự phân tích và viết phần báo cáo của mình.
