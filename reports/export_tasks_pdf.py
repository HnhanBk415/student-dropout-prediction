from pathlib import Path
from urllib.request import urlretrieve
from tempfile import gettempdir
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from pypdf import PdfReader

font_dir = Path(gettempdir()) / 'student_dropout_prompt_fonts'
font_dir.mkdir(exist_ok=True)
for weight in ('Regular', 'SemiBold'):
    path = font_dir / f'Prompt-{weight}.ttf'
    if not path.exists():
        urlretrieve(f'https://raw.githubusercontent.com/google/fonts/main/ofl/prompt/Prompt-{weight}.ttf', path)
    pdfmetrics.registerFont(TTFont(f'Prompt-{weight}', str(path)))

items = [
('Chuẩn bị và kiểm tra dữ liệu', 'Đọc dữ liệu CSV, chuẩn hóa tên cột và kiểm tra nhãn. Kiểm tra giá trị thiếu, dòng trùng; phân nhóm đặc trưng số, phân loại và nhị phân.'),
('Xây dựng bộ chia dữ liệu dùng chung', 'Chia Train/Validation/Test theo tỷ lệ 70/15/15, phân tầng theo nhãn và cố định seed. Tạo 5 fold trên tập Train phục vụ cross-validation. Lưu chỉ số chia tập và mã băm SHA-256 để kiểm tra tính nhất quán của dữ liệu.'),
('Thực hiện EDA trên tập Train', 'Thống kê mô tả, số lượng và tỷ lệ từng lớp. Xuất bảng dữ liệu thiếu, biểu đồ phân bố nhãn và histogram các đặc trưng số.'),
('Xây dựng pipeline tiền xử lý', 'Điền giá trị thiếu bằng median hoặc giá trị phổ biến nhất. Chuẩn hóa đặc trưng số bằng StandardScaler, mã hóa phân loại bằng OneHotEncoder. Đưa tiền xử lý vào pipeline để fit riêng trong từng fold, hạn chế rò rỉ dữ liệu.'),
('Xây dựng khung huấn luyện cho bốn mô hình', 'Xây dựng Dummy Classifier làm baseline, Logistic Regression, Random Forest và MLP bằng PyTorch. Tìm siêu tham số cho Logistic Regression và Random Forest bằng GridSearchCV, sử dụng Macro-F1.'),
('Chuẩn hóa đánh giá mô hình', 'Tính Accuracy, Macro-F1, F2 cho lớp Dropout, Precision/Recall/F1 từng lớp và confusion matrix. Lưu kết quả validation và tạo bảng so sánh bốn mô hình.'),
('Lưu trữ và quản lý kết quả thực nghiệm', 'Lưu mô hình, dự đoán, xác suất từng lớp, cấu hình và phiên bản thư viện theo từng lượt chạy. Kiểm tra các lượt chạy sử dụng cùng bộ chia trước khi so sánh.'),
('Tổ chức quy trình bằng notebook', 'Tách các hàm dùng chung vào common/. Xây dựng notebook cho EDA, Dummy baseline, so sánh validation và đánh giá test cuối cùng. Bổ sung thiết lập chạy trên VS Code/Jupyter và Google Colab.'),
('Bổ sung kiểm tra tính hợp lệ của bộ chia', 'Phát hiện chỉ số trùng trong từng tập và từng fold. Viết 5 test trong tests/test_splits.py cho bộ chia hợp lệ, chỉ số trùng, tập chồng lấn và sai mã băm dữ liệu.'),
('Chuẩn bị bước đánh giá cuối cùng', 'Viết quy trình huấn luyện lại trên Train + Validation sau khi chốt mô hình. Bổ sung điều kiện khóa lựa chọn và ngăn ghi đè kết quả final test.')]

out = Path(__file__).with_name('danh_sach_cong_viec_da_thuc_hien.pdf')
title = ParagraphStyle('Title', fontName='Prompt-SemiBold', fontSize=16, leading=23, textColor=HexColor('#17365D'), spaceAfter=7)
sub = ParagraphStyle('Subtitle', fontName='Prompt-Regular', fontSize=9, leading=14, textColor=HexColor('#555555'), spaceAfter=15)
heading = ParagraphStyle('Tas`k', fontName='Prompt-SemiBold', fontSize=10, leading=14, spaceAfter=3, textColor=HexColor('#17365D'))
body = ParagraphStyle('Body', fontName='Prompt-Regular', fontSize=9, leading=13, spaceAfter=9)
story = [Paragraph('DANH SÁCH CÔNG VIỆC ĐÃ THỰC HIỆN', title), Paragraph('Project: Student Dropout &amp; Academic Success Prediction', sub)]
for i, (h, b) in enumerate(items, 1):
    story.append(KeepTogether([Paragraph(f'{i}. {h}', heading), Paragraph(b, body)]))

def footer(c, d):
    c.setFont('Prompt-Regular', 8)
    c.setFillColor(HexColor('#777777'))
    c.drawRightString(A4[0] - 40, 23, f'Trang {d.page}')

doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=32, bottomMargin=36, title='Danh sách công việc đã thực hiện')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
reader = PdfReader(out)
text = '\n'.join(p.extract_text() for p in reader.pages)
assert 'DANH SÁCH CÔNG VIỆC ĐÃ THỰC HIỆN' in text
assert '?' not in text
print(f'PDF verified: {len(reader.pages)} page(s), Vietnamese text preserved.')
