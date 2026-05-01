# Datathon 2026: The Gridbreakers

Dự án này là mã nguồn tham gia cuộc thi **Datathon 2026: The Gridbreakers**, do VinTelligence - VinUni DS&AI Club tổ chức. Dự án tập trung vào phân tích dữ liệu e-commerce, khám phá insight kinh doanh, và xây dựng mô hình dự báo doanh thu.

## 📂 Cấu trúc Thư mục

```text
Datathon/
├── Data/                   # Chứa các file dữ liệu cuộc thi (.csv)
├── Part_1/                 # Phần 1: Câu hỏi Trắc nghiệm (MCQ)
│   └── QA.ipynb            
├── Part_2/                 # Phần 2: Trực quan hoá và Phân tích Dữ liệu (EDA)
│   ├── insight1-datathon.ipynb
│   └── insight2-datathon.ipynb
├── Part_3/                 # Phần 3: Mô hình Dự báo Doanh thu (Sales Forecasting)
│   └── model.py            
└── README.md
```

## 📝 Chi tiết Từng Phần (Dựa trên Source Code)

### 1. Data (Dữ liệu)
Thư mục chứa 15 bảng dữ liệu của hệ thống thương mại điện tử, được chia làm 4 lớp:
- **Master**: `products.csv`, `customers.csv`, `promotions.csv`, `geography.csv`
- **Transaction**: `orders.csv`, `order_items.csv`, `payments.csv`, `shipments.csv`, `returns.csv`, `reviews.csv`
- **Analytical**: `sales.csv` (dữ liệu huấn luyện 2012-2022), `sample_submission.csv`
- **Operational**: `inventory.csv`, `web_traffic.csv`

### 2. Phần 1 — Câu hỏi Trắc nghiệm (MCQ)
- **Thư mục:** `Part_1/QA.ipynb`
- **Nội dung:** Chứa mã nguồn Pandas tính toán để giải quyết 10 câu hỏi trắc nghiệm (MCQ). Phân tích trên tập `orders.csv`, `products.csv`, `returns.csv`, `web_traffic.csv`... để tìm ra inter-order gap, tỷ suất lợi nhuận gộp, lý do hoàn trả lớn nhất, và tỷ lệ thoát traffic.

### 3. Phần 2 — Trực quan hoá và Phân tích Dữ liệu (EDA)
Thư mục `Part_2/` chứa 2 báo cáo chuyên sâu áp dụng Framework 4 cấp độ (Descriptive, Diagnostic, Predictive, Prescriptive):
- **`insight1-datathon.ipynb` (The Chain of Destruction - Chuỗi phản ứng hủy diệt):** 
  - Khám phá nghịch lý kinh doanh từ sự "bốc hơi lợi nhuận".
  - Vẽ biểu đồ Waterfall (P&L) bóc tách từ Tổng Doanh Thu đến Lợi nhuận ròng.
  - Phân tích sự liên kết giữa Quản lý Tồn kho (Inventory), Định giá (Pricing) và Khuyến mãi (Promotions) bằng dữ liệu định lượng.
- **`insight2-datathon.ipynb` (Giải phẫu Hệ sinh thái Web Traffic):**
  - Chuyển hướng tập trung vào dữ liệu Marketing (Web Analytics).
  - Khai thác lưu lượng truy cập (Sessions) làm Proxy đại diện cho phân bổ ngân sách.
  - Bóc tách mối quan hệ giữa Traffic, Tỷ lệ chuyển đổi (CVR) và biến động Mùa Hè.

### 4. Phần 3 — Mô hình Dự báo Doanh thu (Sales Forecasting)
- **Thư mục:** `Part_3/model.py`
- **Nội dung:** Xây dựng mô hình Ensemble Machine Learning kết hợp mô hình Gradient Boosting (LightGBM với mục tiêu Regression, Tweedie, Huber) và Mạng nơ-ron (MLPRegressor) để dự báo Revenue và COGS.
- **Tính năng nổi bật (Feature Engineering):**
  - **Seasonal Factors:** Trích xuất hệ số mùa vụ Tết Nguyên Đán (`tf`), phân rã hệ số ngày trong tuần (`df`) và ngày trong năm (`mf`).
  - **Cyclical Encoding:** Dùng Sin/Cos cho biến chu kỳ thời gian (tháng, thứ, ngày).
  - **Payday Distance:** Đo khoảng cách đến các chu kỳ nhận lương mùng 1 và 15 hàng tháng.
- **Khả năng giải thích (Interpretability):** Tích hợp module phân tích tự động bằng `SHAP values` và `Permutation Importance`. Code tự động sinh ra các biểu đồ giải thích mô hình (Feature Importance, SHAP Summary, Dependence plot) và lưu vào thư mục `explainpicture/`.
- **Tính tái lập:** Thiết lập sẵn hệ số cố định `SEED = 42` để đảm bảo kết quả nhất quán. Kết quả đầu ra dự báo cuối cùng lưu ở file `submission.csv` / `final_submission.csv` chuẩn định dạng Kaggle.

## 🚀 Hướng dẫn Chạy lại kết quả (Reproducibility)

Để tái lập (reproduce) toàn bộ kết quả trong repository này, vui lòng thực hiện theo các bước sau:

**Bước 1: Cài đặt môi trường**
Cài đặt toàn bộ các thư viện Python cần thiết (bao gồm thư viện cho mô hình, Jupyter và vẽ biểu đồ) dựa trên file `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Bước 2: Chuẩn bị dữ liệu**
Toàn bộ 15 file dữ liệu `.csv` của cuộc thi cần được đặt trong thư mục `Data/` ở ngoài cùng (cùng cấp với `Part_1`, `Part_2`, `Part_3`).

**Bước 3: Hướng dẫn chạy lại từng phần (Cấu hình đường dẫn)**

*   **Phần 1 (MCQ - `Part_1/QA.ipynb`) & Phần 2 (EDA - `Part_2/*.ipynb`)**: 
    - Khi mở các file Notebook (`.ipynb`) bằng Jupyter, thư mục làm việc hiện tại sẽ là thư mục chứa file đó. 
    - Các file gốc đang dùng đường dẫn môi trường của Kaggle (ví dụ: `/kaggle/input/datasets/`). Khi chạy tại máy tính cá nhân, bạn hãy `Find and Replace` (Ctrl+F) để chuyển tất cả đường dẫn đọc file thành đường dẫn lùi về một cấp thư mục.
    - *Ví dụ:* Đổi `pd.read_csv('/kaggle/input/.../orders.csv')` thành:
      ```python
      df = pd.read_csv('../Data/orders.csv')
      ```
    - Chạy từ trên xuống dưới (Run All) để xem kết quả tính toán và biểu đồ.

*   **Phần 3 (Mô hình - `Part_3/model.py`)**:
    - Trong file `model.py`, để mô hình đọc được dữ liệu từ thư mục `Data`, hãy cập nhật đường dẫn ở đầu file.
    - Đổi tên file thành `../Data/sales.csv`, `../Data/sample_submission.csv` và `../Data/promotions.csv`:
      ```python
      train_raw = pd.read_csv('../Data/sales.csv', parse_dates=['Date'])
      test_raw = pd.read_csv('../Data/sample_submission.csv', parse_dates=['Date'])
      promos = pd.read_csv('../Data/promotions.csv', parse_dates=['start_date', 'end_date'])
      ```
    - Mở terminal, điều hướng vào thư mục Part 3 và chạy script:
      ```bash
      cd Part_3
      python model.py
      ```
    - Script sẽ sinh ra file dự báo `final_submission.csv` và thư mục `explainpicture/` chứa các biểu đồ giải thích (SHAP plots).
