# Hệ thống Bảo trì Dự đoán (Predictive Maintenance)

Ứng dụng web Flask dự đoán **Remaining Useful Life (RUL)** của máy móc công nghiệp dựa trên dữ liệu cảm biến SMART, sử dụng mô hình Machine Learning (scikit-learn).

## Cấu trúc dự án

```
├── app.py                      # Ứng dụng Flask chính
├── config.py                   # Cấu hình (DB, đường dẫn, ngưỡng RUL)
├── requirements.txt            # Thư viện Python
├── database/
│   ├── connection.py           # Kết nối MySQL
│   ├── schema.sql              # Script tạo database & bảng
│   └── migrate.py              # Thêm cột SMART vào DB cũ (nếu cần)
├── models/
│   ├── predictor.py            # Load model & dự đoán RUL
│   └── trainer.py              # Training model từ dữ liệu CSV
├── data/                       # Dữ liệu training (CSV)
├── trained_models/             # File model đã train (.pkl)
├── templates/                  # Giao diện HTML (Jinja2)
└── uploads/                    # File CSV người dùng upload
```

---

## Yêu cầu hệ thống

- Python 3.10+
- MySQL Server (khuyến nghị dùng [XAMPP](https://www.apachefriends.org/) hoặc MySQL Community)

---

## Cài đặt

### 1. Tạo môi trường ảo & cài thư viện

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux/Mac:**
  ```bash
  source .venv/bin/activate
  ```

Cài thư viện:

```bash
pip install -r requirements.txt
```

### 2. Khởi động MySQL (Docker)

```bash
docker compose up -d
```

Docker sẽ tự tạo database `qlbaotri` và chạy schema. Mặc định port 3306, user `root`, password `root123`.

Nếu port 3306 đã bị chiếm (ví dụ có MySQL local), đổi port:

```bash
DB_PORT=3307 docker compose up -d
```

Và set biến môi trường trước khi chạy app:

```bash
set DB_PORT=3307
```

> Tất cả config DB có thể override qua biến môi trường: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`

### 3. Chuẩn bị data training

Copy file CSV dữ liệu SMART vào `data/smart_data.csv`. File mẫu có sẵn trong `uploads/DU-LIEU-SMART.csv`:

```bash
cp uploads/DU-LIEU-SMART.csv data/smart_data.csv
```

### 4. Train model (lần đầu)

Dự án đã có sẵn file model trong `trained_models/`. Nếu muốn train lại:

```bash
python -m models.trainer
```

Lệnh này sẽ:
- Đọc dữ liệu từ `data/smart_data.csv` (100,000 dòng)
- Train 4 model: Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- Tự động chọn model tốt nhất (theo R² score)
- Lưu vào `trained_models/smart_maintenance_model.pkl`

Train với file CSV khác:

```bash
python -m models.trainer path/to/your_data.csv
```

### 5. Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt: **http://localhost:8000**

---

## Hướng dẫn sử dụng

### Dashboard (`/`)

Trang chủ hiển thị tổng quan:
- Tổng số máy, số lần bảo trì, số máy Critical
- Nhiệt độ trung bình, độ ẩm, năng lượng tiêu thụ
- Số anomaly phát hiện
- Biểu đồ: chi phí bảo trì theo tháng, phân bố sức khỏe máy, xu hướng bảo trì

> Tất cả dữ liệu biểu đồ được lấy thật từ database.

### Quản lý máy (`/machines`)

- **Xem danh sách máy** với dữ liệu cảm biến và trạng thái RUL dự đoán
- **Thêm máy mới:** Điền form (ngày, nhiệt độ, rung, áp suất, độ ẩm, năng lượng, loại lỗi...)
- **Dự đoán RUL:** Nhấn nút "Predict" → xem chi tiết dự đoán cho từng máy
- **Xóa máy:** Nhấn nút "Delete"
- **Tìm kiếm:** Gõ vào ô search để lọc nhanh

Trạng thái RUL:
| RUL (giờ) | Trạng thái | Ý nghĩa |
|-----------|-----------|---------|
| > 1200 | 🟢 Good | Máy hoạt động bình thường |
| 400 – 1200 | 🟡 Warning | Cần kiểm tra / lên lịch bảo trì |
| < 400 | 🔴 Critical | Nguy cơ hỏng cao, cần bảo trì ngay |

### Quản lý bảo trì (`/maintenance`)

- **Xem lịch sử bảo trì** của tất cả máy
- **Thêm bản ghi bảo trì:** Chọn máy, ngày, kỹ thuật viên, chi phí, mô tả
- **Xóa bản ghi**
- **Tìm kiếm** theo bất kỳ trường nào

### Upload & Dự đoán hàng loạt (`/upload_predict`)

Upload file CSV chứa dữ liệu cảm biến → hệ thống dự đoán RUL cho tất cả máy trong file.

**Cột bắt buộc:** `Machine_id`, `Temperature`, `Vibration`, `Pressure`

**Cột tùy chọn** (tự điền mặc định nếu thiếu): `Humidity`, `Energy_consumption`, `Machine_status`, `Anomaly_flag`, `Failure_type`, `Downtime_risk`, `Maintenance_required`

> Tên cột không phân biệt hoa/thường. Ví dụ `temperature`, `TEMPERATURE`, `Temperature` đều được.

File CSV mẫu tối thiểu:

```csv
Machine_id,Temperature,Vibration,Pressure
1,75.5,12.3,101.3
2,82.1,15.7,100.9
```

Sau khi upload:
- Hiển thị bảng kết quả dự đoán (RUL, trạng thái, dữ liệu cảm biến)
- Dữ liệu tự động được lưu vào database

### Train model từ web (`/train`)

- Upload file CSV dataset
- Chọn loại model (giao diện hiển thị tùy chọn, hệ thống sẽ train tất cả và chọn tốt nhất)
- Nhấn "Train Model"
- Model mới được lưu và tự động reload — không cần restart app

**Yêu cầu file CSV training:** Phải có cột `predicted_remaining_life` (target) và các cột cảm biến SMART.

### Quản lý dataset (`/datasets`)

- Xem danh sách file CSV đã upload để training
- Xem trước 100 dòng đầu
- Download hoặc xóa dataset

---

## Dữ liệu SMART

File dữ liệu gốc: `data/smart_data.csv` (100,000 dòng)

| Cột | Mô tả |
|-----|-------|
| `timestamp` | Thời gian ghi nhận |
| `machine_id` | ID máy (1-60) |
| `temperature` | Nhiệt độ (°C) |
| `vibration` | Độ rung |
| `humidity` | Độ ẩm (%) |
| `pressure` | Áp suất |
| `energy_consumption` | Năng lượng tiêu thụ (kWh) |
| `machine_status` | Trạng thái máy (1=chạy, 0=dừng) |
| `anomaly_flag` | Cờ bất thường (0/1) |
| `predicted_remaining_life` | **RUL — target dự đoán** (giờ) |
| `failure_type` | Loại lỗi (Normal, Wear, ...) |
| `downtime_risk` | Mức rủi ro downtime |
| `maintenance_required` | Cần bảo trì (0/1) |

---

## Cấu hình

Tất cả cấu hình nằm trong `config.py`:

| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `DB_HOST` | MySQL host | `localhost` |
| `DB_USER` | MySQL user | `root` |
| `DB_PASSWORD` | MySQL password | (trống) |
| `DB_NAME` | Tên database | `qlbaotri` |
| `RUL_GOOD_THRESHOLD` | Ngưỡng Good | `1200` |
| `RUL_WARNING_THRESHOLD` | Ngưỡng Warning | `400` |
| `RUL_DEFAULT` | Giá trị mặc định khi model chưa sẵn sàng | `1500` |
