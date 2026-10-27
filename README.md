# GarageGuard — Hệ thống cảnh báo hành vi nghi vấn quanh xe máy


## 1. Chức năng

- Đăng nhập theo vai trò: quản trị viên, nhân viên an ninh và người quản lý.
- Tải video MP4/AVI/MOV/MKV/M4V lên để xử lý.
- Tiền xử lý bằng OpenCV: giữ tỷ lệ khung hình khi thu nhỏ, tùy chọn CLAHE và khử nhiễu màu.
- YOLO phát hiện lớp `person` và `motorcycle`; ByteTrack theo dõi ID qua khung hình.
- Luật theo thời gian đánh dấu sự kiện khi người ở gần xe lâu bất thường hoặc xe dịch chuyển trong lúc người vẫn ở gần.
- Xem video đã gắn nhãn, danh sách sự kiện, ảnh chụp và tải `alerts.csv`.
- Demo tổng hợp chạy không cần YOLO/trọng số mô hình, phục vụ kiểm tra logic và giao diện.
- Dùng SQLite để lưu tài khoản demo và lịch sử phân tích.

## 2. Yêu cầu

- Python 3.10–3.12 được khuyến nghị.
- Để chạy giao diện web và demo tổng hợp: cài `requirements-demo.txt` (không cần Ultralytics hay trọng số YOLO).
- Để xử lý video CCTV thật: cài `requirements.txt`; lần đầu có thể cần Internet để tải trọng số `yolo11n.pt` (hoặc `yolov8n.pt`), trừ khi file trọng số đã có sẵn.
- CPU vẫn có thể chạy nhưng chậm; GPU tương thích CUDA có thể tăng tốc.

## 3. Cài đặt trên Windows

Mở PowerShell/CMD tại thư mục chứa source code:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python web_app.py
```

Mở trình duyệt tại `http://127.0.0.1:5000`.

Có thể chạy file `run_windows.bat` để tự tạo môi trường, cài đầy đủ thư viện và mở web app. Nếu chỉ cần mở giao diện và chạy demo tổng hợp, có thể thay lệnh cài đặt bằng `pip install -r requirements-demo.txt`; chế độ video CCTV thật sẽ cần cài thêm `ultralytics` bằng `pip install ultralytics`. Nếu PowerShell chặn kích hoạt môi trường ảo, dùng CMD và chạy `.venv\Scripts\activate.bat`.

## 4. Tài khoản demo

| Vai trò | Tên đăng nhập | Mật khẩu | Quyền minh họa |
|---|---|---|---|
| Quản trị viên | `admin` | `admin123` | Tải/xử lý video, chạy demo, xem lịch sử |
| Nhân viên an ninh | `baove` | `baove123` | Tải/xử lý video, chạy demo, xem lịch sử |
| Người quản lý | `quanly` | `quanly123` | Xem lịch sử và kết quả |



## 5. Cách chạy

### A. Chạy demo từ giao diện web

1. Đăng nhập bằng `baove / baove123`.
2. Nhấn **Chạy demo tổng hợp**.
3. Mở trang kết quả để xem video MP4, log CSV và khung hình minh họa.


### B. Phân tích video thật

1. Đăng nhập bằng tài khoản `admin` hoặc `baove`.
2. Chọn video được phép sử dụng.
3. Giữ tham số mặc định lần đầu; bấm **Phân tích video**.
4. Chờ xử lý kết thúc; trang kết quả sẽ hiển thị video đã vẽ khung bao, log sự kiện và ảnh chụp cảnh báo nếu có.

Lần đầu xử lý video thật có thể cần mạng để tải trọng số YOLO. Nếu không có kết nối mạng, hãy chạy demo tổng hợp trước.

### C. Chạy CLI không qua giao diện web

```bash
python main.py --demo --output-dir outputs
python -m unittest -v test_rule_logic.py
```

Trên video thật:

```bash
python main.py --source "D:\\CCTV\\ham_xe.mp4" --output-dir outputs
```

Tiền xử lý tùy chọn:

```bash
python main.py --source "D:\\CCTV\\thieu_sang.mp4" --clahe
python main.py --source "D:\\CCTV\\nhieu.mp4" --denoise
```

## 6. Cấu trúc thư mục

```text
Canh_bao_trom_xe_Web/
├── web_app.py              # Flask: đăng nhập, tải video, lịch sử, trang kết quả
├── main.py                 # OpenCV + YOLO + ByteTrack + luật cảnh báo
├── test_rule_logic.py      # Kiểm thử luật cảnh báo
├── requirements.txt       # Đủ dependency cho video thật
├── requirements-demo.txt  # Dependency nhẹ để chạy web/demo
├── templates/              # Các trang HTML
├── static/style.css        # Giao diện
├── instance/app.db         # Tự sinh khi chạy
└── storage/
    ├── uploads/            # Video đầu vào
    └── runs/               # Video đầu ra, CSV, ảnh sự kiện
```

## 7. Ý nghĩa và giới hạn kỹ thuật

- YOLO pretrained chỉ phát hiện đối tượng (người/xe máy), không tự phân loại hành vi trộm xe.
- Luật nền dùng khoảng cách giữa hộp bao, thời gian ở gần và độ dịch chuyển của hộp xe. Chủ xe đang lấy xe, nhân viên bảo vệ hoặc người sửa xe cũng có thể kích hoạt cảnh báo.
- Mô hình chưa được huấn luyện riêng trên dữ liệu nhà xe của đề tài. Không thể báo cáo accuracy/precision/recall thực tế nếu chưa có video gán nhãn và tập test độc lập.
- Muốn nâng cấp đúng hướng báo cáo, cần thu thập video có đồng ý/được phép, gán nhãn `normal`, `suspicious`, `theft_simulated`, sau đó huấn luyện/đánh giá mô hình nhận dạng hành động theo chuỗi thời gian và hiệu chỉnh ngưỡng cảnh báo.
- Ứng dụng hiện chạy cục bộ và xử lý đồng bộ một video mỗi lượt; chưa phải hệ thống vận hành production với camera trực tiếp, hàng đợi tác vụ hoặc push notification.

## 8. Thư viện và tham khảo

- OpenCV: đọc/ghi video, resize, CLAHE, lọc nhiễu, vẽ khung bao.
- Ultralytics YOLO: phát hiện đối tượng.
- ByteTrack qua API tracking của Ultralytics: duy trì ID theo dõi.
- Flask + SQLite: giao diện web và lưu lịch sử.
- Tài liệu: https://docs.opencv.org/ ; https://docs.ultralytics.com/modes/track ; https://flask.palletsprojects.com/
