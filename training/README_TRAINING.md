# Hướng Dẫn Huấn Luyện

Thư mục `training/` chứa toàn bộ phần phục vụ huấn luyện model YOLO riêng cho dự án.

## Luồng cơ bản

1. Đưa dữ liệu vào `dataset/raw/images` và `dataset/raw/labels`.
2. Chạy `python training/split_dataset.py` để chia train/val/test.
3. Cập nhật class names trong `training/data.yaml`.
4. Chạy `python run_train.py`.
5. Validate bằng `python training/validate_model.py`.
6. Export bằng `python training/export_model.py`.

## Cấu hình khuyến nghị cho RTX 3050 Ti 4GB

- Mặc định: `yolo26s.pt`, `imgsz=512`, `batch=4`
- Nếu VRAM lỗi: hạ xuống `yolo26n.pt`, `imgsz=416`
- Nếu vẫn lỗi: tiếp tục giảm `imgsz=320`, `batch=2`

## Kết quả đầu ra

- Kết quả train: `runs/train/`
- Model tốt nhất: `models/trained/best.pt`
- Kết quả validate: `runs/val/`
- Model export: `models/exported/`
