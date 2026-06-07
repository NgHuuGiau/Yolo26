# Project Overview

## Giới thiệu dự án

Đây là dự án nhận diện vật thể realtime bằng webcam, chạy trên desktop Windows với Python, OpenCV, Ultralytics YOLO11 và PyTorch.

## Tính năng chính

- giao diện terminal tiếng Việt
- tự dò cấu hình máy trước khi chạy
- tự gợi ý mức phù hợp theo `CPU`, `RAM`, `GPU`, `VRAM`, `PyTorch`, `CUDA`
- hỗ trợ 5 model `YOLO11`
- hỗ trợ chụp mẫu train trực tiếp từ webcam bằng phím `T`
- pipeline huấn luyện, kiểm tra và export model riêng
- menu tổng để chạy nhanh các file chính
- bộ test để kiểm tra toàn hệ thống
- tự động fallback khi model hoặc cấu hình chính lỗi

## Ngôn ngữ và thư viện đang dùng

**Ngôn ngữ chính**

- `Python`

**Thư viện AI / Computer Vision**

- `ultralytics`
- `torch`
- `torchvision`
- `torchaudio`
- `opencv-python`
- `pillow`
- `numpy`

**Thư viện dữ liệu / hỗ trợ**

- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `PyYAML`
- `psutil`
- `GPUtil`

## 5 model YOLO11 đang dùng

- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

## 3 mức người dùng sẽ thấy khi chạy camera

- `Cao nhất`
- `Trung bình`
- `Yếu`

Hệ thống không ép model lớn nhất bằng mọi giá. Nó sẽ cố chọn mức cao nhất mà máy còn chạy ổn định.

Ví dụ:

- máy rất mạnh, VRAM lớn -> có thể lên `yolo11x.pt`
- máy mạnh -> có thể lên `yolo11l.pt`
- máy tầm trung như `RTX 3050 Ti 4GB` -> thường về `yolo11s.pt` hoặc `yolo11m.pt` (theo optimized selection)
- máy yếu hoặc CPU-only -> thường về `yolo11n.pt`

## Cấu hình fallback

Theo code thực tế:

- RTX 3050 Ti 4GB (entry GPU):
  - High: `yolo11s.pt` / `cuda:0` / `imgsz 640` / `max_det 150`
  - Medium: `yolo11s.pt` / `cuda:0` / `imgsz 512` / `max_det 120`
  - Low: `yolo11n.pt` / `cuda:0` / `imgsz 416` / `max_det 100`
- Fallback bắt buộc trong `settings.yaml`:
  - GPU + `yolo11m.pt` 768
  - GPU + `yolo11s.pt` 512
  - GPU + `yolo11s.pt` 416
  - GPU + `yolo11n.pt` 416
  - CPU + `yolo11s.pt` 416
  - CPU + `yolo11n.pt` 320

## Cấu trúc thư mục

```text
YOLO/
|-- app/
|   |-- camera_app.py
|
|-- config/
|   |-- model_config.yaml
|   `-- settings.yaml
|
|-- core/
|   |-- camera_runner.py
|   |-- fallback_manager.py
|   |-- hardware_info.py
|   |-- model_loader.py
|   |-- model_selector.py
|   `-- runtime_advisor.py
|
|-- dataset/
|   |-- raw/
|   |   |-- images/
|   |   `-- labels/
|   |
|   |-- processed/
|   |   |-- images/
|   |   |   |-- train/
|   |   |   |-- val/
|   |   |   `-- test/
|   |   `-- labels/
|   |       |-- train/
|   |       |-- val/
|   |       `-- test/
|   |
|   `-- sample/
|       |-- images/
|       `-- labels/
|
|-- docs/
|   |-- install_guide.md
|   |-- project_overview.md
|   |-- runtime_tool_guide.md
|   `-- training_guide.md
|
|-- models/
|   |-- pretrained/
|   |-- trained/
|   `-- exported/
|
|-- output/
|   |-- logs/
|   |-- screenshots/
|   `-- videos/
|
|-- runs/
|   |-- train/
|   |-- detect/
|   `-- val/
|
|-- tests/
|
|-- tools/
|   `-- runtime_tool.py
|
|-- training/
|   |-- _training_bootstrap.py
|   |-- auto_label_raw.py
|   |-- data.yaml
|   |-- download_models.py
|   |-- export_model.py
|   |-- model_paths.py
|   |-- prepare_dataset.py
|   |-- promote_samples.py
|   |-- split_dataset.py
|   |-- terminal_ui.py
|   |-- train_config.yaml
|   |-- train_model.py
|   `-- validate_model.py
|
|-- utils/
|   |-- console_ui.py
|   |-- draw_utils.py
|   |-- file_utils.py
|   `-- logger.py
|
|-- README.md
|-- requirements.txt
|-- run_app.py
|-- run_detect.py
|-- run_doctor.py
|-- run_menu.py
|-- run_tests.py
|-- run_train.py
|-- run_tools.py
```

### Giải thích nhanh từng nhóm

`app/`

- chứa luồng app camera cấp cao
- `camera_app.py`: nối phần cứng, chọn mode, terminal UI và camera runtime

`config/`

- chứa cấu hình YAML
- `settings.yaml`: model assignment, device, imgsz cho từng profile, camera presets, inference params
- `model_config.yaml`: thứ tự ưu tiên load model, preferred models hiển thị

`core/`

- lõi xử lý chính của dự án
- `camera_runner.py`: chạy camera, đọc frame, detect, lưu sample
- `hardware_info.py`: đọc CPU, RAM, GPU, VRAM, CUDA
- `model_loader.py`: nạp model YOLO theo thứ tự ưu tiên
- `model_selector.py`: quyết định cấu hình runtime theo settings.yaml
- `runtime_advisor.py`: đường tối ưu riêng dùng cho app/detect, có bảng spec hardcoded
- `fallback_manager.py`: tạo chuỗi fallback khi cấu hình chính lỗi

`dataset/`

- chứa toàn bộ dữ liệu
- `sample/`: ảnh và label chụp trực tiếp từ app (phím T)
- `raw/`: dữ liệu gốc chính thức để train
- `processed/`: dữ liệu đã chia thành `train / val / test`

`docs/`

- tài liệu của dự án

`models/`

- `pretrained/`: 5 model YOLO11 tải sẵn
- `trained/`: `best.pt` sau khi train
- `exported/`: model đã export như ONNX

`output/`

- log, screenshot, video do app hoặc script tạo ra

`runs/`

- output train chuẩn của Ultralytics

`tests/`

- toàn bộ test tự động của hệ thống

`tools/`

- công cụ thăm dò cấu hình máy
- `runtime_tool.py`: xem cấu hình, 3 mức tối ưu, đánh giá hệ thống

`training/`

- toàn bộ pipeline huấn luyện
- `prepare_dataset.py`: tạo cấu trúc thư mục dự án
- `download_models.py`: tải 5 model YOLO11
- `auto_label_raw.py`: tự động sinh label cho ảnh chưa có label
- `promote_samples.py`: chuyển sample sang raw
- `split_dataset.py`: chia raw thành processed theo tỉ lệ 70/15/15
- `train_model.py`: train model với auto-prepare + fallback
- `validate_model.py`: kiểm tra model sau train
- `export_model.py`: export model sang ONNX
- `terminal_ui.py`: giao diện terminal cho pipeline training
- `model_paths.py`: resolve path model và data

`utils/`

- tiện ích chung
- `console_ui.py`: UI terminal cho app, dashboard, prompt
- `draw_utils.py`: hàm vẽ detect lên ảnh/frame
- `file_utils.py`: đọc ghi file/yaml, tạo thư mục dự án
- `logger.py`: logging ra file + console

### File chính nên nhớ

- `run_menu.py` = menu tổng
- `run_app.py` = chạy app camera chính (desktop)
- `run_detect.py` = detect camera (CLI)
- `run_tools.py` = xem cấu hình máy và 3 mức tối ưu
- `run_doctor.py` = kiểm tra hệ thống
- `run_tests.py` = kiểm tra code
- `run_train.py` = huấn luyện model
