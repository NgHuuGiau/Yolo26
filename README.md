# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Ứng dụng nhận diện vật thể realtime bằng YOLO, chạy thuần Python + OpenCV, có chọn mode trong terminal, có fallback theo phần cứng, có pipeline train/validate/export, và có chức năng chụp mẫu train trực tiếp từ camera.

---

## Mục lục

- [1. Tổng quan](#1-tổng-quan)
- [2. Cài đặt môi trường](#2-cài-đặt-môi-trường)
- [3. Cách chạy dự án](#3-cách-chạy-dự-án)
- [4. Hướng dẫn huấn luyện](#4-hướng-dẫn-huấn-luyện)
- [5. Cấu trúc thư mục và file](#5-cấu-trúc-thư-mục-và-file)
- [6. Khắc phục lỗi nhanh](#6-khắc-phục-lỗi-nhanh)

---

## 1. Tổng quan

### Dự án làm gì

Dự án mở webcam, chạy YOLO để nhận diện vật thể theo thời gian thực và hiển thị kết quả trực tiếp trên cửa sổ OpenCV.

Luồng cơ bản:

```text
Webcam -> OpenCV -> YOLO -> Python -> Bounding boxes -> Camera window
```

### Tính năng chính

- Chạy local trên desktop, không phụ thuộc web UI
- Có 2 entrypoint camera:
  - `run_app.py`
  - `run_detect.py`
- Có 4 mode runtime:
  - `auto`
  - `high`
  - `medium`
  - `low`
- Có dashboard terminal hiển thị:
  - CPU
  - RAM
  - GPU
  - VRAM
  - PyTorch
  - CUDA
- Có fallback runtime và fallback model
- Có pipeline train riêng
- Có test dashboard `run_tests.py`
- Có chức năng bấm `T` để chụp mẫu train

### Camera hiển thị gì

Trên cửa sổ camera hiện có:

- bounding box
- tên class
- confidence
- tọa độ box ở góc trái dưới của box
- màu khác nhau theo từng label

### Kích thước camera hiện tại

Cả 4 mode hiện dùng chung:

- `1200 x 750`

### Chọn model local như thế nào

Việc load model được xử lý bởi `core/model_loader.py` và `config/model_config.yaml`.

Thứ tự ưu tiên hiện tại:

1. `models/trained/best.pt`
2. model trong `models/pretrained/`
3. file model local cùng tên ở thư mục gốc

---

## 2. Cài đặt môi trường

### Yêu cầu

- Windows + PowerShell
- Python `3.10+`
- Webcam hoạt động bình thường
- Nếu muốn dùng GPU NVIDIA: cần PyTorch bản CUDA phù hợp

### Tạo và kích hoạt môi trường

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Cài thư viện

```powershell
pip install -r requirements.txt
```

### Cài PyTorch CPU

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

### Cài PyTorch CUDA

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

### Kiểm tra PyTorch và CUDA

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Nếu `torch.cuda.is_available()` là `False` thì môi trường hiện tại chưa chạy CUDA được.

### Thư mục được tạo tự động

Khi chạy app hoặc training, `utils/file_utils.py` sẽ đảm bảo có sẵn:

- `dataset/raw/images`
- `dataset/raw/labels`
- `dataset/processed/images/train`
- `dataset/processed/images/val`
- `dataset/processed/images/test`
- `dataset/processed/labels/train`
- `dataset/processed/labels/val`
- `dataset/processed/labels/test`
- `dataset/sample/images`
- `dataset/sample/labels`
- `models/pretrained`
- `models/trained`
- `models/exported`
- `output/logs`
- `output/screenshots`
- `output/videos`
- `runs/train`
- `runs/detect`
- `runs/val`

---

## 3. Cách chạy dự án

### Chạy app camera chính

```powershell
.\.venv\Scripts\python run_app.py
```

### Chạy detect camera kiểu CLI

```powershell
.\.venv\Scripts\python run_detect.py
```

### Chạy với mode cố định

```powershell
.\.venv\Scripts\python run_app.py --mode high
.\.venv\Scripts\python run_app.py --mode medium
.\.venv\Scripts\python run_detect.py --mode low
```

### Chạy với camera index khác

```powershell
.\.venv\Scripts\python run_app.py --camera-index 1
.\.venv\Scripts\python run_detect.py --camera-index 1
```

### `run_app.py` và `run_detect.py` khác nhau gì

| File | Vai trò |
|---|---|
| `run_app.py` | Entry point desktop chính |
| `run_detect.py` | Entry point detect kiểu CLI |

Cả hai đều:

- gọi `detect_hardware()`
- gọi `select_runtime_config()`
- in dashboard terminal
- gọi `run_camera_session()` trong `core/camera_runner.py`

### Camera session làm gì

`core/camera_runner.py` chịu trách nhiệm:

1. nhận runtime hiện tại
2. load model local
3. mở webcam
4. đọc frame liên tục
5. chạy `model.predict(...)`
6. vẽ box và label lên frame
7. fallback nếu inference lỗi

### Chụp mẫu train bằng phím `T`

Khi camera đang chạy:

- bấm `T` để vào chế độ chụp mẫu
- hệ thống kiểm tra độ ổn định khung hình trong `5` giây
- nếu rung/lắc vượt ngưỡng, bộ đếm sẽ reset
- đủ ổn định thì hiện cửa sổ `YOLO Capture Assistant`
- nhập tên mẫu
  - `Enter` để lưu
  - `Backspace` để xóa
  - `Esc` để hủy

Dữ liệu sẽ lưu vào:

- `dataset/sample/images/`
- `dataset/sample/labels/`

Lưu ý:

- `dataset/sample/` là nơi gom mẫu nhanh
- không phải nguồn train chính thức

### Chạy test toàn bộ hệ thống

```powershell
.\.venv\Scripts\python run_tests.py
```

Trạng thái hiện tại:

- `45 / 45 PASS`

### Các lệnh nhanh

| Tác vụ | Lệnh |
|---|---|
| Chạy app camera | `.\.venv\Scripts\python run_app.py` |
| Chạy detect CLI | `.\.venv\Scripts\python run_detect.py` |
| Chạy train | `.\.venv\Scripts\python run_train.py` |
| Chạy test | `.\.venv\Scripts\python run_tests.py` |
| Kiểm tra dataset raw | `.\.venv\Scripts\python training/validate_dataset.py` |
| Chia dataset | `.\.venv\Scripts\python training/split_dataset.py` |
| Validate model | `.\.venv\Scripts\python training/validate_model.py` |
| Export model | `.\.venv\Scripts\python training/export_model.py` |

---

## 4. Hướng dẫn huấn luyện

### Dữ liệu train lấy ở đâu

Nguồn train chính thức là:

- `dataset/raw/images/`
- `dataset/raw/labels/`

Không train trực tiếp từ `dataset/sample/`.

### Cấu trúc dataset YOLO

Mỗi ảnh phải có file `.txt` cùng tên.

Ví dụ:

```text
dataset/
`-- raw/
    |-- images/
    |   |-- frame_001.jpg
    |   `-- frame_002.jpg
    `-- labels/
        |-- frame_001.txt
        `-- frame_002.txt
```

Format mỗi dòng trong label:

```text
<class_id> <x_center> <y_center> <width> <height>
```

### Bước 1: tạo sẵn thư mục dataset

```powershell
.\.venv\Scripts\python training/prepare_dataset.py
```

### Bước 2: kiểm tra dataset raw

```powershell
.\.venv\Scripts\python training/validate_dataset.py
```

Script này sẽ báo:

- ảnh thiếu label
- label lỗi format
- label mồ côi
- label rỗng

### Bước 3: chia train / val / test

```powershell
.\.venv\Scripts\python training/split_dataset.py
```

Script sẽ:

- đọc từ `dataset/raw/`
- chia dữ liệu sang `dataset/processed/`
- dùng tỷ lệ:
  - `train = 70%`
  - `val = 15%`
  - `test = 15%`
- xóa dữ liệu cũ trong `dataset/processed/` trước khi chia lại

### Bước 4: cập nhật class trong `training/data.yaml`

Ví dụ:

```yaml
path: ../dataset/processed
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: car
  2: motorbike
  3: helmet
```

### Bước 5: kiểm tra `training/train_config.yaml`

Các trường quan trọng:

- `model`
- `fallback_model`
- `epochs`
- `imgsz`
- `batch`
- `device`
- `project`
- `name`

### Bước 6: chạy train

```powershell
.\.venv\Scripts\python run_train.py
```

Luồng train:

1. đọc `training/train_config.yaml`
2. kiểm tra `dataset/processed/images/train` và `val`
3. train với model chính
4. nếu lỗi thì fallback model nhẹ hơn
5. copy `best.pt` về `models/trained/best.pt`

### Bước 7: validate model

```powershell
.\.venv\Scripts\python training/validate_model.py
```

### Bước 8: export ONNX

```powershell
.\.venv\Scripts\python training/export_model.py
```

### Quy trình đầy đủ

```powershell
.\.venv\Scripts\python training/prepare_dataset.py
.\.venv\Scripts\python training/validate_dataset.py
.\.venv\Scripts\python training/split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training/validate_model.py
.\.venv\Scripts\python training/export_model.py
```

### Sau khi train xong app có tự dùng model mới không

Có.

`core/model_loader.py` luôn ưu tiên:

- `models/trained/best.pt`

---

## 5. Cấu trúc thư mục và file

### Cây thư mục chính

```text
YOLO/
|-- app/
|-- config/
|-- core/
|-- dataset/
|-- docs/
|-- models/
|-- output/
|-- runs/
|-- tests/
|-- training/
|-- utils/
|-- README.md
|-- requirements.txt
|-- run_app.py
|-- run_detect.py
|-- run_train.py
`-- run_tests.py
```

### Giải thích thư mục

| Thư mục | Vai trò |
|---|---|
| `app/` | Entry helper cho app camera |
| `config/` | Toàn bộ YAML cấu hình |
| `core/` | Lõi detect realtime và runtime |
| `dataset/` | Dữ liệu raw, processed, sample |
| `docs/` | Tài liệu phụ |
| `models/` | Model pretrained, trained, exported |
| `output/` | Output phụ của hệ thống |
| `runs/` | Artifact do Ultralytics sinh ra |
| `tests/` | Test tự động |
| `training/` | Pipeline train |
| `utils/` | Helper dùng chung |

### `dataset/`

| Đường dẫn | Vai trò |
|---|---|
| `dataset/raw/` | Nguồn dữ liệu chính thức để train |
| `dataset/processed/` | Dataset đã chia train/val/test |
| `dataset/sample/` | Mẫu chụp nhanh từ camera |

### `models/`

| Đường dẫn | Vai trò |
|---|---|
| `models/pretrained/` | Model local sẵn có |
| `models/trained/` | Model train xong, quan trọng nhất là `best.pt` |
| `models/exported/` | Nơi dành cho model export |

### `core/`

| File | Vai trò |
|---|---|
| `core/hardware_info.py` | Đọc CPU, RAM, GPU, VRAM, CUDA |
| `core/model_selector.py` | Chọn runtime theo mode và phần cứng |
| `core/model_loader.py` | Load model local |
| `core/fallback_manager.py` | Sinh chuỗi fallback runtime |
| `core/camera_runner.py` | Chạy camera realtime |

### `training/`

| File | Vai trò |
|---|---|
| `training/prepare_dataset.py` | Tạo sẵn cấu trúc dataset |
| `training/validate_dataset.py` | Kiểm tra dataset raw |
| `training/split_dataset.py` | Chia train/val/test |
| `training/train_model.py` | Logic train chính |
| `training/validate_model.py` | Validate model |
| `training/export_model.py` | Export ONNX |
| `training/model_paths.py` | Resolve đường dẫn model và data |
| `training/_training_bootstrap.py` | Bootstrap import path |
| `training/train_config.yaml` | Hyperparameter train |
| `training/data.yaml` | Dataset config cho YOLO |

### `utils/`

| File | Vai trò |
|---|---|
| `utils/file_utils.py` | Tạo thư mục, đọc/ghi YAML |
| `utils/logger.py` | Logger dùng chung |
| `utils/console_ui.py` | Prompt mode và dashboard terminal |
| `utils/draw_utils.py` | Vẽ detect lên frame |

---

## 6. Khắc phục lỗi nhanh

### Có GPU nhưng vẫn chạy CPU

Kiểm tra:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### Không mở được webcam

- thử `--camera-index 1`
- đóng ứng dụng khác đang dùng webcam
- kiểm tra webcam ở ứng dụng khác

### Camera lag

- thử `--mode medium`
- nếu vẫn lag, chuyển `--mode low`

### Không detect được vật thể

- tăng ánh sáng
- đưa vật thể gần camera hơn
- kiểm tra model đang dùng
- nếu dùng model tự train, xác nhận `models/trained/best.pt` là file đúng

### Lỗi khi train

- kiểm tra `training/data.yaml`
- kiểm tra `dataset/raw/` đã có ảnh và label chưa
- chạy `training/validate_dataset.py`
- chạy `training/split_dataset.py`
- nếu thiếu VRAM, giảm `imgsz` và `batch`

### Lỗi không load được model local

Kiểm tra:

- `models/pretrained/`
- `models/trained/best.pt`
- `config/model_config.yaml`

---

## Ghi chú

- Nên chạy toàn bộ dự án bằng Python trong `.venv`
- Nếu muốn dùng GPU NVIDIA, cần cài đúng bản PyTorch có CUDA
- Sau khi train xong, app sẽ tự ưu tiên `models/trained/best.pt`
- Trạng thái hiện tại của test hệ thống: `45/45 PASS`
