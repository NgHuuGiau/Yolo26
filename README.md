# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Ứng dụng nhận diện vật thể realtime bằng YOLO, chạy thuần Python + OpenCV, có menu chọn mode trong terminal, có fallback theo phần cứng, có pipeline train/validate/export, và có chức năng chụp mẫu train ngay từ camera.

---

## Mục lục

- [1. Tổng quan dự án](#1-tổng-quan-dự-án)
- [2. Cài đặt môi trường](#2-cài-đặt-môi-trường)
- [3. Cách chạy dự án](#3-cách-chạy-dự-án)
- [4. Hướng dẫn huấn luyện đầy đủ](#4-hướng-dẫn-huấn-luyện-đầy-đủ)
- [5. Giải thích thư mục và file](#5-giải-thích-thư-mục-và-file)
- [6. Khắc phục lỗi nhanh](#6-khắc-phục-lỗi-nhanh)

---

## 1. Tổng quan dự án

### 1.1. Dự án dùng để làm gì

Dự án mở webcam, chạy YOLO để nhận diện vật thể theo thời gian thực và hiển thị kết quả trực tiếp bằng cửa sổ OpenCV.

Luồng chạy:

```text
Webcam -> OpenCV -> YOLO -> Python -> Bounding boxes -> Cửa sổ camera
```

### 1.2. Những gì đang có trong mã nguồn

- Chạy desktop local, không dùng web UI
- Có 2 entrypoint detect: `run_app.py` và `run_detect.py`
- Có menu chọn mode `auto`, `high`, `medium`, `low`
- Có dashboard terminal hiển thị CPU, RAM, GPU, VRAM, PyTorch, CUDA và runtime
- Có fallback khi runtime hoặc model hiện tại không chạy được
- Có model local trong `models/pretrained/`
- Có pipeline train, validate, export
- Có bộ test hệ thống `run_tests.py`
- Có chức năng chụp mẫu train bằng phím `T`

### 1.3. Camera đang hiển thị gì

Trên khung hình camera hiện có:

- Bounding box
- Tên class
- Confidence
- Tọa độ `(x1,y1) (x2,y2)` ở phía dưới box
- Mỗi loại vật thể có màu riêng ổn định theo label

### 1.4. Chế độ và kích thước camera

Các mode hiện có:

| Phím | Mode | Ý nghĩa |
|---|---|---|
| `1` | `auto` | Tự đọc phần cứng rồi chọn profile phù hợp |
| `2` | `high` | Ưu tiên chất lượng |
| `3` | `medium` | Cân bằng giữa tốc độ và độ chính xác |
| `4` | `low` | Ưu tiên ổn định, nhẹ hơn |
| `0` | `exit` | Thoát tại menu |

Hiện tại cả 4 mode đều dùng cùng kích thước camera:

- `1280 x 800`

Ngoài cửa sổ camera chính, hệ thống mở thêm một cửa sổ phụ:

- `YOLO Capture Assistant`

Cửa sổ này dùng riêng cho đếm ngược ổn định và đặt tên mẫu train, không phụ thuộc việc bạn phóng to hay thu nhỏ camera.

### 1.5. Logic chọn runtime

Dự án tách rõ:

- `requested`: cấu hình người dùng muốn chạy
- `resolved`: cấu hình máy thực tế chạy được

Ví dụ:

- Bạn chọn `high`
- Nhưng môi trường PyTorch là `CPU-only`
- Hệ thống sẽ tự rơi về CPU profile thay vì cố ép CUDA

### 1.6. Thứ tự ưu tiên load model local

Việc load model được điều khiển bởi `config/model_config.yaml` và `core/yolo_loader.py`.

Thứ tự ưu tiên hiện tại:

1. `models/trained/best.pt`
2. `models/pretrained/yolo26s.pt`
3. file local cùng tên model ở thư mục gốc
4. `yolo11s.pt`
5. `yolov8s.pt`

---

## 2. Cài đặt môi trường

### 2.1. Yêu cầu

- Windows + PowerShell
- Python `3.10+`
- Webcam hoạt động bình thường
- Nếu muốn dùng GPU NVIDIA: cần cài đúng bản PyTorch có CUDA

### 2.2. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 2.3. Kích hoạt môi trường

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2.4. Cài thư viện

```powershell
pip install -r requirements.txt
```

### 2.5. Các thư viện chính trong `requirements.txt`

Runtime:

- `ultralytics`
- `opencv-python`
- `numpy`
- `pillow`
- `psutil`
- `GPUtil`
- `PyYAML`
- `torch`
- `torchvision`
- `torchaudio`

Training và xử lý dữ liệu:

- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`

### 2.6. Cài PyTorch theo loại máy

#### Máy chỉ dùng CPU

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

#### Máy có GPU NVIDIA

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

### 2.7. Kiểm tra PyTorch và CUDA

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Nếu kết quả là:

- version có hậu tố `+cpu`
- `torch.version.cuda = None`
- `torch.cuda.is_available() = False`

thì môi trường hiện tại là CPU-only.

### 2.8. Thư mục được tạo tự động

Mỗi lần chạy app hoặc training, `utils/file_utils.py` sẽ đảm bảo các thư mục sau tồn tại:

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
- `output/screenshots`
- `output/videos`
- `output/logs`
- `runs/train`
- `runs/detect`
- `runs/val`

---

## 3. Cách chạy dự án

### 3.1. Chạy app desktop chính

```powershell
.\.venv\Scripts\python run_app.py
```

### 3.2. Chạy detect kiểu CLI

```powershell
.\.venv\Scripts\python run_detect.py
```

### 3.3. Chạy với mode cố định

```powershell
.\.venv\Scripts\python run_app.py --mode high
.\.venv\Scripts\python run_app.py --mode medium
.\.venv\Scripts\python run_detect.py --mode low
```

### 3.4. Chạy với camera khác

```powershell
.\.venv\Scripts\python run_app.py --camera-index 1
.\.venv\Scripts\python run_detect.py --camera-index 1
```

### 3.5. `run_app.py` và `run_detect.py` khác nhau gì

| File | Vai trò |
|---|---|
| `run_app.py` | Entry point desktop chính |
| `run_detect.py` | Entry point detect dạng CLI |

Cả hai đều:

- gọi `detect_hardware()`
- gọi `select_runtime_config()`
- in dashboard terminal
- gọi `run_camera_session()` trong `core/camera_detector.py`

### 3.6. Bên trong camera session xảy ra gì

`core/camera_detector.py` làm các việc chính:

1. Chọn runtime hiện tại
2. Load model local
3. Mở webcam
4. Đọc frame liên tục
5. Gọi `model.predict(...)`
6. Vẽ detect lên frame
7. Nếu inference lỗi, tự fallback sang cấu hình an toàn hơn

### 3.7. Chụp mẫu train bằng phím `T`

Khi camera đang chạy:

- bấm `T` để bắt đầu quy trình chụp mẫu train
- hệ thống không chụp ngay, mà kiểm tra độ ổn định khung hình trong `5 giây`
- nếu phát hiện rung/lắc vượt ngưỡng, bộ đếm sẽ reset
- khi đủ ổn định, cửa sổ `YOLO Capture Assistant` chuyển sang chế độ đặt tên
- gõ tên mẫu rồi:
  - `Enter`: lưu mẫu
  - `Backspace`: xóa ký tự
  - `Esc`: hủy

Dữ liệu được lưu vào:

- ảnh: `dataset/sample/images/`
- nhãn YOLO `.txt`: `dataset/sample/labels/`

Lưu ý:

- đây là mẫu sinh từ kết quả detect hiện tại của model
- vẫn nên kiểm tra lại trước khi chuyển sang `dataset/raw/` để train thật

### 3.8. Thoát camera

- nhấn `Esc`

### 3.9. Chạy toàn bộ test hệ thống

```powershell
.\.venv\Scripts\python run_tests.py
```

Trạng thái hiện tại:

- `44 / 44 PASS`

### 3.10. Lệnh nhanh

| Tác vụ | Lệnh |
|---|---|
| Chạy app chính | `.\.venv\Scripts\python run_app.py` |
| Chạy detect CLI | `.\.venv\Scripts\python run_detect.py` |
| Chạy app mode medium | `.\.venv\Scripts\python run_app.py --mode medium` |
| Chạy detect mode low | `.\.venv\Scripts\python run_detect.py --mode low` |
| Chạy camera index 1 | `.\.venv\Scripts\python run_app.py --camera-index 1` |
| Kiểm tra dataset raw | `.\.venv\Scripts\python training/validate_dataset.py` |
| Chia dataset | `.\.venv\Scripts\python training/split_dataset.py` |
| Chạy train | `.\.venv\Scripts\python run_train.py` |
| Validate model | `.\.venv\Scripts\python training/validate_model.py` |
| Export model | `.\.venv\Scripts\python training/export_model.py` |
| Chạy test hệ thống | `.\.venv\Scripts\python run_tests.py` |

---

## 4. Hướng dẫn huấn luyện đầy đủ

### 4.1. Train thật lấy dữ liệu ở đâu

Nguồn dữ liệu chính thức để train là:

- `dataset/raw/images/`
- `dataset/raw/labels/`

Không train trực tiếp từ `dataset/sample/`.

`dataset/sample/` chỉ là nơi lưu mẫu chụp nhanh từ camera để bạn sàng lọc trước.

### 4.2. Cấu trúc dữ liệu YOLO đầu vào

Mỗi ảnh phải có file label `.txt` cùng stem.

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

### 4.3. Bước 0: tạo sẵn thư mục dataset

```powershell
.\.venv\Scripts\python training/prepare_dataset.py
```

### 4.4. Bước 1: kiểm tra dataset raw

```powershell
.\.venv\Scripts\python training/validate_dataset.py
```

Script này sẽ báo:

- ảnh thiếu label
- label lỗi format
- label mồ côi
- label rỗng

### 4.5. Bước 2: chia train / val / test

```powershell
.\.venv\Scripts\python training/split_dataset.py
```

Script này:

- đọc ảnh từ `dataset/raw/images/`
- đọc label từ `dataset/raw/labels/`
- trộn dữ liệu với seed `42`
- chia tỷ lệ:
  - `train = 70%`
  - `val = 15%`
  - `test = 15%`
- copy sang `dataset/processed/...`
- trước khi chia lại sẽ xóa sạch dữ liệu cũ trong `dataset/processed/`

### 4.6. Bước 3: cập nhật class names trong `training/data.yaml`

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

### 4.7. Bước 4: kiểm tra cấu hình train

Mở `training/train_config.yaml`.

Các trường quan trọng:

- `model`
- `fallback_model`
- `epochs`
- `imgsz`
- `batch`
- `device`
- `project`
- `name`

### 4.8. Bước 5: chạy train

```powershell
.\.venv\Scripts\python run_train.py
```

Luồng train:

1. đọc `training/train_config.yaml`
2. train với model chính
3. nếu lỗi thì fallback model nhẹ hơn
4. giảm `imgsz` và `batch` khi cần
5. copy `weights/best.pt` về `models/trained/best.pt`

### 4.9. Bước 6: validate

```powershell
.\.venv\Scripts\python training/validate_model.py
```

### 4.10. Bước 7: export ONNX

```powershell
.\.venv\Scripts\python training/export_model.py
```

### 4.11. Quy trình đầy đủ

```powershell
.\.venv\Scripts\python training/prepare_dataset.py
.\.venv\Scripts\python training/validate_dataset.py
.\.venv\Scripts\python training/split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training/validate_model.py
.\.venv\Scripts\python training/export_model.py
```

### 4.12. Sau khi train xong detect có tự dùng model mới không

Có.

Vì `core/yolo_loader.py` ưu tiên:

- `models/trained/best.pt`

---

## 5. Giải thích thư mục và file

### 5.1. Cây thư mục chính

```text
YOLO/
|-- .venv/
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
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- run_app.py
|-- run_detect.py
|-- run_tests.py
`-- run_train.py
```

### 5.2. Giải thích từng thư mục

| Thư mục | Vai trò |
|---|---|
| `.venv/` | Môi trường ảo của dự án |
| `app/` | Helper tầng ứng dụng |
| `config/` | Toàn bộ YAML cấu hình |
| `core/` | Lõi detect realtime |
| `dataset/` | Dữ liệu train, dữ liệu chia sẵn, mẫu chụp từ camera |
| `docs/` | Tài liệu phụ |
| `models/` | Model preload, model train xong, model export |
| `output/` | Output phụ của hệ thống |
| `runs/` | Artifact do Ultralytics sinh ra |
| `tests/` | Bộ test tự động |
| `training/` | Toàn bộ pipeline training |
| `utils/` | Hàm tiện ích dùng chung |

### 5.3. Giải thích `dataset/`

| Thư mục | Vai trò |
|---|---|
| `dataset/raw/` | Nguồn chính thức để train |
| `dataset/raw/images/` | Ảnh gốc |
| `dataset/raw/labels/` | Label YOLO gốc |
| `dataset/processed/` | Dữ liệu đã chia train/val/test |
| `dataset/sample/` | Mẫu chụp nhanh từ camera bằng phím `T` |
| `dataset/sample/images/` | Ảnh mẫu |
| `dataset/sample/labels/` | Label YOLO sinh từ detect hiện tại |

### 5.4. Giải thích `models/`

| Thư mục | Vai trò |
|---|---|
| `models/pretrained/` | Model local preload |
| `models/trained/` | Model sau khi train, quan trọng nhất là `best.pt` |
| `models/exported/` | Nơi dành cho model export |

### 5.5. Giải thích `runs/` và `output/`

| Thư mục | Vai trò |
|---|---|
| `runs/` | Artifact do Ultralytics sinh ra khi train/val/detect |
| `output/` | Output phụ của dự án như logs, screenshots, videos |

### 5.6. Giải thích `core/`

| File | Vai trò |
|---|---|
| `core/hardware_detector.py` | Đọc CPU, RAM, GPU, VRAM, PyTorch, CUDA |
| `core/model_selector.py` | Chọn runtime theo mode và phần cứng |
| `core/yolo_loader.py` | Load model local theo thứ tự ưu tiên |
| `core/fallback_manager.py` | Sinh chuỗi fallback runtime |
| `core/camera_detector.py` | Lõi detect realtime, chụp mẫu train, cửa sổ trợ lý |

### 5.7. Giải thích `training/`

| File | Vai trò |
|---|---|
| `training/prepare_dataset.py` | Tạo sẵn cấu trúc thư mục dataset |
| `training/validate_dataset.py` | Kiểm tra dataset raw trước khi train |
| `training/split_dataset.py` | Chia train/val/test |
| `training/train_yolo.py` | Logic train chính |
| `training/validate_model.py` | Validate model |
| `training/export_model.py` | Export ONNX |
| `training/train_config.yaml` | Hyperparameter train |
| `training/data.yaml` | Dataset config cho Ultralytics |
| `training/README_TRAINING.md` | Tài liệu ngắn riêng cho training |

### 5.8. Giải thích `utils/`

| File | Vai trò |
|---|---|
| `utils/file_utils.py` | Tạo thư mục, đọc YAML, ghi YAML |
| `utils/logger.py` | Logger dùng chung |
| `utils/runtime_prompt.py` | Menu mode, dashboard terminal |
| `utils/visualization.py` | Vẽ detect lên frame |

### 5.9. Giải thích file ở root

| File | Vai trò |
|---|---|
| `run_app.py` | Entry point desktop chính |
| `run_detect.py` | Entry point detect CLI |
| `run_train.py` | Entry point train |
| `run_tests.py` | Chạy toàn bộ test |
| `requirements.txt` | Package cần cài |
| `README.md` | Tài liệu chính |
| `LICENSE` | Giấy phép MIT |

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

- chạy `--mode medium`
- nếu vẫn lag, chuyển `--mode low`

### Không detect được vật thể

- tăng ánh sáng
- đưa vật thể gần camera hơn
- kiểm tra model hiện tại
- nếu dùng model tự train, xác nhận `models/trained/best.pt` là file đúng

### Lỗi khi train

- kiểm tra `training/data.yaml`
- kiểm tra `dataset/raw/` đã có đủ ảnh và label chưa
- chạy `training/validate_dataset.py`
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
- Sau khi train xong, detect sẽ tự ưu tiên `models/trained/best.pt`
- Hiện tại toàn bộ test hệ thống đang pass `44/44`
