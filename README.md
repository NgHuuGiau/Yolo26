# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![YAML](https://img.shields.io/badge/YAML-Config-CB171E?logo=yaml&logoColor=white)](https://yaml.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Ứng dụng nhận diện vật thể realtime bằng `YOLO`, chạy thuần `Python + OpenCV`, có menu cấu hình trong terminal, có cơ chế fallback theo phần cứng, và có sẵn pipeline train/validate/export model.

---

## Mục lục

- [1. Giới thiệu dự án](#1-giới-thiệu-dự-án)
- [2. Tạo và cài đặt](#2-tạo-và-cài-đặt)
- [3. Hướng dẫn chạy](#3-hướng-dẫn-chạy)
- [4. Giải thích từng thư mục](#4-giải-thích-từng-thư-mục)

---

## 1. Giới thiệu dự án

### 1.1. Dự án này dùng để làm gì

Dự án mở webcam, chạy nhận diện vật thể realtime bằng `YOLO`, sau đó hiển thị kết quả bằng cửa sổ camera OpenCV.

Luồng hoạt động thực tế:

```text
Webcam -> OpenCV -> YOLO -> Python -> Bounding Box -> Cửa sổ camera
```

### 1.2. Trạng thái hiện tại của dự án

Những gì hiện có trong mã nguồn:

- chạy thuần desktop, không dùng web UI
- không dùng Streamlit
- mở trực tiếp cửa sổ camera bằng OpenCV
- có menu chọn cấu hình ngay trong terminal
- có dashboard terminal hiển thị CPU, RAM, GPU, VRAM, PyTorch, CUDA và runtime thực tế
- có fallback khi runtime chính không chạy được
- có model local preload sẵn trong `models/pretrained/`
- có pipeline train, validate, export
- có bộ test hệ thống riêng trong terminal

Những gì hiện không còn dùng:

- Streamlit app
- giao diện trình duyệt
- hiển thị FPS / model / labels trên khung hình camera

### 1.3. Khung hình camera đang hiển thị gì

Trên khung hình camera hiện tại chỉ còn:

- `bounding box`

Không còn hiển thị:

- FPS
- Device
- Model
- imgsz
- Labels

### 1.4. Các mode chạy hiện có

| Phím | Mode | Ý nghĩa thực tế |
|---|---|---|
| `1` | `auto` | Tự kiểm tra phần cứng và chọn runtime phù hợp |
| `2` | `high` | Ưu tiên chất lượng, nặng hơn |
| `3` | `medium` | Cân bằng giữa tốc độ và độ chính xác |
| `4` | `low` | Ưu tiên mượt và ổn định |
| `0` | `exit` | Thoát ngay từ menu |

Khuyến nghị cho máy `RTX 3050 Ti 4GB`:

- chọn `3 = Trung bình`

### 1.5. Logic chọn runtime trong dự án

Hệ thống tách rõ:

- `Mục tiêu`: cấu hình bạn yêu cầu
- `Thực tế`: cấu hình máy hiện tại thật sự chạy được

Ví dụ:

- bạn chọn `2 = Cao nhất`
- nhưng `.venv` đang là `torch +cpu`
- hệ thống sẽ không cố ép GPU
- terminal sẽ hiện rõ đang rơi về `CPU fallback`

### 1.6. Dashboard terminal hiện hiển thị gì

Khi chạy `run_app.py` hoặc `run_detect.py`, terminal hiện:

- `Lua chon`
- `Muc tieu`
- `Thuc te`
- `Thanh khoi dong`
- `CPU`
- `RAM / OS`
- `GPU`
- `VRAM / GPU count`
- `Torch / build`
- `CUDA runtime`
- `Ly do CUDA`
- `Model dang chay`
- `Model CUDA`
- `imgsz / max_det`
- `Camera / Index`
- `Half precision`
- `Trang thai`

Màu trạng thái trong terminal:

- `xanh lá`: đang chạy đúng hoặc dùng được thật
- `vàng`: trạng thái trung gian, fallback, hoặc có phần cứng nhưng chưa tận dụng hết
- `đỏ`: không dùng được

### 1.7. Model local đang có trong dự án

Hiện thư mục `models/pretrained/` có:

- `yolo26n.pt`
- `yolo26s.pt`
- `yolo11n.pt`
- `yolo11s.pt`

Thứ tự ưu tiên load model trong code được điều khiển bởi `config/model_config.yaml` và hiện ưu tiên:

1. `models/trained/best.pt`
2. `models/pretrained/yolo26s.pt`
3. file local cùng tên model
4. `yolo11s.pt`
5. `yolov8s.pt`

---

## 2. Tạo và cài đặt

### 2.1. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 2.2. Kích hoạt môi trường

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2.3. Cài thư viện cơ bản

```powershell
pip install -r requirements.txt
```

### 2.4. Danh sách thư viện hiện có trong `requirements.txt`

#### Runtime

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

#### Hỗ trợ training / xử lý dữ liệu

- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`

### 2.5. Cài PyTorch theo loại máy

#### Máy CPU-only

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

#### Máy có GPU NVIDIA

Khuyến nghị thực dụng cho dự án này:

- `torch 2.10.0`
- `torchvision 0.25.0`
- `torchaudio 2.10.0`
- `CUDA 12.6`

```powershell
.\.venv\Scripts\python -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
```

### 2.6. Kiểm tra PyTorch và CUDA

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Nếu kết quả là:

- `torch ... +cpu`
- `torch.version.cuda = None`
- `torch.cuda.is_available() = False`

thì môi trường hiện tại đang là `CPU-only`, dù máy vẫn có GPU NVIDIA thật.

### 2.7. Thư mục được tạo tự động trong dự án

Code hiện tại có hàm `ensure_project_directories()` trong `utils/file_utils.py`, dùng để đảm bảo các thư mục sau tồn tại:

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

## 3. Hướng dẫn chạy

### 3.1. Chạy desktop chính

```powershell
.\.venv\Scripts\python run_app.py
```

Đây là entrypoint desktop chính của dự án.

### 3.2. Chạy detect kiểu CLI

```powershell
.\.venv\Scripts\python run_detect.py
```

Đây là bản chạy nhanh kiểu dòng lệnh, vẫn dùng chung lõi detect.

### 3.3. Chạy với mode cố định

```powershell
.\.venv\Scripts\python run_app.py --mode high
.\.venv\Scripts\python run_detect.py --mode medium
```

### 3.4. Chạy camera index khác

```powershell
.\.venv\Scripts\python run_detect.py --camera-index 1
```

### 3.5. `run_app.py` và `run_detect.py` khác nhau gì

| File | Vai trò |
|---|---|
| `run_app.py` | Entry point desktop chính, phù hợp để dùng như ứng dụng mặc định |
| `run_detect.py` | Entry point detect kiểu CLI, phù hợp khi muốn chạy nhanh hoặc test gọn |

Hiện tại cả hai đều:

- gọi `detect_hardware()`
- gọi `select_runtime_config()`
- hiển thị dashboard terminal
- dùng `run_camera_session()` trong `core/camera_detector.py`

### 3.6. Thoát camera

- bấm phím `Esc`

### 3.7. Chạy huấn luyện

#### Bước 1: chuẩn bị dữ liệu

Đặt dữ liệu gốc vào:

- `dataset/raw/images/`
- `dataset/raw/labels/`

#### Bước 2: chia dataset

```powershell
.\.venv\Scripts\python training/split_dataset.py
```

Script này:

- đọc ảnh từ `dataset/raw/images/`
- đọc nhãn tương ứng từ `dataset/raw/labels/`
- chia theo tỷ lệ:
  - `train = 70%`
  - `val = 15%`
  - `test = 15%`
- copy sang:
  - `dataset/processed/images/train`
  - `dataset/processed/images/val`
  - `dataset/processed/images/test`
  - `dataset/processed/labels/train`
  - `dataset/processed/labels/val`
  - `dataset/processed/labels/test`

#### Bước 3: train model

```powershell
.\.venv\Scripts\python run_train.py
```

Luồng train thực tế trong `training/train_yolo.py`:

- đọc cấu hình từ `training/train_config.yaml`
- tạo model theo trường `model`
- gọi `model.train(**config)`
- nếu cấu hình chính thất bại:
  - đổi sang `fallback_model`
  - hạ `imgsz`
  - hạ `batch`
- sau khi train xong:
  - copy `weights/best.pt`
  - lưu về `models/trained/best.pt`

#### Bước 4: validate model

```powershell
.\.venv\Scripts\python training/validate_model.py
```

Script này:

- ưu tiên dùng `models/trained/best.pt`
- nếu chưa có thì fallback sang `yolo11n.pt`
- chạy `model.val(...)`
- lưu kết quả vào `runs/val`

#### Bước 5: export model

```powershell
.\.venv\Scripts\python training/export_model.py
```

Script này:

- yêu cầu phải có `models/trained/best.pt`
- load model đó
- export ra định dạng `ONNX`

### 3.8. Chạy kiểm thử toàn hệ thống

```powershell
.\.venv\Scripts\python run_tests.py
```

Trạng thái hiện tại:

- `37 tests`
- `37 pass`

Dashboard test terminal hiện có:

- nhóm theo module
- chạy tuần tự từng test
- `PASS / FAIL / ERROR / SKIP`
- progress bar dạng `█ █ █`
- tổng kết cuối phiên

### 3.9. Lệnh nhanh

| Tác vụ | Lệnh |
|---|---|
| Chạy desktop chính | `.\.venv\Scripts\python run_app.py` |
| Chạy detect CLI | `.\.venv\Scripts\python run_detect.py` |
| Chạy detect mode trung bình | `.\.venv\Scripts\python run_detect.py --mode medium` |
| Chạy app với camera khác | `.\.venv\Scripts\python run_app.py --camera-index 1` |
| Chạy train | `.\.venv\Scripts\python run_train.py` |
| Validate model | `.\.venv\Scripts\python training/validate_model.py` |
| Export model | `.\.venv\Scripts\python training/export_model.py` |
| Chạy test hệ thống | `.\.venv\Scripts\python run_tests.py` |

---

## 4. Giải thích từng thư mục

### 4.1. Cây thư mục chính

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

### 4.2. Vai trò từng thư mục và file

| Thư mục / file | Tác dụng thực tế trong dự án |
|---|---|
| `.venv/` | Môi trường ảo, chứa toàn bộ thư viện Python của dự án |
| `app/` | Thành phần hỗ trợ tầng ứng dụng, hiện có `camera_view.py` |
| `config/` | Chứa `settings.yaml`, `model_config.yaml`, `camera_config.yaml` |
| `core/` | Lõi dự án: phần cứng, runtime, detector, fallback, loader |
| `dataset/` | Dữ liệu gốc và dữ liệu đã chia cho train/val/test |
| `docs/` | Tài liệu kỹ thuật phụ của dự án |
| `models/` | Model local preload, model đã train, model export |
| `output/` | Log và output phụ của hệ thống |
| `runs/` | Artifact do Ultralytics sinh ra trong train/val/detect |
| `tests/` | Toàn bộ test của hệ thống |
| `training/` | Script huấn luyện, chia dataset, validate, export |
| `utils/` | Hàm tiện ích như YAML, logger, prompt terminal, visualization |
| `LICENSE` | Giấy phép MIT |
| `README.md` | Tài liệu mô tả và hướng dẫn sử dụng dự án |
| `requirements.txt` | Danh sách thư viện cần cài |
| `run_app.py` | Entry point desktop chính |
| `run_detect.py` | Entry point detect kiểu CLI |
| `run_tests.py` | Entry point kiểm thử toàn hệ thống |
| `run_train.py` | Entry point huấn luyện model |

### 4.3. Một số file quan trọng trong `core/`

| File | Vai trò |
|---|---|
| `hardware_detector.py` | Đọc CPU, RAM, GPU, VRAM, PyTorch, CUDA |
| `model_selector.py` | Chọn runtime theo mode và phần cứng |
| `camera_detector.py` | Mở webcam, chạy YOLO, fallback khi lỗi |
| `yolo_loader.py` | Load model local theo thứ tự ưu tiên |
| `fallback_manager.py` | Tạo chuỗi fallback runtime |

### 4.4. Một số file quan trọng trong `training/`

| File | Vai trò |
|---|---|
| `split_dataset.py` | Chia dữ liệu gốc thành train / val / test |
| `train_yolo.py` | Huấn luyện model YOLO và copy `best.pt` |
| `validate_model.py` | Validate model đã train hoặc model fallback |
| `export_model.py` | Export model sang ONNX |
| `train_config.yaml` | Cấu hình train |
| `data.yaml` | Cấu hình dataset cho Ultralytics |

---

## Khắc phục lỗi nhanh

### Có GPU nhưng vẫn chạy CPU

Nguyên nhân thường gặp:

- `.venv` đang dùng `torch +cpu`
- PyTorch không có CUDA build
- cài sai bản `torch`

Kiểm tra:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### Không mở được webcam

- đổi `--camera-index`
- đóng ứng dụng khác đang dùng webcam

### Camera lag

- chọn `3 = Trung bình`
- nếu vẫn lag, chọn `4 = Yếu`

### Không detect được vật thể

- tăng ánh sáng
- đưa vật thể lại gần camera
- kiểm tra model local

### Lỗi model local

Kiểm tra:

- `models/pretrained/`
- `models/trained/best.pt`

---

## Ghi chú

- nên chạy toàn bộ dự án bằng `.venv`
- nếu muốn dùng GPU NVIDIA, cần cài đúng `PyTorch + CUDA`, không dùng bản `+cpu`
- với máy `RTX 3050 Ti 4GB`, mode hợp lý nhất để dùng hàng ngày là `3 = Trung bình`
- nội dung README này chỉ mô tả những gì đang thực sự có trong mã nguồn hiện tại của dự án
