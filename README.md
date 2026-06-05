# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Windows](https://img.shields.io/badge/Windows-11%2B-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![PowerShell](https://img.shields.io/badge/PowerShell-7%2B-5391FE?logo=powershell&logoColor=white)](https://learn.microsoft.com/powershell/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-8CAAE6)](https://python-pillow.org/)
[![YAML](https://img.shields.io/badge/YAML-Config-CB171E?logo=yaml&logoColor=white)](https://yaml.org/)

## 1. Giới thiệu dự án

Đây là dự án nhận diện vật thể realtime bằng webcam, chạy trên desktop Windows với Python, OpenCV, Ultralytics YOLO11 và PyTorch.

Dự án hiện có:

- giao diện terminal tiếng Việt
- tự dò cấu hình máy trước khi chạy
- tự gợi ý mức phù hợp theo `CPU`, `RAM`, `GPU`, `VRAM`, `PyTorch`, `CUDA`
- hỗ trợ 5 model `YOLO11`
- hỗ trợ chụp mẫu train trực tiếp từ webcam bằng phím `T`
- pipeline huấn luyện, kiểm tra và export model riêng
- menu tổng để chạy nhanh các file chính
- bộ test để kiểm tra toàn hệ thống

### Ngôn ngữ và thư viện đang dùng

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

### 5 model YOLO11 đang dùng

- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

### 3 mức người dùng sẽ thấy khi chạy camera

- `Cao nhất`
- `Trung bình`
- `Yếu`

Hệ thống không ép model lớn nhất bằng mọi giá. Nó sẽ cố chọn mức cao nhất mà máy còn chạy ổn định.

Ví dụ:

- máy rất mạnh, VRAM lớn -> có thể lên `yolo11x.pt`
- máy mạnh -> có thể lên `yolo11l.pt`
- máy tầm trung như `RTX 3050 Ti 4GB` -> thường hợp `yolo11s.pt` hoặc `yolo11m.pt`
- máy yếu hoặc CPU-only -> thường về `yolo11n.pt`

### Bảng chọn model theo máy

| Điều kiện phần cứng | High | Medium | Low |
|---|---|---|---|
| GPU >= 12GB VRAM | `yolo11x.pt` / GPU / `imgsz 960` | `yolo11m.pt` / GPU / `imgsz 768` | `yolo11n.pt` / GPU / `imgsz 416` |
| GPU >= 8GB VRAM | `yolo11l.pt` / GPU / `imgsz 896` | `yolo11m.pt` / GPU / `imgsz 768` | `yolo11n.pt` / GPU / `imgsz 416` |
| GPU >= 4GB VRAM | `yolo11m.pt` / GPU / `imgsz 768` | `yolo11s.pt` / GPU / `imgsz 640` | `yolo11n.pt` / GPU / `imgsz 416` |
| GPU 3GB - dưới 4GB VRAM | `yolo11s.pt` / GPU / `imgsz 640` | `yolo11s.pt` / GPU / `imgsz 640` | `yolo11n.pt` / GPU / `imgsz 416` |
| GPU dưới 3GB VRAM | `yolo11n.pt` / GPU / `imgsz 512` | `yolo11n.pt` / GPU / `imgsz 512` | `yolo11n.pt` / CPU / `imgsz 320` |
| Không có CUDA / CPU-only | `yolo11n.pt` hoặc `yolo11s.pt` / CPU / `imgsz 416` | `yolo11n.pt` / CPU / `imgsz 416` | `yolo11n.pt` / CPU / `imgsz 320` |

### Lưu ý quan trọng

- repo này chỉ chứa code
- repo này không kèm model `.pt`
- repo này không kèm dataset
- bạn phải tự đặt model vào `models/pretrained/`

## 2. Cách cài

### 2.1. Yêu cầu

- Windows + PowerShell
- Python `3.10+`
- webcam hoạt động bình thường nếu muốn dùng camera
- nếu muốn chạy GPU NVIDIA thì cần PyTorch CUDA phù hợp

### 2.2. Vào thư mục dự án

```powershell
cd D:\YOLO
```

Nếu bạn clone từ Git trước đó thì có thể là:

```powershell
git clone <repo-url>
cd D:\YOLO
```

### 2.3. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 2.4. Kích hoạt môi trường

```powershell
.\.venv\Scripts\Activate.ps1
```

Khi thành công, bạn sẽ thấy:

```powershell
(.venv) PS D:\YOLO>
```

### 2.5. Cài thư viện Python

```powershell
pip install -r requirements.txt
```

### 2.6. Cài PyTorch

**Nếu chỉ chạy CPU**

```powershell
pip install torch torchvision torchaudio
```

**Nếu chạy NVIDIA CUDA**

Ví dụ CUDA 12.6:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Nếu máy bạn dùng bản CUDA khác, hãy chọn đúng lệnh ở trang chính thức của PyTorch.

### 2.7. Kiểm tra PyTorch và CUDA

```powershell
.\.venv\Scripts\python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

Ý nghĩa:

- `Torch:` phiên bản PyTorch đang dùng
- `CUDA: True` nghĩa là PyTorch đã thấy CUDA
- `GPU count:` số GPU PyTorch nhìn thấy

### 2.8. Tạo cấu trúc thư mục dataset

```powershell
.\.venv\Scripts\python training\prepare_dataset.py
```

Lệnh này chỉ tạo sẵn thư mục:

- `dataset/raw`
- `dataset/processed`
- `dataset/sample`

### 2.9. Cài 5 model YOLO11

Bạn có 2 cách.

**Cách 1: dùng script của dự án**

```powershell
.\.venv\Scripts\python training\download_models.py
```

Lệnh này sẽ tải đủ:

- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

vào:

- `models/pretrained/`

**Cách 2: tải thủ công bằng PowerShell**

```powershell
New-Item -ItemType Directory -Force -Path .\models\pretrained | Out-Null

Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt" -OutFile ".\models\pretrained\yolo11n.pt"
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt" -OutFile ".\models\pretrained\yolo11s.pt"
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt" -OutFile ".\models\pretrained\yolo11m.pt"
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt" -OutFile ".\models\pretrained\yolo11l.pt"
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt" -OutFile ".\models\pretrained\yolo11x.pt"
```

### 2.10. Kiểm tra model đã cài đủ chưa

```powershell
Get-ChildItem .\models\pretrained\yolo11*.pt
```

Bạn nên thấy đủ 5 file:

- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

### 2.11. Kiểm tra sức khỏe hệ thống

```powershell
.\.venv\Scripts\python run_doctor.py
```

Lệnh này sẽ kiểm tra:

- CPU, RAM
- GPU, VRAM
- PyTorch, CUDA
- có đủ model chưa
- có dữ liệu raw chưa
- đã có dữ liệu split chưa

### 2.12. Chạy test toàn hệ thống

```powershell
.\.venv\Scripts\python run_tests.py
```

## 3. Cách chạy

### 3.1. Mở menu tổng

Nếu bạn muốn có menu để chọn nhanh file cần chạy:

```powershell
.\.venv\Scripts\python run_menu.py
```

Menu hiện có:

- `1` chạy `run_app.py`
- `2` chạy `run_detect.py`
- `3` chạy `run_train.py`
- `4` chạy `run_tests.py`
- `5` chạy `run_doctor.py`
- `0` thoát

### 3.2. Chạy app camera chính

```powershell
.\.venv\Scripts\python run_app.py
```

App sẽ:

- dò cấu hình máy trước
- gợi ý mức chạy phù hợp
- cho bạn chọn `Cao nhất`, `Trung bình`, `Yếu`
- tự nạp model YOLO11 phù hợp với máy

### 3.3. Chạy detect camera

```powershell
.\.venv\Scripts\python run_detect.py
```

Lệnh này tương tự `run_app.py`, nhưng dành cho luồng detect trực tiếp.

### 3.4. Chạy kiểm tra sức khỏe nhanh

```powershell
.\.venv\Scripts\python run_doctor.py
```

Nên chạy lệnh này khi:

- mới cài xong
- nghi ngờ thiếu model
- nghi ngờ CUDA không hoạt động
- muốn biết máy nên chạy mức nào

### 3.5. Chạy test hệ thống

```powershell
.\.venv\Scripts\python run_tests.py
```

### 3.6. Ý nghĩa màu trong terminal

- `xanh lá` = chạy tốt, sẵn sàng
- `vàng` = cảnh báo, 50/50, cần chú ý
- `đỏ` = lỗi hoặc không đủ điều kiện để chạy

### 3.7. Khi chạy camera có thể làm gì

- quan sát kết quả detect realtime
- bấm `T` để chụp mẫu train
- mẫu chụp sẽ được lưu vào `dataset/sample/`

## 4. Huấn luyện

### 4.1. Hiểu đúng 3 thư mục dữ liệu

**`dataset/sample/`**

- nơi lưu ảnh và label khi bạn bấm `T` trong app
- đây là mẫu chụp từ camera

**`dataset/raw/`**

- nơi chứa dữ liệu gốc chính thức để train
- nếu bạn đã có sẵn ảnh và file `.txt`, hãy bỏ thẳng vào đây

**`dataset/processed/`**

- nơi chứa dữ liệu đã được hệ thống chia thành:
  - `train`
  - `val`
  - `test`
- model sẽ train từ đây, không train trực tiếp từ `raw`

Luồng dữ liệu đúng:

```text
sample -> raw -> processed -> train
```

### 4.2. Trường hợp nào dùng thư mục nào

**Nếu bạn chụp từ app bằng phím `T`**

- dữ liệu nằm ở `dataset/sample/`
- nếu muốn đưa vào pipeline train chính thì chạy:

```powershell
.\.venv\Scripts\python training\promote_samples.py
```

Lệnh này sẽ copy:

- `dataset/sample/images` -> `dataset/raw/images`
- `dataset/sample/labels` -> `dataset/raw/labels`

**Nếu bạn đã có sẵn ảnh + txt**

- bỏ thẳng vào:
  - `dataset/raw/images/`
  - `dataset/raw/labels/`

### 4.3. Quy tắc tên file ảnh và label

Tên ảnh và file `.txt` phải trùng nhau.

Ví dụ đúng:

```text
dataset/raw/images/anh_001.jpg
dataset/raw/labels/anh_001.txt
```

Ví dụ sai:

```text
dataset/raw/images/anh_001.jpg
dataset/raw/labels/anh_002.txt
```

### 4.4. Format file label YOLO

Mỗi dòng trong file `.txt` có dạng:

```text
class_id x_center y_center width height
```

Ví dụ một object:

```text
0 0.512 0.477 0.210 0.330
```

Ví dụ nhiều object trong một ảnh:

```text
0 0.512 0.477 0.210 0.330
1 0.221 0.610 0.150 0.240
```

Trong đó:

- `class_id` là số class
- các giá trị còn lại là tọa độ chuẩn hóa từ `0` đến `1`

### 4.5. Có dữ liệu trong `raw` rồi thì làm gì tiếp

Khi đã có dữ liệu trong:

- `dataset/raw/images/`
- `dataset/raw/labels/`

hãy chạy đúng thứ tự này:

```powershell
.\.venv\Scripts\python training\validate_dataset.py
.\.venv\Scripts\python training\split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training\validate_model.py
.\.venv\Scripts\python training\export_model.py
```

### 4.6. Giải thích 5 lệnh huấn luyện

**1. `training\validate_dataset.py`**

- kiểm tra dữ liệu trong `raw`
- phát hiện:
  - ảnh thiếu label
  - label rỗng
  - label lỗi
  - label mồ côi

**2. `training\split_dataset.py`**

- lấy dữ liệu hợp lệ từ `raw`
- chia sang:
  - `processed/images/train`
  - `processed/images/val`
  - `processed/images/test`
  - `processed/labels/train`
  - `processed/labels/val`
  - `processed/labels/test`

**3. `run_train.py`**

- train model từ dữ liệu đã chia trong `processed`
- model tốt nhất sẽ được đưa về:
  - `models/trained/best.pt`

**4. `training\validate_model.py`**

- dùng tập `val` để kiểm tra chất lượng model sau train

**5. `training\export_model.py`**

- export model sang định dạng dùng để deploy, hiện là ONNX

### 4.7. Sau khi chạy xong 5 lệnh thì làm gì nữa

Sau khi train xong, bạn mở app để dùng model mới train:

```powershell
.\.venv\Scripts\python run_app.py
```

Hệ thống sẽ ưu tiên:

- `models/trained/best.pt`

nếu file này tồn tại.

### 4.8. Tỷ lệ chia dữ liệu

Mặc định pipeline chia:

- `train` khoảng `70%`
- `val` khoảng `15%`
- `test` khoảng `15%`

### 4.9. Gợi ý theo cấu hình máy

**Máy yếu**

- nên dùng `Yếu`
- thường về `yolo11n.pt`

**Máy tầm trung**

- nên dùng `Trung bình`
- thường về `yolo11s.pt`

**Máy mạnh**

- có thể dùng `Cao nhất`
- thường lên `yolo11m.pt`, `yolo11l.pt` hoặc `yolo11x.pt` tùy VRAM

### 4.10. Lỗi thường gặp khi train

**Không có dữ liệu trong `raw`**

- nghĩa là bạn chưa bỏ ảnh và label vào đúng thư mục

**Không có dữ liệu trong `processed`**

- nghĩa là bạn chưa chạy `split_dataset.py`

**Không có `models/trained/best.pt`**

- nghĩa là train chưa xong nên chưa thể export

**Thiếu model YOLO11**

- hãy chạy:

```powershell
.\.venv\Scripts\python training\download_models.py
```

## 5. Giải thích cấu trúc thư mục

```text
YOLO/
|-- app/
|   |-- camera_app.py
|
|-- config/
|   |-- camera_config.yaml
|   |-- model_config.yaml
|   `-- settings.yaml
|
|-- core/
|   |-- camera_runner.py
|   |-- fallback_manager.py
|   |-- hardware_info.py
|   |-- model_loader.py
|   `-- model_selector.py
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
|   `-- training_guide.md
|
|-- models/
|   |-- pretrained/
|   |-- trained/
|   `-- exported/
|
|-- output/
|
|-- runs/
|
|-- tests/
|   |-- test_camera_detector.py
|   |-- test_console_ui.py
|   |-- test_doctor.py
|   |-- test_download_models.py
|   |-- test_file_utils.py
|   |-- test_hardware_detector.py
|   |-- test_model_selector.py
|   |-- test_prepare_dataset.py
|   |-- test_promote_samples.py
|   |-- test_run_entrypoints.py
|   |-- test_run_menu.py
|   |-- test_run_tests_dashboard.py
|   |-- test_runtime_prompt.py
|   |-- test_split_dataset.py
|   |-- test_training_pipeline.py
|   |-- test_visualization.py
|   `-- test_yolo_loader.py
|
|-- training/
|   |-- _training_bootstrap.py
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
`-- run_train.py
```

### Giải thích nhanh từng nhóm

`app/`

- chứa luồng app camera cấp cao
- `camera_app.py` là nơi nối phần cứng, chọn mode, terminal UI và camera runtime

`config/`

- chứa cấu hình YAML
- `camera_config.yaml` cho camera
- `model_config.yaml` cho model
- `settings.yaml` cho rule chọn cấu hình tự động

`core/`

- lõi xử lý chính của dự án
- `camera_runner.py`: chạy camera, đọc frame, detect, lưu sample
- `hardware_info.py`: đọc CPU, RAM, GPU, VRAM, CUDA
- `model_loader.py`: nạp model YOLO
- `model_selector.py`: quyết định nên dùng `n/s/m/l/x`
- `fallback_manager.py`: tạo chuỗi fallback khi cấu hình chính lỗi

`dataset/`

- chứa toàn bộ dữ liệu
- `sample/`: ảnh và label chụp trực tiếp từ app bằng phím `T`
- `raw/`: dữ liệu gốc chính thức để train
- `processed/`: dữ liệu đã chia thành `train / val / test`

`docs/`

- tài liệu phụ của dự án

`models/`

- `pretrained/`: 5 model YOLO11 tải sẵn
- `trained/`: `best.pt` sau khi train
- `exported/`: model đã export như ONNX

`output/`

- output phụ do app hoặc script tạo ra

`runs/`

- output train chuẩn của Ultralytics

`tests/`

- toàn bộ test tự động của hệ thống

`training/`

- toàn bộ pipeline huấn luyện
- `prepare_dataset.py`: tạo cấu trúc dataset
- `download_models.py`: tải 5 model YOLO11
- `promote_samples.py`: chuyển sample sang raw
- `split_dataset.py`: chia raw thành processed
- `train_model.py`: train model
- `validate_model.py`: kiểm tra model sau train
- `export_model.py`: export model
- `terminal_ui.py`: giao diện terminal cho pipeline training
- `model_paths.py`: resolve path model

`utils/`

- tiện ích chung
- `console_ui.py`: UI terminal cho app và dashboard
- `draw_utils.py`: hàm vẽ detect lên ảnh/frame
- `file_utils.py`: đọc ghi file/yaml
- `logger.py`: logging

### File chính nên nhớ

- `run_menu.py` = menu tổng
- `run_app.py` = chạy app camera chính
- `run_detect.py` = detect camera
- `run_doctor.py` = kiểm tra hệ thống
- `run_tests.py` = kiểm tra code
- `run_train.py` = train model

## 6. Chuỗi chạy ngắn gọn dễ nhớ

### Nếu chỉ muốn kiểm tra hệ thống

```powershell
.\.venv\Scripts\python run_doctor.py
```

### Nếu chỉ muốn chạy camera

```powershell
.\.venv\Scripts\python run_app.py
```

### Nếu đã có sẵn ảnh + txt để train

```powershell
.\.venv\Scripts\python training\validate_dataset.py
.\.venv\Scripts\python training\split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training\validate_model.py
.\.venv\Scripts\python training\export_model.py
.\.venv\Scripts\python run_app.py
```

### Nếu chụp bằng app rồi mới train

```powershell
.\.venv\Scripts\python training\promote_samples.py
.\.venv\Scripts\python training\validate_dataset.py
.\.venv\Scripts\python training\split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training\validate_model.py
.\.venv\Scripts\python training\export_model.py
.\.venv\Scripts\python run_app.py
```
