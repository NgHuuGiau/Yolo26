# Hướng Dẫn Cài Đặt

## 1. Yêu cầu

- Windows + PowerShell
- Python `3.10+`
- webcam hoạt động bình thường nếu muốn dùng camera
- nếu muốn chạy GPU NVIDIA thì cần PyTorch CUDA phù hợp

## 2. Vào thư mục dự án

```powershell
cd D:\YOLO
```

Nếu bạn clone từ Git trước đó thì có thể là:

```powershell
git clone <repo-url>
cd D:\YOLO
```

## 3. Tạo môi trường ảo

```powershell
python -m venv .venv
```

## 4. Kích hoạt môi trường

```powershell
.\.venv\Scripts\Activate.ps1
```

Khi thành công, bạn sẽ thấy:

```powershell
(.venv) PS D:\YOLO>
```

## 5. Cài thư viện Python

```powershell
pip install -r requirements.txt
```

## 6. Cài PyTorch

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

## 7. Kiểm tra PyTorch và CUDA

```powershell
.\.venv\Scripts\python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

Ý nghĩa:

- `Torch:` phiên bản PyTorch đang dùng
- `CUDA: True` nghĩa là PyTorch đã thấy CUDA
- `GPU count:` số GPU PyTorch nhìn thấy

## 8. Tạo cấu trúc thư mục

```powershell
.\.venv\Scripts\python training\prepare_dataset.py
```

Lệnh này tạo toàn bộ thư mục cần thiết:

- `dataset/raw/images`, `dataset/raw/labels`
- `dataset/processed/images/{train,val,test}`, `dataset/processed/labels/{train,val,test}`
- `dataset/sample/images`, `dataset/sample/labels`
- `models/pretrained`, `models/trained`, `models/exported`
- `output/screenshots`, `output/videos`, `output/logs`
- `runs/train`, `runs/detect`, `runs/val`

## 9. Cài 5 model YOLO11

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

## 10. Kiểm tra model đã cài đủ chưa

```powershell
Get-ChildItem .\models\pretrained\yolo11*.pt
```

Bạn nên thấy đủ 5 file:

- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

## 11. Kiểm tra sức khỏe hệ thống

```powershell
.\.venv\Scripts\python run_doctor.py
```

Lệnh này sẽ kiểm tra:

- CPU, RAM, GPU, VRAM thực tế
- PyTorch, CUDA
- có đủ model chưa
- webcam có hoạt động không
- dữ liệu raw đã có chưa
- dữ liệu split đã có chưa

## 12. Chạy test toàn hệ thống

```powershell
.\.venv\Scripts\python run_tests.py
```

## 13. Cách chạy

### 13.1. Mở menu tổng

```powershell
.\.venv\Scripts\python run_menu.py
```

Menu có 6 tùy chọn:

| Phím | Lệnh | Mô tả |
|------|------|-------|
| `1` | `run_app.py` | Camera app - chế độ desktop |
| `2` | `run_detect.py` | Detect camera - chế độ CLI |
| `3` | `run_tools.py` | Xem cấu hình máy và 3 mức tối ưu |
| `4` | `run_train.py` | Huấn luyện model |
| `5` | `run_tests.py` | Chạy toàn bộ test |
| `6` | `run_doctor.py` | Kiểm tra toàn hệ thống |
| `0` | | Thoát |

### 13.2. Chạy camera app chính

```powershell
.\.venv\Scripts\python run_app.py
```

App sẽ:

- đọc cấu hình máy thực tế (CPU, RAM, GPU, VRAM, CUDA, PyTorch)
- tự đề xuất mức chạy phù hợp
- cho bạn chọn `Cao nhất`, `Trung bình` hoặc `Yếu`
- tự nạp model YOLO11 phù hợp, thử fallback nếu cần

```powershell
# chạy trực tiếp không cần chọn mode
.\.venv\Scripts\python run_app.py --mode auto
.\.venv\Scripts\python run_app.py --mode high
.\.venv\Scripts\python run_app.py --mode medium
.\.venv\Scripts\python run_app.py --mode low
```

### 13.3. Chạy detect camera

```powershell
.\.venv\Scripts\python run_detect.py
```

Giống `run_app.py` nhưng chạy ở chế độ CLI detect trực tiếp.

### 13.4. Xem cấu hình máy và 3 mức tối ưu

```powershell
.\.venv\Scripts\python run_tools.py
```

Hiển thị chi tiết:

- CPU, RAM, GPU, VRAM, CUDA, PyTorch
- 5 phiên bản YOLO11 đang có
- 3 mức chạy: `mạnh nhất`, `trung bình`, `yếu nhất`
- model, device, imgsz, max_det cho từng mức
- phân hạng GPU và tải hệ thống hiện tại

### 13.5. Chạy train

```powershell
.\.venv\Scripts\python run_train.py
```

### 13.6. Kiểm tra sức khỏe nhanh

```powershell
.\.venv\Scripts\python run_doctor.py
```

Nên chạy lệnh này khi:

- mới cài xong
- nghi ngờ thiếu model
- nghi ngờ CUDA không hoạt động
- muốn biết máy nên chạy mức nào

### 13.7. Chạy test hệ thống

```powershell
.\.venv\Scripts\python run_tests.py
```

## 14. Kiểm tra nhanh toàn bộ hệ thống

```powershell
.\.venv\Scripts\python run_doctor.py
.\.venv\Scripts\python run_tests.py
```

## 15. Khi chạy camera có thể làm gì

- quan sát kết quả detect theo thời gian thực
- bấm `T` để chụp mẫu train
- chụp mẫu có 2 chế độ:
  - `auto`: tự động đặt tên theo timestamp
  - `name`: nhập tên class thủ công
- mẫu chụp sẽ được lưu vào `dataset/sample/` với ảnh và label YOLO format
- cửa sổ luôn cố định kích thước 800×600
