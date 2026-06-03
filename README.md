# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

## 1. Giới thiệu dự án

Đây là dự án nhận diện vật thể realtime bằng YOLO chạy trên desktop Python + OpenCV.

Dự án hiện có:

- giao diện terminal tiếng Việt
- tự dò cấu hình máy trước khi chạy
- tự gợi ý mức phù hợp theo `CPU`, `RAM`, `GPU`, `VRAM`, `PyTorch`, `CUDA`
- hỗ trợ 5 model `YOLO26`
- hỗ trợ chụp mẫu train trực tiếp từ webcam
- pipeline train, validate, export riêng
- test dashboard để kiểm tra toàn hệ thống

### 5 model YOLO26 đang dùng

- `yolo26n.pt`
- `yolo26s.pt`
- `yolo26m.pt`
- `yolo26l.pt`
- `yolo26x.pt`

### 3 mức người dùng sẽ thấy

- `Cao nhất`
- `Trung bình`
- `Yếu`

Hệ thống sẽ không ép model lớn nhất bằng mọi giá. Nó sẽ chọn mức cao nhất mà máy còn chạy ổn định.

Ví dụ:

- máy rất mạnh -> có thể dùng `yolo26x.pt`
- máy mạnh -> có thể dùng `yolo26l.pt`
- máy tầm trung như RTX 3050 Ti 4GB -> thường hợp `yolo26s.pt`
- máy yếu hoặc CPU-only -> thường về `yolo26n.pt`

### Lưu ý quan trọng

- repo này chỉ chứa code
- repo này không kèm model `.pt`
- repo này không kèm dataset
- bạn phải tự đặt model vào `models/pretrained/`

### Chạy nhanh nếu muốn vào việc ngay

Nếu bạn muốn đi theo đường ngắn nhất, làm đúng thứ tự này:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python training\prepare_dataset.py
.\.venv\Scripts\python training\download_models.py
.\.venv\Scripts\python run_doctor.py
.\.venv\Scripts\python run_app.py
```

Ý nghĩa:

1. tạo môi trường riêng cho dự án
2. cài thư viện
3. tạo thư mục hệ thống
4. tải đủ model YOLO26
5. kiểm tra toàn hệ thống
6. mở app camera

## 2. Cách cài

### 2.1. Yêu cầu

- Windows + PowerShell
- Python `3.10+`
- webcam hoạt động bình thường
- nếu muốn chạy GPU NVIDIA thì cần PyTorch CUDA phù hợp

### 2.2. Clone dự án

```powershell
git clone <repo-url>
cd D:\YOLO
```

Nếu bạn đã có sẵn thư mục dự án rồi thì chỉ cần:

```powershell
cd D:\YOLO
```

### 2.3. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 2.4. Kích hoạt môi trường

```powershell
.\\.venv\\Scripts\\Activate.ps1
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

#### Nếu chỉ chạy CPU

```powershell
.\\.venv\\Scripts\\python -m pip uninstall -y torch torchvision torchaudio
.\\.venv\\Scripts\\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Nếu chạy GPU NVIDIA CUDA

```powershell
.\\.venv\\Scripts\\python -m pip uninstall -y torch torchvision torchaudio
.\\.venv\\Scripts\\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 2.7. Kiểm tra PyTorch và CUDA

```powershell
.\\.venv\\Scripts\\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Nếu `torch.cuda.is_available()` là `False` thì máy đang chạy CPU hoặc môi trường CUDA chưa đúng.

### 2.8. Tạo thư mục hệ thống

```powershell
.\\.venv\\Scripts\\python training\prepare_dataset.py
```

Lệnh này tạo sẵn:

- `dataset/raw/images`
- `dataset/raw/labels`
- `dataset/processed/...`
- `dataset/sample/...`
- `models/pretrained`
- `models/trained`
- `models/exported`

### 2.9. Cài model YOLO26

Repo không kèm model. Bạn phải tự đặt model vào:

```text
models/pretrained/
```

Nên có đủ:

- `models/pretrained/yolo26n.pt`
- `models/pretrained/yolo26s.pt`
- `models/pretrained/yolo26m.pt`
- `models/pretrained/yolo26l.pt`
- `models/pretrained/yolo26x.pt`

#### Tải đủ 5 model tự động

```powershell
.\\.venv\\Scripts\\python training\download_models.py
```

#### Tải lại dù file đã có

```powershell
.\\.venv\\Scripts\\python training\download_models.py --force
```

#### Tải riêng một vài model

```powershell
.\\.venv\\Scripts\\python training\download_models.py --models yolo26n.pt yolo26s.pt
```

### 2.10. Kiểm tra toàn hệ thống trước khi chạy

```powershell
.\\.venv\\Scripts\\python run_doctor.py
```

Lệnh này sẽ kiểm tra:

- CPU, RAM, GPU, VRAM
- PyTorch, CUDA
- model YOLO26 đang có hay thiếu
- dữ liệu raw đã có chưa
- dữ liệu train/val đã split chưa
- máy nên chạy mức nào

### 2.11. Chạy test

```powershell
.\\.venv\\Scripts\\python run_tests.py
```

Trạng thái hiện tại:

- `56/56 PASS`

### 2.12. Sau khi cài xong nên kiểm tra theo thứ tự nào

Tôi khuyên chạy theo đúng thứ tự này:

```powershell
.\.venv\Scripts\python run_doctor.py
.\.venv\Scripts\python run_tests.py
.\.venv\Scripts\python run_app.py
```

Giải thích:

- `run_doctor.py`: kiểm tra máy, model, dataset, CUDA
- `run_tests.py`: kiểm tra code và logic hệ thống
- `run_app.py`: chạy webcam thật

## 3. Cách chạy

### 3.1. Chạy app camera chính

```powershell
.\\.venv\\Scripts\\python run_app.py
```

File này dùng khi:

- bạn muốn mở camera theo luồng desktop chính
- muốn chọn mode trực tiếp trước khi chạy
- muốn dùng tính năng chụp mẫu bằng phím `T`

### 3.2. Chạy detect camera

```powershell
.\\.venv\\Scripts\\python run_detect.py
```

File này dùng khi:

- bạn muốn chạy detect camera theo entrypoint detect riêng
- muốn test riêng luồng detect mà không đi qua app chính

Thực tế 2 file này đều chạy webcam và detect. Khác nhau chủ yếu ở entrypoint và cách tổ chức luồng gọi.

### 3.2A. Khi nào nên chạy file nào

| File | Khi nào dùng |
|---|---|
| `run_app.py` | dùng hằng ngày để mở camera chính |
| `run_detect.py` | dùng khi muốn test riêng luồng detect |
| `run_train.py` | dùng khi bắt đầu huấn luyện |
| `run_tests.py` | dùng khi muốn kiểm tra toàn bộ code |
| `run_doctor.py` | dùng khi muốn kiểm tra máy đang thiếu gì |

### 3.3. Chạy với mode cố định

```powershell
.\\.venv\\Scripts\\python run_app.py --mode high
.\\.venv\\Scripts\\python run_app.py --mode medium
.\\.venv\\Scripts\\python run_app.py --mode low
```

Ý nghĩa:

- `high`: yêu cầu mức cao
- `medium`: yêu cầu mức cân bằng
- `low`: yêu cầu mức nhẹ

Hệ thống vẫn có thể tự hạ nếu phần cứng không chịu nổi.

### 3.4. Đổi camera index

```powershell
.\\.venv\\Scripts\\python run_app.py --camera-index 1
.\\.venv\\Scripts\\python run_detect.py --camera-index 1
```

### 3.5. Màu terminal nghĩa là gì

- xanh lá: chạy được, trạng thái tốt
- vàng: cảnh báo hoặc trung gian
- đỏ: lỗi hoặc không đủ điều kiện chạy

Nếu không chạy được, terminal sẽ hiện:

- `LÝ DO`
- `Lý do không chạy`
- `GỢI Ý`
- `LỆNH THỬ` hoặc `LỆNH NHANH`

### 3.6. Chụp mẫu train từ camera

Khi camera đang chạy:

- bấm `T` để vào chế độ chụp mẫu
- hệ thống kiểm tra độ ổn định trong `5` giây
- nếu rung/lắc thì đếm lại
- nếu đủ ổn định thì hiện bảng nhập tên mẫu

Phím dùng:

- `Enter`: lưu
- `Backspace`: xóa
- `Esc`: hủy

Mẫu sẽ lưu vào:

- `dataset/sample/images/`
- `dataset/sample/labels/`

## 4. Huấn luyện

Toàn bộ lệnh huấn luyện chạy từ thư mục gốc dự án:

```powershell
PS D:\YOLO>
```

### 4.1. Hiểu đúng 3 thư mục dữ liệu

`dataset/sample/`

- nơi chứa mẫu vừa chụp bằng phím `T`
- không train trực tiếp từ đây

`dataset/raw/`

- nơi chứa dữ liệu gốc chính thức để train
- nếu bạn đã có sẵn ảnh và `.txt`, hãy bỏ thẳng vào đây

`dataset/processed/`

- là dữ liệu đã được hệ thống chia sẵn thành:
  - `train`
  - `val`
  - `test`
- `run_train.py` train từ đây, không train trực tiếp từ `raw`

Luồng đúng:

```text
sample -> raw -> processed -> train
```

### 4.2. Nếu bạn có sẵn ảnh và txt

Bạn chỉ cần tự bỏ vào:

- ảnh -> `dataset/raw/images/`
- label -> `dataset/raw/labels/`

Ví dụ:

```text
dataset/raw/images/frame_001.jpg
dataset/raw/labels/frame_001.txt
```

Mỗi file label YOLO có dạng:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ:

```text
0 0.512 0.438 0.220 0.310
```

### 4.3. Nếu bạn chụp bằng phím `T`

Chụp xong dữ liệu sẽ nằm ở:

- `dataset/sample/images/`
- `dataset/sample/labels/`

Sau đó chuyển sang `raw` bằng:

```powershell
.\\.venv\\Scripts\\python training\promote_samples.py
```

Lệnh này sẽ copy:

- `dataset/sample/images/*` -> `dataset/raw/images/`
- `dataset/sample/labels/*` -> `dataset/raw/labels/`

### 4.4. Kiểm tra dataset raw

```powershell
.\\.venv\\Scripts\\python training\validate_dataset.py
```

Lệnh này báo:

- tổng số ảnh raw
- ảnh hợp lệ
- ảnh thiếu label
- label rỗng
- label lỗi
- label mồ côi

### 4.5. Chia train / val / test

```powershell
.\\.venv\\Scripts\\python training\split_dataset.py
```

Sau bước này dữ liệu sẽ nằm ở:

- `dataset/processed/images/train`
- `dataset/processed/images/val`
- `dataset/processed/images/test`
- `dataset/processed/labels/train`
- `dataset/processed/labels/val`
- `dataset/processed/labels/test`

### 4.6. Kiểm tra `training/data.yaml`

File này map `class_id` sang tên class thật.

Ví dụ:

```yaml
path: ../dataset/processed
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: helmet
```

### 4.7. Kiểm tra `training/train_config.yaml`

Các mục quan trọng:

- `model`
- `fallback_model`
- `epochs`
- `imgsz`
- `batch`
- `device`
- `project`
- `name`

Nếu máy yếu hoặc thiếu VRAM, nên giảm:

- `imgsz`
- `batch`

### 4.8. Chạy train

```powershell
.\\.venv\\Scripts\\python run_train.py
```

Lệnh này sẽ:

1. đọc `training/train_config.yaml`
2. kiểm tra dữ liệu đã split
3. load model train chính
4. fallback sang model nhẹ hơn nếu cần
5. copy `best.pt` về `models/trained/best.pt`

### 4.9. Validate model

```powershell
.\\.venv\\Scripts\\python training\validate_model.py
```

### 4.10. Export model

```powershell
.\\.venv\\Scripts\\python training\export_model.py
```

### 4.11. Chuỗi lệnh huấn luyện đầy đủ

Nếu bạn đã có sẵn ảnh và txt:

```powershell
.\\.venv\\Scripts\\python training\validate_dataset.py
.\\.venv\\Scripts\\python training\split_dataset.py
.\\.venv\\Scripts\\python run_train.py
.\\.venv\\Scripts\\python training\validate_model.py
.\\.venv\\Scripts\\python training\export_model.py
```

Nếu bạn chụp từ camera bằng `T`:

```powershell
.\\.venv\\Scripts\\python training\promote_samples.py
.\\.venv\\Scripts\\python training\validate_dataset.py
.\\.venv\\Scripts\\python training\split_dataset.py
.\\.venv\\Scripts\\python run_train.py
```

### 4.12. Hiểu đúng ai tự làm gì, ai phải làm tay

Bạn phải tự làm:

- tự bỏ ảnh + txt vào `dataset/raw/` nếu đã có sẵn dữ liệu
- hoặc tự chụp mẫu bằng `T`
- hoặc tự chạy `promote_samples.py` để đưa mẫu từ `sample` sang `raw`

Hệ thống tự làm:

- kiểm tra dataset raw
- chia raw sang processed
- train từ processed
- validate model
- export model

Nói ngắn:

- có sẵn ảnh + txt -> bạn tự bỏ vào `raw`
- chụp trong app -> hệ thống tự lưu vào `sample`
- muốn train -> phải đi qua `raw` rồi `processed`

## 5. Giải thích cấu trúc thư mục

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
|-- run_tests.py
`-- run_doctor.py
```

### `app/`

- helper cho entrypoint camera

### `config/`

- YAML cấu hình hệ thống

### `core/`

- lõi detect
- dò phần cứng
- chọn runtime
- load model
- chạy camera realtime

### `dataset/`

- `sample/`: mẫu chụp từ camera
- `raw/`: dữ liệu gốc để train
- `processed/`: dữ liệu đã chia train/val/test

### `docs/`

- tài liệu phụ

### `models/`

- `pretrained/`: model local bạn tự đặt vào
- `trained/`: model train xong, quan trọng nhất là `best.pt`
- `exported/`: model export

### `output/`

- ảnh, log, video sinh ra khi chạy

### `runs/`

- artifact do Ultralytics sinh ra

### `tests/`

- test tự động

### `training/`

- `prepare_dataset.py`: tạo thư mục dataset
- `validate_dataset.py`: kiểm tra raw
- `split_dataset.py`: chia train/val/test
- `train_model.py`: logic train chính
- `validate_model.py`: validate model
- `export_model.py`: export model
- `download_models.py`: tải model YOLO26
- `promote_samples.py`: chuyển mẫu từ sample sang raw

### `utils/`

- helper dùng chung như terminal UI, file utils, draw utils

### Các file chạy chính

- `run_app.py`: chạy app camera chính
- `run_detect.py`: chạy detect camera
- `run_train.py`: chạy train
- `run_tests.py`: chạy test dashboard
- `run_doctor.py`: kiểm tra sức khỏe toàn hệ thống

## 6. Trạng thái hiện tại

- terminal tiếng Việt có dấu
- menu chọn mode đã gợi ý theo cấu hình máy
- hỗ trợ đủ logic cho `YOLO26 n/s/m/l/x`
- `run_tests.py` hiện pass `56/56`
