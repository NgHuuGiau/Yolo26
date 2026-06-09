# Hướng Dẫn Training

## 1. Hiểu đúng 3 thư mục dữ liệu

**`dataset/sample/`**

- nơi lưu ảnh và label khi bạn bấm `T` trong app
- đây là mẫu chụp từ camera
- có 2 chế độ chụp:
  - `auto`: tự động đặt tên theo timestamp
  - `name`: nhập tên class thủ công

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

## 2. Trường hợp nào dùng thư mục nào

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

## 3. Quy tắc tên file ảnh và label

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

Đuôi ảnh hỗ trợ: jpg, jpeg, png, bmp, webp, tif, tiff

## 4. Format file label YOLO

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

- `class_id` là số class (số nguyên)
- các giá trị còn lại là tọa độ chuẩn hóa từ `0` đến `1`
- mỗi dòng phải có đúng 5 cột

## 5. Pipeline huấn luyện đầy đủ

Khi đã có dữ liệu trong `dataset/raw/`, hãy chạy đúng thứ tự này:

```powershell
.\.venv\Scripts\python training\validate_dataset.py
.\.venv\Scripts\python training\split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training\validate_model.py
.\.venv\Scripts\python training\export_model.py
```

Hoặc chạy trực tiếp bằng `run_train.py` (sẽ tự động kiểm tra và phân chia dữ liệu nếu cần):

```powershell
.\.venv\Scripts\python run_train.py
```

## 6. Giải thích từng lệnh huấn luyện

**Bước 0: Chuẩn bị**

```powershell
.\.venv\Scripts\python training\prepare_dataset.py
```

- tạo toàn bộ cấu trúc thư mục dự án

**1. `training\validate_dataset.py`**

- kiểm tra dữ liệu trong `raw`
- phát hiện:
  - ảnh thiếu label
  - label rỗng
  - label lỗi (sai format)
  - label mồ côi (không có ảnh tương ứng)

**2. `training\split_dataset.py`**

- lấy dữ liệu hợp lệ từ `raw`
- chia ngẫu nhiên theo seed 42:
  - `train` ~70%
  - `val` ~15%
  - `test` ~15%
- copy sang:
  - `processed/images/train`
  - `processed/images/val`
  - `processed/images/test`
  - `processed/labels/train`
  - `processed/labels/val`
  - `processed/labels/test`

**3. `run_train.py`**

- train model từ dữ liệu đã chia trong `processed`
- trước khi train, tự động:
  - kiểm tra ảnh thiếu label
  - nếu có ảnh thiếu label, dùng model hiện có để tự sinh label (auto-label)
  - phân chia lại dữ liệu
- config train: `training/train_config.yaml`
  - model: `yolo11s.pt`
  - fallback_model: `yolo11n.pt` (nếu train lỗi)
  - imgsz: 512
  - batch: 4
  - epochs: 80
  - patience: 20
  - device: GPU nếu có
- nếu train fail với `yolo11s.pt`:
  - tự động fallback sang `yolo11n.pt`
  - giới hạn imgsz ≤ 416, batch ≤ 4
- model tốt nhất được copy về:
  - `models/trained/best.pt`

**4. `training\validate_model.py`**

- dùng tập `val` để kiểm tra chất lượng model sau train
- ưu tiên `models/trained/best.pt`, nếu không có thì dùng `models/pretrained/yolo11n.pt`
- output: `runs/val/validation`

**5. `training\export_model.py`**

- export model sang định dạng ONNX để deploy
- yêu cầu `models/trained/best.pt` phải tồn tại
- nếu thiếu sẽ báo lỗi và hướng dẫn cần train trước

## 7. Sau khi train xong

- mở app để dùng model mới train:

```powershell
.\.venv\Scripts\python run_app.py
```

- hoặc kiểm tra model:

```powershell
.\.venv\Scripts\python training\validate_model.py
```

Hệ thống ưu tiên:

- `models/trained/best.pt` nếu tồn tại

## 8. Auto-label dữ liệu

Nếu có ảnh trong `raw` nhưng thiếu label, bạn có thể chạy:

```powershell
.\.venv\Scripts\python training\auto_label_raw.py
```

Lệnh này sẽ:

- dùng model hiện có để tự sinh label cho ảnh chưa có
- thứ tự model: `models/trained/best.pt` → `models/pretrained/yolo11s.pt` → `models/pretrained/yolo11n.pt`
- bỏ qua ảnh đã có label trừ khi dùng `--overwrite`
- confidence mặc định: 0.25

## 9. Lỗi thường gặp khi train

**Không có dữ liệu trong `raw`**

- nghĩa là bạn chưa bỏ ảnh và label vào đúng thư mục

**Không có dữ liệu trong `processed`**

- nghĩa là bạn chưa chạy `split_dataset.py`
- hoặc `run_train.py` chưa tự động phân chia

**Không có `models/trained/best.pt`**

- nghĩa là train chưa xong nên chưa thể export
- hoặc train lỗi và fallback chưa tạo được model

**Thiếu model YOLO11**

- hãy chạy:

```powershell
.\.venv\Scripts\python training\download_models.py
```

**Label sai format**

- kiểm tra lại: mỗi dòng phải có 5 cột
- class_id là số nguyên, các giá trị khác từ 0 đến 1

## 10. Chat AI

Sau khi cài đặt các phụ thuộc Chat AI, bạn có thể sử dụng giao diện chat:

Tính năng chat AI bao gồm:

- Giao diện chat đa ngôn ngữ (Tiếng Anh/Tiếng Việt)
- Gửi ảnh, text file, hoặc chụp từ camera làm attachment
- Nhận dạng giọng nói (Whisper)
- Lưu trữ cuộc trò chuyện bằng SQLite
