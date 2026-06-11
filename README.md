# YOLO Realtime Camera Project

Bo project nay tap trung vao YOLO realtime tren webcam, menu terminal de mo cac cong cu chinh, va pipeline training tach rieng cho dataset cua ban.

## He thong hien tai

- Runtime camera chi con luong detect realtime, khong con luong chup sample bang phim `T` hoac `C`.
- Thu muc `dataset/sample/` va script `training/promote_samples.py` da duoc bo.
- Ket qua detect van hien thi bang box tren frame camera.
- Khi chay camera, he thong uu tien model pretrained YOLO COCO de nhan dien nhieu vat pho thong.
- Pipeline training van giu rieng model custom trong `models/trained/best.pt`.

## Cac script chinh

- `run_menu.py`: menu tong de mo nhanh cac script chinh.
- `run_app.py`: camera runtime chinh, co dashboard va luong chon mode.
- `run_detect.py`: camera detect toi gian theo CLI.
- `run_chat.py`: giao dien desktop/chat.
- `run_doctor.py`: kiem tra camera, model, CUDA, PyTorch va cac thanh phan runtime.
- `run_tests.py`: chay test toan repo.
- `run_train.py`: chay huan luyen.
- `run_tools.py`: bo tu van runtime, chi de xem de xuat theo may.

## Menu dang co

Khi mo `run_menu.py`, menu hien tai gom:

- `1`: `run_app.py`
- `2`: `run_detect.py`
- `3`: `run_chat.py`
- `4`: `run_tests.py`
- `5`: `run_doctor.py`
- `6`: `run_train.py`
- `0`: thoat

## Runtime camera

Khi chay `run_app.py` hoac `run_detect.py`:

- he thong doc CPU, RAM, GPU, VRAM va kha nang CUDA
- neu ban chua truyen `--mode`, chuong trinh se cho chon `high`, `medium`, `low`
- moi mode la muc tai toi uu theo phan cung, khong phai 3 model co dinh cho moi may
- model local duoc load theo thu tu uu tien trong `config/model_config.yaml`
- runtime camera hien uu tien `models/pretrained/yolo11s.pt`, `yolo11s.pt`, `models/pretrained/yolo11n.pt`, `yolo11n.pt`, sau do moi fallback sang `models/trained/best.pt`
- muc dich cua thu tu nay la de webcam nhan dien duoc nhieu doi tuong pho thong nhu `person`, `car`, `bottle`, `chair`, `cell phone`
- nhan `Esc` de thoat camera

Dieu quan trong:

- `run_app.py` va `run_detect.py` deu co co che tu chon mode runtime
- chung khong tu dong bat camera theo mot mode bi an nua neu ban chua chon `--mode`
- model custom train rieng van duoc su dung cho cac luong train, validate, export

## Dataset va training

Du lieu train duoc dat truc tiep vao:

- `dataset/raw/images`
- `dataset/raw/labels`

Cau truc train sau khi tach tap:

```text
dataset/
  raw/
    images/
    labels/
  processed/
    images/train
    images/val
    images/test
    labels/train
    labels/val
    labels/test
```

File class hien tai duoc khai bao trong `training/data.yaml`. O trang thai hien tai repo dang co:

```yaml
names:
  0: person
```

Neu ban muon train them `phone` hoac class khac, can cap nhat dataset va `training/data.yaml` dong bo.

## Cai dat nhanh

```powershell
cd D:\YOLO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python training\prepare_dataset.py
.\.venv\Scripts\python training\download_models.py
```

Neu dung NVIDIA GPU, hay cai dung ban PyTorch CUDA phu hop truoc.

## Cach chay

Menu tong:

```powershell
.\.venv\Scripts\python run_menu.py
```

Chay truc tiep:

```powershell
.\.venv\Scripts\python run_app.py
.\.venv\Scripts\python run_detect.py
.\.venv\Scripts\python run_chat.py
.\.venv\Scripts\python run_doctor.py
.\.venv\Scripts\python run_tests.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python run_tools.py
```

Chon mode bang CLI:

```powershell
.\.venv\Scripts\python run_app.py --mode high
.\.venv\Scripts\python run_detect.py --mode medium
```

## Quy trinh training de xuat

```powershell
.\.venv\Scripts\python training\validate_dataset.py
.\.venv\Scripts\python training\split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training\validate_model.py
.\.venv\Scripts\python training\export_model.py
```

## Kiem tra he thong

```powershell
.\.venv\Scripts\python run_doctor.py
.\.venv\Scripts\python run_tests.py
```

## Khong con trong repo

- chup sample train truc tiep tu camera runtime
- doi ten sample trong luc dang mo webcam
- dem nguoc giu yen de chup sample
- flow `camera -> dataset/sample -> promote_samples -> raw`
