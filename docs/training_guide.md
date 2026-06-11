# Training Guide

## Dau vao cua pipeline

Pipeline train hien tai chi dung du lieu dat truc tiep vao:

- `dataset/raw/images`
- `dataset/raw/labels`

Khong con luong trung gian `dataset/sample`.

## Cau truc label

Moi file label YOLO co dang:

```text
class_id x_center y_center width height
```

Tat ca gia tri toa do phai duoc normalize trong khoang `0..1`.

## Class hien tai

File `training/data.yaml` dang quy dinh class train cua repo. O hien tai:

```yaml
names:
  0: person
```

Neu muon train them `phone` hoac class khac:

- them anh va label dung class id
- cap nhat `training/data.yaml`
- dam bao tat ca label trong dataset dung cung mapping class

## Thu tu chay de xuat

```powershell
.\.venv\Scripts\python training\prepare_dataset.py
.\.venv\Scripts\python training\validate_dataset.py
.\.venv\Scripts\python training\split_dataset.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python training\validate_model.py
.\.venv\Scripts\python training\export_model.py
```

Y nghia tung buoc:

- `prepare_dataset.py`: tao thu muc can thiet va chuan bi khung du an
- `validate_dataset.py`: soat loi label, file thieu, class sai, format sai
- `split_dataset.py`: chia train/val/test
- `run_train.py`: huan luyen model
- `validate_model.py`: danh gia model sau train
- `export_model.py`: xuat model sang dinh dang phuc vu deploy

## Luu y ve model

- Model custom sau khi train nam o `models/trained/best.pt`
- Cac script train, validate, export duoc thiet ke xoay quanh model custom nay
- Runtime camera hien tai uu tien model pretrained COCO cho detect da vat
- Vi vay, can tach ro hai muc dich:

`models/pretrained/*.pt`: dung cho runtime webcam tong quat

`models/trained/best.pt`: dung cho bai toan train rieng cua dataset

## Khi nao nen dung model custom trong camera

Ban nen doi logic uu tien model neu:

- dataset cua ban rat chuyen biet
- ban muon webcam chi nhan dien class rieng
- do chinh xac tren class custom quan trong hon detect nhieu vat pho thong

Neu khong, cau hinh hien tai hop ly hon cho nhu cau detect tong quat.

## Khong con ho tro

- chup sample train truc tiep tu camera runtime
- phim `T/C` de tao sample
- script `training/promote_samples.py`
- flow `camera -> sample -> raw`
