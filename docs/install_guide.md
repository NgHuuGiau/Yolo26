# Install Guide

## Yeu cau

- Windows
- Python 3.10 tro len
- webcam neu muon chay camera realtime
- GPU NVIDIA la tuy chon, nhung neu co thi nen cai PyTorch CUDA dung ban

## Buoc 1: tao moi truong

```powershell
cd D:\YOLO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu PowerShell chan script:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Buoc 2: chuan bi thu muc du an

```powershell
.\.venv\Scripts\python training\prepare_dataset.py
.\.venv\Scripts\python training\download_models.py
```

Hai lenh nay se:

- tao cac thu muc dataset/models neu chua co
- tai model pretrained co ban de runtime camera co the chay ngay

## Buoc 3: kiem tra cai dat

```powershell
.\.venv\Scripts\python run_doctor.py
.\.venv\Scripts\python run_tests.py
```

Ban nen uu tien `run_doctor.py` truoc de xem:

- camera co mo duoc khong
- PyTorch co nhan CUDA khong
- model local co san khong
- cac file cau hinh co day du khong

## Buoc 4: chay thu

Menu tong:

```powershell
.\.venv\Scripts\python run_menu.py
```

Hoac chay tung script:

```powershell
.\.venv\Scripts\python run_app.py
.\.venv\Scripts\python run_detect.py
.\.venv\Scripts\python run_train.py
.\.venv\Scripts\python run_tools.py
```

## Ghi chu ve PyTorch CUDA

Neu may co GPU NVIDIA nhung `run_doctor.py` bao dang chay CPU:

- kiem tra ban PyTorch hien tai co ho tro CUDA khong
- kiem tra driver NVIDIA
- cai lai PyTorch theo dung CUDA version cua may

Neu chua dung duoc CUDA, he thong van co the chay CPU, nhung se cham hon.
