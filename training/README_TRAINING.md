# Huong Dan Huấn Luyen

Thu muc `training/` chua toan bo pipeline train model YOLO rieng cho du an.

## Luong co ban

1. Dua du lieu vao `dataset/raw/images` va `dataset/raw/labels`.
2. Chay `python training/validate_dataset.py` de kiem tra dataset raw.
3. Chay `python training/split_dataset.py` de chia train/val/test.
4. Cap nhat class names trong `training/data.yaml`.
5. Chay `python run_train.py`.
6. Validate bang `python training/validate_model.py`.
7. Export bang `python training/export_model.py`.

## Dataset hop le can gi

- Moi anh phai co file label `.txt` cung stem.
- Moi dong label phai theo format YOLO:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id` phai la so nguyen >= 0.
- 4 gia tri toa do phai nam trong khoang `0..1`.
- Label rong van hop le neu anh do la negative sample.

## Toi uu dataset trong repo hien tai

- `split_dataset.py` se xoa sach `dataset/processed/` truoc khi chia lai.
- Anh thieu label se bi bo qua.
- Label loi se bi bo qua.
- Label mo coi se duoc bao ra de ban sua.

## Cau hinh khuyen nghi cho RTX 3050 Ti 4GB

- Mac dinh: `yolo11s.pt`, `imgsz=512`, `batch=4`
- Neu VRAM loi: ha xuong `yolo11n.pt`, `imgsz=416`
- Neu van loi: tiep tuc giam `imgsz=320`, `batch=2`

## Dau ra

- Ket qua train: `runs/train/`
- Model tot nhat: `models/trained/best.pt`
- Ket qua validate: `runs/val/`
- Model export: ONNX do script export sinh ra
