# Hướng Dẫn Training

## Chuẩn bị dữ liệu

1. Đưa ảnh vào `dataset/raw/images/`
2. Đưa label YOLO vào `dataset/raw/labels/`
3. Chạy:

```bash
python training/split_dataset.py
```

## Cập nhật file dữ liệu

Sửa `training/data.yaml` cho đúng class và đường dẫn.

## Chạy train

```bash
python run_train.py
```

## Validate

```bash
python training/validate_model.py
```

## Export

```bash
python training/export_model.py
```
