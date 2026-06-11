# Project Overview

## Muc tieu

Repo nay tap trung vao 3 nhom chuc nang:

- chay YOLO realtime tu webcam
- tu van va chon runtime theo phan cung
- train, validate, export model tu dataset co san

## Tong quan kien truc

```text
run_menu.py
  -> run_app.py / run_detect.py / run_chat.py / run_doctor.py / run_tests.py / run_train.py

run_app.py, run_detect.py
  -> app.runtime_entry
  -> core.camera_runner
  -> core.model_selector
  -> core.model_loader
  -> core.runtime_advisor

run_tools.py
  -> tools.runtime_tool

run_train.py
  -> training.train_model
```

## Sau cleanup

He thong da duoc rut gon de de bao tri hon:

- bo luong chup sample train trong runtime camera
- bo `dataset/sample/`
- bo script `training/promote_samples.py`
- bo helper va UI phu thuoc vao phim `T/C`
- giu lai phan detect realtime bang box tren frame

Dieu nay co nghia runtime camera hien chi con:

- mo camera
- chay detect
- ve box
- hien thong tin runtime/FPS
- thoat bang `Esc`

## Cach he thong chon runtime

Khi vao `run_app.py` hoac `run_detect.py`:

- he thong doc phan cung hien tai
- tinh de xuat cho `high`, `medium`, `low`
- neu chua truyen `--mode`, nguoi dung se duoc chon 1 trong 3 muc
- sau do runtime duoc resolve ra `model`, `device`, `imgsz`, `max_det`

`high`, `medium`, `low` la muc tai theo may, khong phai 3 script khac nhau.

## Chien luoc model hien tai

Runtime camera dang uu tien model pretrained de detect tong quat:

- `models/pretrained/yolo11s.pt`
- `yolo11s.pt`
- `models/pretrained/yolo11n.pt`
- `yolo11n.pt`
- `models/trained/best.pt`

Ly do:

- camera can detect nhieu doi tuong pho thong
- pretrained COCO hop hon cho nhu cau su dung ngay
- model custom train rieng van duoc bao luu cho luong training

## Thu muc chinh

```text
app/
core/
docs/
models/
tests/
tools/
training/
utils/
run_app.py
run_chat.py
run_detect.py
run_doctor.py
run_menu.py
run_tests.py
run_tools.py
run_train.py
```

## Core modules quan trong

- `core/camera_runner.py`: dieu phoi camera, inference, render frame, filter detection
- `core/model_loader.py`: load model local theo thu tu uu tien va fallback
- `core/model_selector.py`: resolve runtime config co ban theo mode va phan cung
- `core/runtime_advisor.py`: xay de xuat toi uu cho `high`, `medium`, `low`
- `core/hardware_info.py`: doc CPU, RAM, GPU, VRAM, CUDA, PyTorch
- `tools/runtime_tool.py`: hien thi bo tu van runtime de tham khao

## Training flow

```text
dataset/raw -> validate -> split -> train -> validate_model -> export
```

Khong con flow cu:

```text
camera -> dataset/sample -> promote_samples -> raw
```

## Trang thai dataset hien tai

`training/data.yaml` dang kha bao toi thieu:

```yaml
names:
  0: person
```

Neu ban muon repo train them nhieu class hon, can cap nhat dataset va file cau hinh nay dong bo.
