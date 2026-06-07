## Tools folder

Thu muc `tools/` chi giu 1 tool duy nhat.
Tool nay dung de:
- xem cau hinh may hien tai
- phan dinh ro 3 muc chay
- `manh nhat` = cao nhat
- `trung binh` = can bang
- `yeu nhat` = thap nhat / de chay nhat
- xet tren 5 phien ban YOLO11: `yolo11x.pt`, `yolo11l.pt`, `yolo11m.pt`, `yolo11s.pt`, `yolo11n.pt`

### Tool duy nhat

- `runtime_tool.py`: Doc cau hinh may, tinh 3 muc `manh nhat / trung binh / yeu nhat`, in `model`, `device`, `imgsz`, fallback, va muc he thong de xuat nen chay.

### Vi du RTX 3050 Ti 4GB

- `manh nhat`: `yolo11s.pt` / `cuda:0` / `imgsz 640` / `max_det 150`
- `trung binh`: `yolo11s.pt` / `cuda:0` / `imgsz 512` / `max_det 120`
- `yeu nhat`: `yolo11n.pt` / `cuda:0` / `imgsz 416` / `max_det 100`

### Cach dung nhanh

```powershell
.\.venv\Scripts\python run_tools.py
```

### Ghi chu

- `run_tools.py` la lenh chay chinh.
- Trong `tools/` chi con 1 script chinh la `runtime_tool.py`.
