# Runtime Tool Guide

`run_tools.py` la cong cu giai thich runtime theo may hien tai. No khong mo camera, khong train, va khong chay detect.

## Lenh chay

```powershell
.\.venv\Scripts\python run_tools.py
```

## Cong cu nay dung de lam gi

No se tham do may va hien:

- CPU
- RAM
- GPU
- VRAM
- tinh trang CUDA/PyTorch
- model local dang co va model con thieu
- 3 muc runtime `high`, `medium`, `low`
- `model`, `device`, `imgsz`, `max_det` cua tung muc
- muc de xuat nen chay ngay luc hien tai

## Cach doc ket qua

- `high`: muc cao nhat may con ganh duoc
- `medium`: muc can bang de dung thuong xuyen
- `low`: muc nhe nhat de uu tien do muot va an toan

Moi muc la mot goi cau hinh runtime, khong chi don thuan la doi ten model.

## Phan biet voi runtime camera

`run_tools.py` chi giai thich.

`run_app.py` va `run_detect.py` moi la hai diem vao de mo webcam va detect realtime.

## Khi nao nen mo cong cu nay

Nen dung khi ban muon:

- biet may hien tai hop voi mode nao
- xem vi sao he thong de xuat `high`, `medium`, hay `low`
- kiem tra model nao da co san trong may
- doi chieu giua runtime theo du an va kha nang phan cung thuc te
