# YOLO Realtime Camera Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Windows](https://img.shields.io/badge/Windows-11%2B-0078D6?logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![PowerShell](https://img.shields.io/badge/PowerShell-7%2B-5391FE?logo=powershell&logoColor=white)](https://learn.microsoft.com/powershell/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111111)](https://www.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-8CAAE6)](https://python-pillow.org/)
[![YAML](https://img.shields.io/badge/YAML-Config-CB171E?logo=yaml&logoColor=white)](https://yaml.org/)

## Giới thiệu dự án

Đây là dự án nhận diện vật thể realtime bằng webcam, chạy trên desktop Windows với Python, OpenCV, Ultralytics YOLO11 và PyTorch.

Dự án hiện có:

- giao diện terminal tiếng Việt
- tự dò cấu hình máy trước khi chạy
- tự gợi ý mức phù hợp theo `CPU`, `RAM`, `GPU`, `VRAM`, `PyTorch`, `CUDA`
- hỗ trợ 5 model `YOLO11`
- hỗ trợ chụp mẫu train trực tiếp từ webcam bằng phím `T`
- pipeline huấn luyện, kiểm tra và export model riêng
- menu tổng để chạy nhanh các file chính
- bộ test để kiểm tra toàn hệ thống
- tự động fallback khi model hoặc cấu hình chính lỗi
- nhận dạng giọng nói (Whisper)ình chính lỗi
- chat AI với Gemini API (PySide6 GUI)
- nhận dạng giọng nói (Whisper)

### Ngôn ngữ và thư viện đang dùng

**Ngôn ngữ chính**

- `Python`

**Thư viện AI / Computer Vision**

- `ultralytics`
- `torch`
- `torchvision`
- `torchaudio`
- `opencv-python`
- `pillow`
- `numpy`

**Thư viện dữ liệu / hỗ trợ**

- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `PyYAML`
- `psutil`
- `GPUtil`

**Thư viện Chat AI**

- `PySide6`
- `faster-whisper`
- `pyaudio`

### 5 model YOLO11 đang dùng

- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

### 3 mức người dùng sẽ thấy khi chạy camera

- `Cao nhất`
- `Trung bình`
- `Yếu`

Hệ thống không ép model lớn nhất bằng mọi giá. Nó sẽ cố chọn mức cao nhất mà máy còn chạy ổn định.

Ví dụ:

- máy rất mạnh, VRAM lớn -> có thể lên `yolo11x.pt`
- máy mạnh -> có thể lên `yolo11l.pt`
- máy tầm trung như `RTX 3050 Ti 4GB` -> thường về `yolo11s.pt` hoặc `yolo11m.pt`
- máy yếu hoặc CPU-only -> thường về `yolo11n.pt`

### Bảng chọn model theo máy

| Điều kiện phần cứng | High | Medium | Low |
|---|---|---|---|
| GPU >= 12GB VRAM | `yolo11x.pt` / GPU / `imgsz 960` | `yolo11m.pt` / GPU / `imgsz 768` | `yolo11n.pt` / GPU / `imgsz 416` |
| GPU >= 8GB VRAM | `yolo11l.pt` / GPU / `imgsz 896` | `yolo11m.pt` / GPU / `imgsz 768` | `yolo11n.pt` / GPU / `imgsz 416` |
| GPU >= 4GB VRAM | `yolo11s.pt` / GPU / `imgsz 640` | `yolo11s.pt` / GPU / `imgsz 640` | `yolo11n.pt` / GPU / `imgsz 416` |
| GPU 3GB - dưới 4GB VRAM | `yolo11s.pt` / GPU / `imgsz 512` | `yolo11s.pt` / GPU / `imgsz 512` | `yolo11n.pt` / GPU / `imgsz 416` |
| GPU dưới 3GB VRAM | `yolo11n.pt` / GPU / `imgsz 512` | `yolo11n.pt` / GPU / `imgsz 512` | `yolo11n.pt` / CPU / `imgsz 320` |
| Không có CUDA / CPU-only | `yolo11n.pt` hoặc `yolo11s.pt` / CPU / `imgsz 416` | `yolo11n.pt` / CPU / `imgsz 416` | `yolo11n.pt` / CPU / `imgsz 320` |

### Bảng tối ưu FPS theo chế độ

| Chế độ | Nhịp detect | Cách hiển thị frame xen kẽ | Tối ưu nền đang bật |
|---|---|---|---|
| `High` | detect mỗi `1` frame | vẽ detection mới liên tục | worker thread infer, cache panel `BGR`, giảm convert thừa |
| `Medium` | detect mỗi `2` frame (tối đa 5) | giữ box gần nhất để nhìn vẫn mượt | worker thread infer, cache panel `BGR`, `sidebar/stat/chat` refresh thưa hơn |
| `Low` | detect mỗi `5` frame (tối đa 8) | giữ box gần nhất để ưu tiên FPS | worker thread infer, cache panel `BGR`, giảm redraw panel phụ mạnh nhất |

Ghi chú:

- khung camera cố định 800×600 theo `High / Medium / Low`
- `Medium` và `Low` ưu tiên FPS bằng cách giảm số lần gọi YOLO và giảm số lần redraw

### Lưu ý quan trọng

- repo này chỉ chứa code
- repo này không kèm model `.pt`
- repo này không kèm dataset
- bạn phải tự đặt model vào `models/pretrained/`
