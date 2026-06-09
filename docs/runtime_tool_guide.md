# Hướng Dẫn Công Cụ Runtime

Tool trong thư mục `tools/` dùng để xem cấu hình máy hiện tại và đề xuất 3 mức chạy tối ưu cho YOLO11.

## Công dụng

- đọc cấu hình máy: CPU, RAM, GPU, VRAM, CUDA, PyTorch
- phân định rõ 3 mức chạy:
  - `mạnh nhất` = mức cao nhất máy còn gánh được
  - `trung bình` = mức cân bằng để dùng thường xuyên
  - `yếu nhất` = mức thấp nhất / dễ chạy nhất, ưu tiên mượt
- đánh giá trên 5 phiên bản YOLO11: `yolo11x.pt`, `yolo11l.pt`, `yolo11m.pt`, `yolo11s.pt`, `yolo11n.pt`
- hiển thị `model`, `device`, `imgsz`, `max_det`, fallback chain, và mức hệ thống đề xuất

## Ví dụ RTX 3050 Ti 4GB

- `mạnh nhất`: `yolo11s.pt` / `cuda:0` / `imgsz 640` / `max_det 150`
- `trung bình`: `yolo11s.pt` / `cuda:0` / `imgsz 512` / `max_det 120`
- `yếu nhất`: `yolo11n.pt` / `cuda:0` / `imgsz 416` / `max_det 100`

## Cách dùng

```powershell
.\.venv\Scripts\python run_tools.py
```

## Ghi chú

- `run_tools.py` là lệnh chạy chính.
- Trong `tools/` chỉ có 1 script: `runtime_tool.py`.

## Chat AI

Sau khi cài đặt các phụ thuộc Chat AI, chạy chat AI:

```powershell
.\.venv\Scripts\python -c "from app.chat_ai_app import build_chat_arg_parser, launch_chat_ai_app; import sys; args = build_chat_arg_parser('Chat AI').parse_args([]); launch_chat_ai_app(window_title='YOLO Chat AI', camera_index=0)"
```

Hoặc tạo file `run_chat.py` đơn giản để chạy nhanh:

```python
from app.chat_ai_app import build_chat_arg_parser, launch_chat_ai_app

if __name__ == "__main__":
    args = build_chat_arg_parser("YOLO Chat AI").parse_args()
    launch_chat_ai_app(window_title="YOLO Chat AI", camera_index=args.camera_index)
```

Tính năng chat AI bao gồm:

- Giao diện chat đa ngôn ngữ (Tiếng Anh/Tiếng Việt)
- Gửi ảnh, text file, hoặc chụp từ camera làm attachment
- Nhận dạng giọng nói (Whisper)
- Lưu trữ cuộc trò chuyện bằng SQLite