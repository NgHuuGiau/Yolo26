# Project Spec

Tai lieu nay tom tat lai spec trong `YEU_CAU_DU_AN_YOLO_CAMERA_GOM_LAI.md`.

- Uu tien GPU CUDA neu co.
- RTX 3050 Ti 4GB:
  - High: `yolo26s.pt`, `imgsz=640`
  - Medium: `yolo26s.pt`, `imgsz=512`
  - Low: `yolo26n.pt`, `imgsz=416`
- Fallback bat buoc:
  - GPU + `yolo26s.pt` 640
  - GPU + `yolo26s.pt` 512
  - GPU + `yolo26s.pt` 416
  - GPU + `yolo26n.pt` 416
  - CPU + `yolo26n.pt` 320
