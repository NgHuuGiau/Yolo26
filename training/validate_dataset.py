from __future__ import annotations

try:
    from training._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from training.split_dataset import audit_raw_dataset, print_audit_summary
from utils.file_utils import ensure_project_directories


def main() -> int:
    ensure_project_directories()
    audit = audit_raw_dataset()

    print("=== DATASET RAW AUDIT ===")
    print_audit_summary(audit, detail_limit=10)

    if audit.raw_image_count == 0:
        print("Chua co du lieu raw.")
        return 1
    if audit.missing_labels or audit.invalid_labels:
        print("Dataset chua sach. Hay sua label truoc khi train.")
        return 1
    print("Dataset hop le de train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
