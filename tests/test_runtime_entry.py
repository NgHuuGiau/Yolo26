from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.runtime_entry import run_targeted_entrypoint


class RuntimeEntryTests(unittest.TestCase):
    @patch("app.runtime_entry.BootProgress")
    def test_camera_target_runs_boot_progress_before_dashboard_and_camera(self, boot_progress_mock) -> None:
        progress = MagicMock()
        boot_progress_mock.return_value = progress
        runtime = object()
        hardware = object()
        resolve_start_bundle = MagicMock(
            return_value=SimpleNamespace(
                launch_target="camera",
                selected_mode="medium",
                selected_model="yolo11s.pt",
                runtime=runtime,
                hardware=hardware,
            )
        )
        print_runtime_dashboard = MagicMock()
        run_camera_session = MagicMock()

        result = run_targeted_entrypoint(
            args=SimpleNamespace(mode="medium", model=None, target="camera", camera_index=1),
            preferred_target="camera",
            ui_title="ignored",
            dashboard_title="YOLO Detect Camera",
            resolve_start_bundle_fn=resolve_start_bundle,
            launch_chat_ai_app_fn=MagicMock(),
            print_runtime_dashboard_fn=print_runtime_dashboard,
            run_camera_session_fn=run_camera_session,
        )

        self.assertEqual(result, 0)
        boot_progress_mock.assert_called_once_with("YOLO Detect Camera")
        progress.advance_to.assert_any_call(16, "Đang nhận cấu hình khởi động")
        progress.advance_to.assert_any_call(42, "Đang kiểm tra CPU / GPU / CUDA")
        progress.advance_to.assert_any_call(68, "Đang chọn model và runtime phù hợp")
        progress.advance_to.assert_any_call(88, "Đang chuẩn bị mở camera")
        progress.finish.assert_called_once_with("Sẵn sàng mở camera")
        print_runtime_dashboard.assert_called_once_with(
            title="YOLO Detect Camera",
            runtime=runtime,
            hardware=hardware,
            camera_index=1,
            launch_target="camera",
        )
        run_camera_session.assert_called_once_with(runtime=runtime, camera_index=1)

    @patch("app.runtime_entry.BootProgress")
    def test_ui_target_skips_boot_progress_dashboard_and_camera_path(self, boot_progress_mock) -> None:
        launch_chat_ai_app = MagicMock(return_value=7)
        print_runtime_dashboard = MagicMock()
        run_camera_session = MagicMock()

        result = run_targeted_entrypoint(
            args=SimpleNamespace(mode="medium", model="yolo11s.pt", target="ui", camera_index=0),
            preferred_target="ui",
            ui_title="Chat AI",
            dashboard_title="YOLO Camera Realtime",
            resolve_start_bundle_fn=MagicMock(
                return_value=SimpleNamespace(
                    launch_target="ui",
                    selected_mode="medium",
                    selected_model="yolo11s.pt",
                    runtime=object(),
                    hardware=object(),
                )
            ),
            launch_chat_ai_app_fn=launch_chat_ai_app,
            print_runtime_dashboard_fn=print_runtime_dashboard,
            run_camera_session_fn=run_camera_session,
        )

        self.assertEqual(result, 7)
        launch_chat_ai_app.assert_called_once_with(
            window_title="Chat AI",
            camera_index=0,
            app_mode="medium",
            selected_model="yolo11s.pt",
        )
        boot_progress_mock.assert_not_called()
        print_runtime_dashboard.assert_not_called()
        run_camera_session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
