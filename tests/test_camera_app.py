from __future__ import annotations

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.camera_app import run_camera_entrypoint


class CameraAppTests(unittest.TestCase):
    def test_run_camera_entrypoint_logs_error_message_with_exception_text(self) -> None:
        logger = MagicMock()
        detect_hardware_fn = MagicMock(side_effect=RuntimeError("camera failed"))

        with self.assertRaises(RuntimeError):
            run_camera_entrypoint(
                args=argparse.Namespace(mode="medium", camera_index=0),
                boot_title="Boot",
                dashboard_title="Dashboard",
                boot_finish_message="Ready",
                error_message="Cannot start camera",
                logger=logger,
                detect_hardware_fn=detect_hardware_fn,
                select_runtime_config_fn=MagicMock(),
                run_camera_session_fn=MagicMock(),
                prompt_runtime_mode_fn=MagicMock(return_value="medium"),
            )

        logger.error.assert_called_once_with("%s: %s", "Cannot start camera", unittest.mock.ANY)

    def test_run_camera_entrypoint_uses_recommendations_when_mode_not_supplied(self) -> None:
        hardware = SimpleNamespace(pretty_report=lambda: "hw")
        auto_runtime = object()
        medium_runtime = object()
        runtime_map = {
            "auto": auto_runtime,
            "high": object(),
            "medium": medium_runtime,
            "low": object(),
        }
        select_runtime_config_fn = MagicMock(side_effect=lambda mode, hardware: runtime_map[mode])
        run_camera_session_fn = MagicMock()

        result = run_camera_entrypoint(
            args=argparse.Namespace(mode=None, camera_index=1),
            boot_title="Boot",
            dashboard_title="Dashboard",
            boot_finish_message="Ready",
            error_message="Cannot start camera",
            logger=MagicMock(),
            detect_hardware_fn=MagicMock(return_value=hardware),
            select_runtime_config_fn=select_runtime_config_fn,
            run_camera_session_fn=run_camera_session_fn,
            prompt_runtime_mode_fn=MagicMock(return_value="medium"),
        )

        self.assertEqual(result, 0)
        self.assertEqual(select_runtime_config_fn.call_count, 4)
        run_camera_session_fn.assert_called_once_with(runtime=medium_runtime, camera_index=1)


if __name__ == "__main__":
    unittest.main()
