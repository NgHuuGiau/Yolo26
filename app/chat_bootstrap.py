from __future__ import annotations

from core.hardware_info import detect_hardware
from core.runtime_advisor import select_runtime_config_optimized
from tools.runtime_tool import prompt_runtime_mode


def resolve_start_options(*, requested_mode: str | None, requested_model: str | None) -> tuple[str, str]:
    hardware = detect_hardware()
    selected_mode = requested_mode
    runtime = None
    if selected_mode is None:
        recommendations = {
            mode: select_runtime_config_optimized(mode=mode, hardware=hardware)
            for mode in ("auto", "high", "medium", "low")
        }
        selected_mode = prompt_runtime_mode(hardware=hardware, recommendations=recommendations)
        runtime = recommendations[selected_mode]
    else:
        runtime = select_runtime_config_optimized(mode=selected_mode, hardware=hardware)
    selected_model = requested_model or runtime.primary_model_name
    return selected_mode, selected_model
