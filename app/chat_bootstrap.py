from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from core.hardware_info import detect_hardware
from core.runtime_advisor import select_runtime_config_optimized
from utils.console_ui import prompt_launch_target, prompt_runtime_mode


@dataclass
class StartOptions:
    selected_mode: str
    selected_model: str
    launch_target: str
    hardware: Any
    runtime: Any


DEFAULT_CAMERA_MODE = "high"
DEFAULT_UI_MODE = "medium"


def _resolve_runtime(selected_mode: str, hardware: Any) -> Any:
    return select_runtime_config_optimized(mode=selected_mode, hardware=hardware)


def _default_mode_for_target(requested_target: str | None, preferred_target: str) -> str:
    target = requested_target or preferred_target
    if target == "camera":
        return DEFAULT_CAMERA_MODE
    return DEFAULT_UI_MODE


def _apply_selected_model(runtime: Any, selected_model: str) -> Any:
    if selected_model == getattr(runtime, "primary_model_name", None):
        return runtime
    candidate_models = list(dict.fromkeys([selected_model, *getattr(runtime, "candidate_models", [])]))
    return replace(
        runtime,
        requested_model_name=selected_model,
        primary_model_name=selected_model,
        candidate_models=candidate_models,
    )


def resolve_start_bundle(
    *,
    requested_mode: str | None,
    requested_model: str | None,
    requested_target: str | None,
    preferred_target: str = "ui",
    prompt_runtime_mode_fn=None,
) -> StartOptions:
    prompt_runtime_mode_fn = prompt_runtime_mode_fn or prompt_runtime_mode
    hardware = detect_hardware()
    selected_mode = requested_mode
    runtime = None
    recommendations = {
        mode: _resolve_runtime(mode, hardware)
        for mode in ("auto", "high", "medium", "low")
    }
    if selected_mode is None:
        effective_target = requested_target or preferred_target
        if effective_target == "camera":
            selected_mode = prompt_runtime_mode_fn(hardware=hardware, recommendations=recommendations)
        else:
            selected_mode = _default_mode_for_target(requested_target, preferred_target)
        runtime = recommendations[selected_mode]
    else:
        runtime = recommendations.get(selected_mode) or _resolve_runtime(selected_mode, hardware)

    selected_model = requested_model or runtime.primary_model_name
    runtime = _apply_selected_model(runtime, selected_model)
    if requested_target is not None:
        launch_target = requested_target
    elif preferred_target in {"ui", "camera"}:
        launch_target = preferred_target
    else:
        launch_target = prompt_launch_target(
            selected_mode=selected_mode,
            selected_model=selected_model,
            preferred_target=preferred_target,
        )
    return StartOptions(
        selected_mode=selected_mode,
        selected_model=selected_model,
        launch_target=launch_target,
        hardware=hardware,
        runtime=runtime,
    )


def resolve_start_options(*, requested_mode: str | None, requested_model: str | None) -> tuple[str, str]:
    start_options = resolve_start_bundle(
        requested_mode=requested_mode,
        requested_model=requested_model,
        requested_target="ui",
        preferred_target="ui",
    )
    return start_options.selected_mode, start_options.selected_model
