from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from core.model_selector import RuntimeConfig, build_candidates, load_settings


def _resolve_fallback_device(runtime: RuntimeConfig, requested_device: str) -> str:
    if requested_device == "gpu":
        return "cuda:0"
    if requested_device == "auto":
        return "cuda:0" if str(runtime.resolved_device).startswith("cuda") else "cpu"
    return "cpu"


def iter_fallback_configs(runtime: RuntimeConfig) -> Iterable[RuntimeConfig]:
    settings = load_settings()
    yielded = {
        (
            runtime.requested_device,
            runtime.primary_model_name,
            int(runtime.imgsz),
            runtime.resolved_device,
        )
    }
    for item in runtime.fallback_chain:
        resolved_device = _resolve_fallback_device(runtime, item["device"])
        key = (item["device"], item["model"], int(item["imgsz"]), resolved_device)
        if key in yielded:
            continue
        yielded.add(key)
        yield replace(
            runtime,
            requested_device=item["device"],
            resolved_device=resolved_device,
            primary_model_name=item["model"],
            candidate_models=build_candidates(item["model"], settings),
            imgsz=int(item["imgsz"]),
            active_model_name="",
            use_half=bool(settings["inference"].get("use_half_for_cuda", False)) and resolved_device.startswith("cuda"),
        )
