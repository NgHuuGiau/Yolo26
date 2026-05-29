from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from core.model_selector import RuntimeConfig, build_candidates, load_settings


def iter_fallback_configs(runtime: RuntimeConfig) -> Iterable[RuntimeConfig]:
    settings = load_settings()
    yielded = set()
    for item in runtime.fallback_chain:
        key = (item["device"], item["model"], item["imgsz"])
        if key in yielded:
            continue
        yielded.add(key)
        yield replace(
            runtime,
            requested_device=item["device"],
            resolved_device="cuda:0" if item["device"] == "gpu" else "cpu",
            primary_model_name=item["model"],
            candidate_models=build_candidates(item["model"], settings),
            imgsz=int(item["imgsz"]),
        )
