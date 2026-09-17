"""Optional background-change settings for ground-litter camera profiles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ground_litter_change import ChangeConfig


@dataclass(frozen=True, slots=True)
class BackgroundProfile:
    enabled: bool = False
    reference_image: str | None = None
    change: ChangeConfig = ChangeConfig()
    generate_proposals: bool = False
    require_change: bool = False

    @classmethod
    def from_camera(cls, camera: dict[str, Any], mode: str) -> "BackgroundProfile":
        settings = camera.get(mode, {})
        raw = settings.get("background_change", {})
        change = ChangeConfig(
            delta_threshold=float(raw.get("delta_threshold", 25.0)),
            minimum_change_fraction=float(raw.get("minimum_change_fraction", 0.02)),
            maximum_change_fraction=float(raw.get("maximum_change_fraction", 1.0)),
            minimum_component_area=int(raw.get("minimum_component_area", 24)),
            blur_kernel=int(raw.get("blur_kernel", 3)),
        )
        change.validate()
        for key in ('enabled', 'generate_proposals', 'require_change'):
            if type(raw.get(key, False)) is not bool:
                raise ValueError('background flags must be boolean')
        enabled = raw.get("enabled", False)
        reference = raw.get("reference_image") or settings.get("clean_reference_image")
        if enabled and not reference:
            raise ValueError(f"{mode} background_change requires reference_image")
        return cls(enabled=enabled, reference_image=str(reference) if reference else None,
                   change=change, generate_proposals=raw.get('generate_proposals', False),
                   require_change=raw.get('require_change', False))
