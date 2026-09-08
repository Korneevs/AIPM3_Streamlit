"""Public result export without retired brief diagnostics or mutable aliases."""
from __future__ import annotations

from copy import deepcopy


_PRIVATE_MEDIA_FIELDS = {"objective_runs", "transcripts", "prepared_video"}
_RETIRED_PRESENTATION_FIELDS = {"interpretation", "manager_readout"}
_RETIRED_BRIEF_FIELDS = {
    "brief", "current_brief", "brief_alignment", "message_alignment",
    "brief_details", "brief_details_error", "alignment", "alignment_error",
    "uvp", "rtb", "uvp_rtb", "uvp_rtb_alignment",
}


def export_result(result: dict) -> dict:
    """Remove retired output fields from a copy, never from session/model state."""
    exported = deepcopy(result)
    for key in _PRIVATE_MEDIA_FIELDS | _RETIRED_BRIEF_FIELDS | _RETIRED_PRESENTATION_FIELDS:
        exported.pop(key, None)
    diagnostics = exported.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key in _RETIRED_BRIEF_FIELDS:
            diagnostics.pop(key, None)
        if not diagnostics:
            exported.pop("diagnostics")
    return exported
