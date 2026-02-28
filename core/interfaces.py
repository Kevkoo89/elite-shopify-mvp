from __future__ import annotations

from typing import Protocol


class AnalysisPlugin(Protocol):
    plugin_id: str
    display_name: str

    def required_inputs(self) -> dict[str, str]: ...

    def run(self, input_payload: dict[str, object]) -> dict[str, object]: ...
