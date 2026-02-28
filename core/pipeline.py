from __future__ import annotations

from dataclasses import dataclass, field

from .interfaces import AnalysisPlugin
from .logging import get_logger, sanitize_payload_for_logging

logger = get_logger("core.pipeline")


@dataclass
class PluginRegistry:
    _plugins: dict[str, AnalysisPlugin] = field(default_factory=dict)

    def register(self, plugin: AnalysisPlugin) -> None:
        self._plugins[plugin.plugin_id] = plugin
        logger.info("registered plugin=%s", plugin.plugin_id)

    def list_plugins(self) -> list[str]:
        return sorted(self._plugins.keys())

    def run_plugin(self, plugin_id: str, payload: dict[str, object]) -> dict[str, object]:
        plugin = self._plugins.get(plugin_id)
        if plugin is None:
            return {"ok": False, "error": f"plugin_not_found:{plugin_id}"}

        logger.info(
            "running plugin=%s payload=%s", plugin_id, sanitize_payload_for_logging(payload)
        )
        try:
            result = plugin.run(payload)
        except Exception as exc:  # noqa: PERF203
            logger.exception("plugin failed plugin=%s", plugin_id)
            return {"ok": False, "error": f"plugin_execution_failed:{exc.__class__.__name__}"}

        if not isinstance(result, dict):
            return {"ok": False, "error": "plugin_result_must_be_dict"}
        result.setdefault("ok", True)
        result.setdefault("plugin_id", plugin_id)
        return result
