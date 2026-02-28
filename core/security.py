from __future__ import annotations

import os
from datetime import datetime, timedelta


def _env_bool(name: str, default: bool = False) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def security_headers_guidance() -> tuple[bool, dict[str, str]]:
    payload = {
        "Content-Security-Policy": os.getenv(
            "SECURITY_HEADER_CSP", "default-src 'self'; frame-ancestors 'none'; object-src 'none'"
        ),
        "Strict-Transport-Security": os.getenv(
            "SECURITY_HEADER_HSTS", "max-age=31536000; includeSubDomains"
        ),
        "X-Frame-Options": os.getenv("SECURITY_HEADER_X_FRAME_OPTIONS", "DENY"),
        "X-Content-Type-Options": os.getenv("SECURITY_HEADER_X_CONTENT_TYPE_OPTIONS", "nosniff"),
        "Referrer-Policy": os.getenv(
            "SECURITY_HEADER_REFERRER_POLICY", "strict-origin-when-cross-origin"
        ),
    }
    return _env_bool("SECURITY_HEADERS_ENABLED", default=False), payload


def validate_security_headers_payload(payload: dict[str, str]) -> dict[str, object]:
    required = {
        "Content-Security-Policy",
        "Strict-Transport-Security",
        "X-Frame-Options",
        "X-Content-Type-Options",
        "Referrer-Policy",
    }
    issues: list[str] = []
    for key in sorted(required):
        if not str(payload.get(key, "") or "").strip():
            issues.append(f"missing_{key.lower().replace('-', '_')}")
    return {"ok": not issues, "issues": issues}


def is_session_expired(last_activity_at: datetime | None, *, timeout_minutes: int = 30) -> bool:
    if last_activity_at is None:
        return True
    return datetime.utcnow() - last_activity_at > timedelta(minutes=max(1, int(timeout_minutes)))
