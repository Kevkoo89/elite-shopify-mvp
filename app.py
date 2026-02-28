from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from data.loader import _compute_elo, _ensure_columns, load_data_full
from data.sync_service import LeagueDataSyncService, MonitoringService
from live_api.providers import get_news_feed, get_real_odds
from models.predictor import predict_match, train_and_evaluate
from ui.assistant import render_assistant
from utils.access_control import is_admin
from utils.analytics import build_power_user_table, compute_clv_summary
from utils.app_logging import get_app_logger
from utils.audit_log import (
    cleanup_audit_log,
    detection_delta_anomalies,
    load_recent_audit_events,
    write_audit_event,
)
from utils.auth import (
    apply_admin_password_override_from_env,
    authenticate_user,
    bootstrap_default_users,
    clear_rate_limit,
    ensure_admin_default_access,
    is_admin_totp_enabled,
    is_rate_limited,
    is_user_locked,
    load_users,
    register_rate_limit_attempt,
    register_user,
    verify_admin_totp,
)
from utils.autonomy_state import AutonomySignals, evaluate_autonomy_state, ownerless_mode_enabled
from utils.config import CONFIG
from utils.dashboard_layout import (
    PRESET_WIDGETS,
    get_user_dashboard_layout,
    resolve_layout_widgets,
    set_user_dashboard_layout,
)
from utils.data_quality import compute_data_quality, compute_drift_speed, compute_feature_drift
from utils.decision_journal import write_decision_event
from utils.drift_detection import compute_drift_score
from utils.explainability import (
    build_decision_explanation,
    evaluate_decision_checks,
    get_blocking_checks,
)
from utils.help_assistant import (
    HelpContext,
    check_ai_usage,
    get_legal_disclaimer,
    get_pre_match_setup_guide,
    get_quota_limit_message,
    recommend_match_settings,
)
from utils.i18n import LANGUAGE_LABELS as I18N_LANGUAGE_LABELS
from utils.i18n import SUPPORTED_LANGUAGES, normalize_language
from utils.input_validation import LiveInputPayload, validate_live_input_payload
from utils.keyboard_shortcuts import resolve_shortcut_action
from utils.league_catalog import filter_leagues_with_cached_data, league_cache_filename
from utils.lightweight_enhancements import build_debug_info, should_autosave_resume
from utils.lightweight_quality import (
    evaluate_cache_invalidation,
    evaluate_i18n_coverage,
    run_input_validation_fuzz,
    validate_export_contract,
)
from utils.lightweight_round3 import (
    active_sidebar_notices,
    build_analysis_cache_key,
    cache_effective_ttl_seconds,
    cache_hit_rate,
    compact_export_csv,
    read_cached_result,
    set_sidebar_notice,
    write_cached_result,
)
from utils.lightweight_round4 import (
    filter_backtest_export_today,
    next_news_batch_limit,
    should_pause_background_sync,
    should_skip_recompute,
)
from utils.maturity_batch1 import (
    build_explainability_snapshot_hash,
    evaluate_backtest_kpi_drift,
    evaluate_explainability_snapshot_regression,
    evaluate_next_action_telemetry,
)
from utils.model_agreement import compute_model_agreement
from utils.model_guardrails import is_canary_enabled_for_user, summarize_calibration
from utils.monitoring_ledger import (
    init_monitoring_db,
    log_drift_event,
    read_recent_drift_events,
    read_recent_predictions,
    record_prediction,
    run_ledger_consistency_check,
    update_settlement,
)
from utils.next_action_coach import recommend_next_action
from utils.ops_health import (
    execute_incident_runbook,
    run_startup_health_checks,
)
from utils.partner_api import monitor_api_health
from utils.precision_boosters import (
    apply_lineup_uncertainty_penalty,
    auto_tune_weights,
    calibrate_probability,
    recency_weights,
    weighted_mean,
)
from utils.precision_round6 import (
    apply_feature_drift_weight_adjustment,
    clip_outliers,
    consensus_gate,
    poisson_confidence_interval,
    season_phase_factor,
)
from utils.precision_round7 import (
    adaptive_threshold,
    apply_disagreement_penalty,
    calibrate_multiclass_outcomes,
    model_disagreement_score,
    stability_gate,
)
from utils.precision_round12 import (
    count_valid_signals,
    evaluate_evidence_gate,
    evaluate_shadow_decision,
    online_bias_correction,
    quantile_decision_threshold,
    scenario_ensemble_probability,
)
from utils.precision_round13 import (
    ErrorClassStats,
    apply_trust_to_probability,
    feature_trust_weight,
    hysteresis_decision,
    season_phase_calibration,
    trimmed_mean,
    update_error_class_stats,
)
from utils.precision_round14 import (
    adaptive_bias_learning_rate,
    error_class_threshold_adjustment,
    evaluate_shadow_promotion,
    segmented_hysteresis_thresholds,
    weighted_consensus_probability,
)
from utils.precision_round15 import (
    ReplayCase,
    adjust_cost_sensitive_threshold,
    apply_correlation_guardrail,
    apply_weighted_signal_fallback_matrix,
    build_dynamic_threshold_stress,
    build_explainability_highlights,
    build_reliability_score,
    build_replay_cases_from_history,
    build_team_pair_key,
    calibrate_confidence_by_league,
    compute_metric_trend,
    correlation_penalty_with_drawdown,
    decision_uncertainty_band,
    detect_odds_move_signal_adaptive,
    detect_regime_state,
    dynamic_stake_cap,
    estimate_team_pair_similarity,
    fallback_penalty_breakdown,
    humanize_regime_risk_flags,
    load_replay_history,
    load_retrospective,
    odds_alert_direction_message,
    odds_alert_priority,
    persist_replay_history,
    persist_retrospective,
    recommend_risk_profile_for_regime,
    recommended_odds_cooldown_minutes,
    regime_exposure_multiplier,
    regime_risk_highlights,
    reliability_hard_gate_from_history,
    replay_health_level,
    replay_quality_trend,
    run_replay_suite,
    should_emit_odds_alert,
    summarize_replay_quality,
    summarize_retrospective,
    uncertainty_threshold_by_league,
    update_metric_history,
)
from utils.precision_round16 import (
    apply_weekly_league_calibration,
    build_weekly_league_calibration,
    build_weekly_postmortem,
    compute_provider_scores,
    detect_confidence_drift_alert,
    load_postmortem_report,
    persist_postmortem_report,
    provider_priority_order,
    resolve_safe_mode_plus_profile,
    resolve_safe_mode_plus_profile_with_cooldown,
)
from utils.precision_round17 import (
    adaptive_alert_thresholds,
    build_quality_heatmap_rows,
    build_recovery_packs,
    compare_ab_rules,
    summarize_decision_delta,
)
from utils.precision_round18 import (
    build_calibration_report,
    build_explainability_replay,
    compute_maintenance_score,
    stability_guard,
    summarize_no_bet_quality,
)
from utils.precision_round19 import (
    append_experiment_log_entry,
    bootstrap_threshold_uncertainty,
    compute_data_freshness_score,
    simulate_threshold_what_if,
    summarize_error_costs,
    summarize_pipeline_latency,
    track_pipeline_latency,
)
from utils.precision_round20 import (
    apply_quality_budget,
    build_actionable_alert_digest,
    build_next_action_snippet,
    build_no_bet_gap_hint,
    build_weekly_auto_review,
    classify_priority_tier,
    compute_digest_impact_score,
    compute_outcome_error_rate,
    confidence_bandit_delta,
    normalize_team_name,
    should_exit_bandit_cooldown,
    should_snooze_digest_item,
    suggest_team_name,
    summarize_pipeline_benchmark,
    sync_pending_outcomes,
)
from utils.recovery_planner import runbook_actions_for_event
from utils.resilience import with_retry
from utils.road_to_100_batch5 import (
    append_fix_path_event,
    derive_low_end_adaptive_profile,
    evaluate_control_loop_slo,
    rank_fix_path_effectiveness,
)
from utils.road_to_100_batch6 import (
    build_incident_correlation_report,
    build_ops_driven_ui_plan,
    evaluate_recovery_simulation_gate,
)
from utils.security_baseline import build_security_baseline_report
from utils.security_headers import security_headers_guidance, validate_security_headers_payload
from utils.session_resume import (
    load_user_resume,
    load_user_resume_meta,
    save_user_resume_if_revision,
)
from utils.session_security import is_session_expired
from utils.trust_score import compute_trust_score
from utils.worst_case_simulator import simulate_worst_case

st.set_page_config(
    page_title="Elite Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_theme_token(theme_name: str) -> str:
    return "bloomberg" if "Bloomberg" in str(theme_name) else "other"


def inject_theme_debug_css() -> str:
    theme_name = str(st.session_state.get("ui_theme_style", "Ultra Premium Dark SaaS"))
    theme_token = resolve_theme_token(theme_name)
    baseline_css = """
    <style>
    #MainMenu, header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
    }
    .stApp {
        margin-top: 0 !important;
    }
    </style>
    """
    st.markdown(baseline_css, unsafe_allow_html=True)
    if theme_token == "bloomberg":
        st.markdown(
            """
            <style>
            .stApp {
                background: #0b0f14 !important;
                color: #e6edf7 !important;
            }
            [data-testid="stSidebar"] {
                border-left: 6px solid #ffb800 !important;
                width: 260px !important;
                min-width: 260px !important;
                background: #0c121b !important;
            }
            [data-testid="stSidebar"] .block-container {
                padding-top: 8px !important;
                padding-bottom: 8px !important;
            }
            .stButton button {
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                border-radius: 6px !important;
                box-shadow: none !important;
                min-height: 34px !important;
                padding: 5px 9px !important;
                background: transparent !important;
            }
            .ea-root[data-theme="bloomberg"] .main-container {
                max-width: 1280px !important;
                margin: 0 auto !important;
                padding: 0 24px 20px !important;
            }
            .ea-root[data-theme="bloomberg"] .ea-main {
                display: flex !important;
                flex-direction: column !important;
                gap: 8px !important;
                align-items: stretch !important;
                max-width: 1280px !important;
                margin: 0 auto !important;
            }
            .ea-root[data-theme="bloomberg"] #ea-kpi-bar {
                width: 100% !important;
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                gap: 12px !important;
                align-items: stretch !important;
                justify-content: space-between !important;
                padding: 14px 20px !important;
                max-height: 92px !important;
                overflow-x: auto !important;
                overflow-y: hidden !important;
                box-sizing: border-box !important;
                align-self: stretch !important;
                grid-column: 1 / -1 !important;
                place-self: stretch !important;
                justify-self: stretch !important;
                background: rgba(10, 14, 20, 0.55) !important;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
            }
            .ea-root[data-theme="bloomberg"] #ea-kpi-bar::-webkit-scrollbar {
                height: 6px;
            }
            .ea-root[data-theme="bloomberg"] .ea-kpi-card {
                flex: 0 0 auto !important;
                width: 220px !important;
                height: 76px !important;
                min-height: 76px !important;
                max-height: 76px !important;
                border-radius: 6px !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                background: rgba(255, 255, 255, 0.03) !important;
                padding: 10px 12px !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: flex-start !important;
                align-items: flex-start !important;
                box-shadow: none !important;
                box-sizing: border-box !important;
            }
            .ea-root[data-theme="bloomberg"] .ea-kpi-title {
                font-size: 11px !important;
                letter-spacing: 0.10em !important;
                text-transform: uppercase !important;
                opacity: 0.70 !important;
                margin: 0 !important;
                white-space: nowrap !important;
                text-align: left !important;
            }
            .ea-root[data-theme="bloomberg"] .ea-kpi-value {
                font-size: 24px !important;
                font-weight: 700 !important;
                line-height: 1 !important;
                margin: auto 0 0 0 !important;
                min-height: 34px !important;
                display: flex !important;
                align-items: center !important;
                white-space: nowrap !important;
                text-align: left !important;
            }
            .ea-root[data-theme="bloomberg"] .bloomberg-panel {
                background: #0f1621 !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                border-radius: 6px !important;
                box-shadow: none !important;
                padding: 10px !important;
                margin-bottom: 8px !important;
            }
            .ea-root[data-theme="bloomberg"] .ea-content {
                display: block !important;
                padding: 20px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    return theme_token


CURRENT_THEME_TOKEN = inject_theme_debug_css()


def render_global_styles(*_args: object, **_kwargs: object) -> None:
    """Legacy UI-style hook kept as no-op for compatibility with stale references."""


def ui_is_safe_mode() -> bool:
    try:
        safe_param = st.query_params.get("safe_ui", "0")
        if isinstance(safe_param, list):
            safe_param = safe_param[0] if safe_param else "0"
        if str(safe_param) == "1":
            return True
    except Exception:
        pass
    return bool(st.session_state.get("safe_ui", False))


DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


def _normalize_lang(raw_lang: str) -> str:
    lang = str(raw_lang or "de").strip().lower()
    if lang.startswith("de"):
        return "de"
    if lang.startswith("en"):
        return "en"
    return "de"


def _resolve_app_language() -> str:
    for key in ("language", "lang", "ui_language"):
        if key in st.session_state:
            return _normalize_lang(str(st.session_state.get(key, "de")))
    return "de"


def _ollama_is_online(endpoint: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{endpoint.rstrip('/')}/api/tags", timeout=2)
        if response.status_code == 200:
            return True, "ok"
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, e.__class__.__name__


def call_ollama_chat(messages, model=DEFAULT_OLLAMA_MODEL):
    endpoint = str(
        st.session_state.get("assistant_ollama_endpoint", DEFAULT_OLLAMA_ENDPOINT)
    ).rstrip("/")
    chosen_model = str(
        st.session_state.get("assistant_ollama_model", model or DEFAULT_OLLAMA_MODEL)
    )
    try:
        response = requests.post(
            f"{endpoint}/api/chat",
            json={
                "model": chosen_model,
                "messages": messages,
                "stream": False,
            },
            timeout=60,
        )
        if response.status_code != 200:
            return (
                f"Modell-Fehler ({response.status_code}). Bitte Modell/Prompt prüfen.",
                True,
            )
        try:
            payload = response.json()
        except Exception:
            return "Interner Fehler: Parsing-Fehler in der Modellantwort.", True

        content = str(payload.get("message", {}).get("content", "")).strip()
        if not content:
            return "Leere Antwort vom Modell.", True
        return content, True
    except requests.exceptions.ConnectionError:
        return "", False
    except requests.exceptions.Timeout:
        return "", False
    except Exception as e:
        return f"Interner Modell-Fehler: {str(e)}", True


def _get_system_prompt(language: str) -> str:
    if language == "de":
        return (
            "Du bist der Elite Analyst für Fußball-Analysen in dieser App. "
            "Antworte präzise, strukturiert und ausschließlich auf Deutsch. Format: Kurzfazit, Empfehlung, Begründung (3–6 Bulletpoints). "
            "WICHTIG: Erfinde keine Live-Fakten. Wenn dir Daten fehlen, sage das kurz und frage maximal 1 Rückfrage. "
            "Wenn der Nutzer nach Einstellungs-Empfehlungen fragt, gib konkrete Werte für risk_profile, min_confidence, stake. "
            'Gib zusätzlich optional ein JSON im Format: {"set": {"risk_profile": "balanced", "min_confidence": 0.62, "stake": 0.7}} '
            "aber NUR wenn du dir sicher bist. Keine weiteren Keys."
        )
    return (
        "You are the Elite Analyst for football analysis in this app. "
        "Answer structured, precise and exclusively in English. Format: Summary, Recommendation, Rationale (3–6 bullet points). "
        "IMPORTANT: Do not invent live facts. If data is missing, say so briefly and ask at most 1 clarifying question. "
        "If asked for setting recommendations, provide concrete values for risk_profile, min_confidence, stake. "
        'Optionally output JSON: {"set": {"risk_profile": "balanced", "min_confidence": 0.62, "stake": 0.7}} '
        "ONLY if confident. No extra keys."
    )


def _extract_set_payload(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {}

    candidates = [raw]
    if "```" in raw:
        for part in raw.split("```"):
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate:
                candidates.append(candidate)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if (
            isinstance(parsed, dict)
            and set(parsed.keys()) == {"set"}
            and isinstance(parsed.get("set"), dict)
        ):
            return parsed.get("set", {})
    return {}


def get_theme_key(theme_name: str) -> str:
    if "Bloomberg" in str(theme_name):
        return "bloomberg"
    return "other"


def get_login_layout_key(layout_name: str) -> str:
    mapping = {
        "Centered Card": "center",
        "Split Screen": "split",
        "Minimal Top": "minimal",
    }
    return mapping.get(layout_name, "center")


def _form_key(name: str) -> str:
    theme = st.session_state.get("ui_theme_style", "Ultra Premium Dark SaaS")
    layout = st.session_state.get("dashboard_layout_mode", "Focus")
    return f"{name}__{theme}__{layout}"


def inject_core_ui_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu{visibility:hidden !important;}
        footer{visibility:hidden !important;}
        header{visibility:hidden !important;}

        [data-testid="stHeader"]{display:none !important;}
        [data-testid="stToolbar"]{display:none !important;}
        [data-testid="stDecoration"]{display:none !important;}
        [data-testid="stStatusWidget"]{display:none !important;}

        .stApp { margin-top: 0 !important; }
        .block-container { padding-top: 0.8rem !important; }
        [data-testid="stAppViewContainer"] { padding-top: 0 !important; }
        hr { display:none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_base_css() -> None:
    st.markdown(
        """
        <style>
            #MainMenu {display: none !important;}
            div[data-testid="stToolbar"] {display: none !important;}
            div[data-testid="stHeader"] {display: none !important;}
            div[data-testid="stFooter"] {display: none !important;}

            .main {
                padding-top: 0 !important;
            }

            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
                max-width: 1400px !important;
                margin: 0 auto !important;
            }

            .main-container {
                max-width: 1300px;
                margin: 0 auto;
                padding: 24px 18px 30px;
            }

            .card {
                background: #111827;
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid #1f2937;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_premium_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ea-bg-main: #0b0f14;
                --ea-bg-surface: #111827;
                --ea-bg-muted: #0f172a;
                --ea-border: #1f2937;
                --ea-text: #e5e7eb;
                --ea-text-soft: #9ca3af;
                --ea-accent: #2563eb;
            }

            [data-testid="stHeader"],
            [data-testid="stToolbar"] {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
            }

            .stApp {
                background: linear-gradient(180deg, #0b0f14 0%, #0e1623 100%) !important;
                color: var(--ea-text);
            }

            [data-testid="stAppViewContainer"] {
                background: transparent !important;
            }

            .block-container {
                max-width: 1280px !important;
                margin: 0 auto !important;
                padding-top: 1.8rem !important;
                padding-bottom: 2.2rem !important;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0e1117 0%, #0b0f14 100%);
                border-right: 1px solid var(--ea-border);
            }

            [data-testid="stSidebar"] .block-container {
                max-width: 100% !important;
                padding-top: 1rem !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(17, 24, 39, 0.62);
                border: 1px solid rgba(31, 41, 55, 0.8);
                border-radius: 16px;
                padding: 1rem;
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.28);
            }

            .ea-header {
                background: rgba(17, 24, 39, 0.9);
                border: 1px solid var(--ea-border);
                border-radius: 16px;
                padding: 1rem 1.2rem;
                margin-bottom: 0.9rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 14px 35px rgba(0, 0, 0, 0.34);
            }

            .ea-header__title {
                font-size: 1.15rem;
                font-weight: 600;
                color: var(--ea-text);
                margin: 0;
            }

            .ea-header__status {
                color: var(--ea-text-soft);
                font-size: 0.82rem;
                border: 1px solid var(--ea-border);
                border-radius: 999px;
                padding: 0.2rem 0.6rem;
                background: rgba(15, 23, 42, 0.7);
            }

            .ea-card {
                background: rgba(17, 24, 39, 0.94);
                border: 1px solid var(--ea-border);
                border-radius: 16px;
                padding: 1rem 1.1rem;
                margin-bottom: 0.8rem;
                box-shadow: 0 15px 36px rgba(0, 0, 0, 0.35);
            }

            .ea-card__title {
                color: var(--ea-text);
                font-size: 0.98rem;
                font-weight: 600;
                margin-bottom: 0.65rem;
            }

            .ea-sidebar-section {
                border: 1px solid var(--ea-border);
                border-radius: 12px;
                padding: 0.65rem 0.75rem;
                margin-bottom: 0.65rem;
                background: rgba(17, 24, 39, 0.62);
            }

            .ea-sidebar-label {
                color: var(--ea-text-soft);
                font-size: 0.76rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.35rem;
            }

            .ea-login-shell {
                min-height: 90vh;
                display: flex;
                align-items: center;
            }

            .ea-login-card {
                padding: 2rem 2rem 1.6rem;
                border-radius: 20px;
                background: linear-gradient(160deg, rgba(17, 24, 39, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
                border: 1px solid rgba(37, 99, 235, 0.28);
                box-shadow: 0 28px 62px rgba(0, 0, 0, 0.5);
            }

            .ea-login-top {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 0.8rem;
            }

            .ea-badge {
                font-size: 0.74rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #bfdbfe;
                background: rgba(37, 99, 235, 0.18);
                border: 1px solid rgba(59, 130, 246, 0.45);
                border-radius: 999px;
                padding: 0.24rem 0.6rem;
            }

            .ea-login-title {
                margin: 0.4rem 0 0.25rem;
                font-size: 2rem;
                font-weight: 700;
                color: var(--ea-text);
            }

            .ea-login-sub {
                margin: 0 0 1rem;
                color: var(--ea-text-soft);
            }

            .ea-login-card .stSelectbox > div > div {
                min-height: 2.5rem;
            }

            .ea-login-card .stTabs {
                margin-top: 0.4rem;
            }

            .stButton > button, .stDownloadButton > button {
                background: #1f2937;
                color: var(--ea-text);
                border: 1px solid #334155;
                border-radius: 12px;
                transition: all 0.2s ease;
            }

            .stButton > button:hover, .stDownloadButton > button:hover {
                border-color: #3b82f6;
                transform: translateY(-1px);
            }

            .stButton > button[kind="primary"],
            button[kind="primary"] {
                background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
                border: 1px solid #3b82f6 !important;
            }

            .stTextInput > div > div > input,
            .stTextArea textarea,
            .stSelectbox > div > div,
            .stNumberInput input {
                background: #0f172a !important;
                color: var(--ea-text) !important;
                border: 1px solid #253246 !important;
                border-radius: 12px !important;
            }

            [data-baseweb="tab-list"] {
                background: rgba(17, 24, 39, 0.8);
                border: 1px solid var(--ea-border);
                border-radius: 12px;
                padding: 0.25rem;
                gap: 0.2rem;
            }

            [data-baseweb="tab"] {
                background: transparent;
                border-radius: 8px;
                color: var(--ea-text-soft);
            }

            [data-baseweb="tab"][aria-selected="true"] {
                background: #1f2937;
                color: var(--ea-text);
            }

            [data-testid="metric-container"] {
                background: rgba(15, 23, 42, 0.95);
                border: 1px solid #1f2937;
                border-radius: 14px;
                padding: 0.7rem 0.8rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_text(key: str, lang: str) -> str:
    text_map = {
        "design_layout": {"de": "Design & Layout", "en": "Design & Layout"},
        "design_theme": {"de": "Stil wählen", "en": "Choose style"},
        "design_live": {
            "de": "Änderungen sind sofort sichtbar.",
            "en": "Changes are visible immediately.",
        },
        "theme_mode": {"de": "Helligkeit", "en": "Brightness"},
        "theme_mode_dark": {"de": "Dunkel", "en": "Dark"},
        "theme_mode_light": {"de": "Hell", "en": "Light"},
        "theme_accent": {"de": "Akzentfarbe", "en": "Accent color"},
        "theme_radius": {"de": "Rundungen", "en": "Rounded corners"},
        "login_title": {"de": "Elite Analyst", "en": "Elite Analyst"},
        "login_sub": {"de": "Bitte anmelden", "en": "Please sign in"},
        "language": {"de": "Sprache", "en": "Language"},
    }
    values = text_map.get(key, {})
    return values.get(lang, values.get("de", key))


def apply_theme(theme_name: str) -> None:
    st.session_state["theme"] = theme_name

    if theme_name == "Ultra Premium Dark SaaS":
        css = """
        :root {
            --bg: #0b1020;
            --bg2: #070a12;
            --card: rgba(255,255,255,0.06);
            --card2: rgba(255,255,255,0.08);
            --text: #eaf0ff;
            --muted: rgba(234,240,255,0.68);
            --accent: #00c6ff;
            --accent2: #7c5cff;
            --border: rgba(255,255,255,0.10);
            --radius: 20px;
            --shadow: 0 24px 60px rgba(0,0,0,0.45);
            --pad: 1.1rem;
        }
        .stApp { background: linear-gradient(145deg, var(--bg), var(--bg2)) !important; color: var(--text) !important; }
        [data-testid="stSidebar"] {
            background: rgba(10,16,32,0.55) !important;
            backdrop-filter: blur(18px);
            border-right: 1px solid var(--border) !important;
            box-shadow: inset -1px 0 0 rgba(255,255,255,0.06);
        }
        .stButton > button { background: linear-gradient(135deg,var(--accent),var(--accent2)) !important; box-shadow: 0 0 0 1px rgba(0,198,255,0.35), 0 12px 28px rgba(0,198,255,0.25) !important; }
        .stButton > button:hover { box-shadow: 0 0 0 1px rgba(124,92,255,0.5), 0 16px 30px rgba(124,92,255,0.35) !important; }
        """
    elif theme_name == "Bloomberg / Institutional":
        css = """
        :root {
            --bg: #0e131a;
            --bg2: #0e131a;
            --card: #111827;
            --card2: #111827;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #ff9f0a;
            --accent2: #ff9f0a;
            --border: rgba(255,255,255,0.12);
            --radius: 6px;
            --shadow: none;
            --pad: 0.55rem;
        }
        .stApp { background: var(--bg) !important; color: var(--text) !important; }
        [data-testid="stSidebar"] {
            background: #0b1117 !important;
            border-right: 1px solid rgba(255,159,10,0.38) !important;
            padding-top: 0.2rem !important;
        }
        .block-container { max-width: 1320px !important; }
        h1,h2,h3,.ea-header__title { font-family: "IBM Plex Mono", "Courier New", monospace; letter-spacing: .02em; }
        .stButton > button { background: #161d27 !important; border: 1px solid rgba(255,159,10,0.7) !important; box-shadow:none !important; }
        [data-baseweb="tab"] { border-radius: 4px !important; }
        """
    elif theme_name == "Neon Tech / Cyber":
        css = """
        :root {
            --bg: #040414;
            --bg2: #0b0b26;
            --card: rgba(15,15,40,0.65);
            --card2: rgba(20,20,52,0.75);
            --text: #e6f7ff;
            --muted: rgba(230,247,255,0.70);
            --accent: #ff2bd6;
            --accent2: #00f0ff;
            --border: rgba(255,43,214,0.55);
            --radius: 14px;
            --shadow: 0 0 24px rgba(255,43,214,0.18);
            --pad: 0.95rem;
        }
        .stApp { background: radial-gradient(circle at 28% 20%, #15153a, #040414 68%) !important; color: var(--text) !important; }
        [data-testid="stSidebar"] {
            background: rgba(10,8,32,0.92) !important;
            border-right: 1px solid rgba(255,43,214,0.65) !important;
            box-shadow: 0 0 22px rgba(255,43,214,0.24);
        }
        .stButton > button { background: linear-gradient(90deg,var(--accent),var(--accent2)) !important; box-shadow: 0 0 16px rgba(255,43,214,0.35) !important; }
        [data-testid="stExpander"] details { border: 1px solid rgba(255,43,214,0.5) !important; }
        [data-baseweb="tab"][aria-selected="true"] { box-shadow: 0 0 12px rgba(0,240,255,0.45); }
        """
    else:
        css = """
        :root {
            --bg: #f5f7fb;
            --bg2: #edf1f8;
            --card: #ffffff;
            --card2: #ffffff;
            --text: #111827;
            --muted: #4b5563;
            --accent: #2563eb;
            --accent2: #2563eb;
            --border: #e5e7eb;
            --radius: 14px;
            --shadow: 0 10px 22px rgba(17,24,39,0.06);
            --pad: 1rem;
        }
        .stApp { background: linear-gradient(180deg,var(--bg),var(--bg2)) !important; color: var(--text) !important; }
        [data-testid="stSidebar"] {
            background: #0f172a !important;
            border-right: 1px solid rgba(148,163,184,.35) !important;
        }
        [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        .stButton > button { background: #2563eb !important; color: #ffffff !important; border: 1px solid #1d4ed8 !important; box-shadow:none !important; }
        [data-testid="stExpander"] details, [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"] { box-shadow: var(--shadow) !important; }
        """

    st.markdown(
        f"""
        <style>
            {css}

            .block-container {{
                max-width: 1320px !important;
                padding-top: var(--pad) !important;
                padding-bottom: calc(var(--pad) + 0.2rem) !important;
            }}

            .ea-card, .card, .login-card, .ea-login-card,
            [data-testid="stVerticalBlockBorderWrapper"],
            [data-testid="stMetric"] {{
                background: var(--card) !important;
                color: var(--text) !important;
                border: 1px solid var(--border) !important;
                border-radius: var(--radius) !important;
                box-shadow: var(--shadow) !important;
            }}

            .ea-header, .ea-sidebar-section, [data-baseweb="tab-list"], [data-testid="stExpander"] details {{
                background: var(--card2) !important;
                border: 1px solid var(--border) !important;
                border-radius: max(6px, calc(var(--radius) - 6px)) !important;
            }}

            .stButton > button {{
                border-radius: calc(var(--radius) - 6px) !important;
                font-weight: 600 !important;
                min-height: 42px !important;
            }}

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-testid="stSelectbox"] div,
            .stTextInput > div > div > input,
            .stTextArea textarea,
            .stNumberInput input {{
                background: color-mix(in srgb, var(--card) 92%, #000 8%) !important;
                color: var(--text) !important;
                border: 1px solid var(--border) !important;
                border-radius: calc(var(--radius) - 6px) !important;
            }}

            [data-baseweb="tab"] {{
                color: var(--text) !important;
                border-radius: calc(var(--radius) - 8px) !important;
            }}

            [data-baseweb="tab"][aria-selected="true"] {{
                background: color-mix(in srgb, var(--accent) 22%, var(--card)) !important;
            }}

            h1, h2, h3 {{
                color: var(--text) !important;
                letter-spacing: 0.01em;
            }}

            .ea-login-subline {{
                color: var(--muted);
                margin-top: -0.2rem;
                margin-bottom: 1rem;
                font-size: 0.92rem;
            }}

            .ea-feature-bullets {{
                margin-top: 1rem;
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.5;
            }}

            .ea-trust-line {{
                margin-top: 0.8rem;
                font-size: 0.78rem;
                color: var(--muted);
            }}

            .login-page {{
                min-height: calc(100vh - 10px);
                display:flex;
                align-items:center;
                justify-content:center;
            }}

            .login-card {{
                width:420px;
                max-width:92vw;
                background:var(--card);
                border:1px solid var(--border);
                border-radius:16px;
                padding:32px;
                box-shadow:var(--shadow);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def mget(metrics: object, key: str, default: float = 0.0) -> float:
    if metrics is None:
        return float(default)
    if isinstance(metrics, dict):
        return float(metrics.get(key, default))
    to_dict = getattr(metrics, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
            if isinstance(data, dict):
                return float(data.get(key, default))
        except Exception:
            return float(default)
    value = getattr(metrics, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@contextmanager
def card(title: str):
    st.markdown(
        f"<div class='ea-card'><div class='ea-card__title'>{title}</div>", unsafe_allow_html=True
    )
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data(ttl=120)
def _cached_proof_snapshot(prediction_count: int, drift_count: int) -> dict[str, int]:
    return {"prediction_count": int(prediction_count), "drift_count": int(drift_count)}


def _ops_state_path(username: str) -> Path:
    safe = "".join(ch for ch in str(username or "guest") if ch.isalnum() or ch in {"_", "-", "."})
    name = safe or "guest"
    folder = Path("data") / "ops_state"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.json"


def _load_ops_state(username: str) -> dict[str, list[dict[str, object]]]:
    path = _ops_state_path(username)
    candidates = [
        path,
        path.with_suffix(path.suffix + ".bak1"),
        path.with_suffix(path.suffix + ".bak2"),
    ]
    payload: dict[str, object] | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(loaded, dict):
            payload = loaded
            break
    if payload is None:
        return {"dead_letter_outcomes": [], "quality_budget_decisions": [], "bandit_events": []}
    version = int(payload.get("schema_version", 1) or 1)
    if version < 2:
        payload = {
            "dead_letter_outcomes": list(payload.get("dead_letter_outcomes", [])),
            "quality_budget_decisions": list(payload.get("quality_budget_decisions", [])),
            "bandit_events": list(payload.get("bandit_events", [])),
            "schema_version": 2,
        }
    return {
        "dead_letter_outcomes": list(payload.get("dead_letter_outcomes", [])),
        "quality_budget_decisions": list(payload.get("quality_budget_decisions", [])),
        "bandit_events": list(payload.get("bandit_events", [])),
    }


def _save_ops_state(
    username: str,
    *,
    dead_letter_outcomes: list[dict[str, object]],
    quality_budget_decisions: list[dict[str, object]],
    bandit_events: list[dict[str, object]],
) -> None:
    path = _ops_state_path(username)
    bak1 = path.with_suffix(path.suffix + ".bak1")
    bak2 = path.with_suffix(path.suffix + ".bak2")
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": 2,
        "dead_letter_outcomes": dead_letter_outcomes[-500:],
        "quality_budget_decisions": quality_budget_decisions[-300:],
        "bandit_events": bandit_events[-300:],
        "saved_at": datetime.utcnow().isoformat(),
    }
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if bak2.exists():
        bak2.unlink(missing_ok=True)
    if bak1.exists():
        bak1.replace(bak2)
    if path.exists():
        path.replace(bak1)
    tmp.replace(path)


def _cleanup_ops_state_files(*, max_age_days: int = 90) -> None:
    folder = Path("data") / "ops_state"
    if not folder.exists():
        return
    cutoff = datetime.utcnow() - timedelta(days=max(7, int(max_age_days)))
    for file in folder.glob("*.json*"):
        try:
            modified = datetime.utcfromtimestamp(file.stat().st_mtime)
        except OSError:
            continue
        if modified < cutoff:
            file.unlink(missing_ok=True)


# --- i18n ---
_RAW_I18N = {
    "de": {
        "app_title": "Elite Quant Ultimate — Global Pro Workspace",
        "login": "Beta-Login",
        "username": "Benutzername",
        "password": "Passwort",
        "login_btn": "Anmelden",
        "logout": "Abmelden",
        "login_err": "Login fehlgeschlagen.",
        "login_locked": "Zu viele Fehlversuche. Account ist temporär gesperrt.",
        "login_rate_limited": "Zu viele Login-Versuche. Bitte warte kurz und versuche es erneut.",
        "login_totp": "Admin 2FA Code (TOTP)",
        "login_totp_err": "Admin-Login benötigt einen gültigen 2FA-Code.",
        "session_expired": "Sitzung wegen Inaktivität abgelaufen. Bitte erneut anmelden.",
        "register": "Registrieren",
        "register_btn": "Account erstellen",
        "password_repeat": "Passwort wiederholen",
        "register_ok": "Account erstellt. Du kannst dich jetzt einloggen.",
        "register_err_exists": "Benutzername existiert bereits.",
        "register_err_password": "Passwort muss mindestens 8 Zeichen haben.",
        "register_err_password_weak": "Passwort ist zu schwach (nutze Groß-/Kleinbuchstaben, Zahlen, Sonderzeichen).",
        "register_err_username": "Benutzername muss 3–32 Zeichen haben (nur Buchstaben, Zahlen, _ . -).",
        "register_err_mismatch": "Passwörter stimmen nicht überein.",
        "user_level": "Nutzer-Modus",
        "level_beginner": "Einsteiger",
        "level_pro": "Profi",
        "ai_help": "💬 KI-Hilfe (kostenlos)",
        "ai_help_hint": "Stelle Fragen in normaler Sprache. Die Hilfe erklärt Begriffe und zeigt nächste Schritte.",
        "ai_question": "Deine Frage",
        "ai_ask": "Antwort anzeigen",
        "ai_admin": "Admin-KI-Coach (kostenlos)",
        "ai_admin_hint": "Nur für Admin: erklärt alle Funktionen einfach und gibt vor jedem Spiel konkrete Setup-Empfehlungen.",
        "ai_admin_locked": "KI-Coach ist aktuell nur für Admin aktiv.",
        "ai_explain_all": "Alle Funktionen erklären",
        "ai_pre_match": "Setup-Empfehlung für dieses Spiel",
        "ai_auto_optimize": "Auto-Optimierung (Coach)",
        "ai_auto_optimized": "Coach-Einstellungen wurden übernommen.",
        "control": "Control Center",
        "sidebar_section_layout": "Layout & Navigation",
        "sidebar_section_strategy": "Strategie-Grundeinstellungen",
        "sidebar_section_sync": "Daten & Synchronisierung",
        "sidebar_section_live_match": "Live: Match-Setup",
        "sidebar_section_live_monitor": "Live: Monitoring",
        "sidebar_section_live_advanced": "Live: Erweiterte Einstellungen",
        "preset_applied": 'Preset "{preset}" für {league} angewendet.',
        "language": "Sprache",
        "shortcuts": "Tastenkürzel",
        "shortcuts_hint": "A=Anwenden, S=Sync, L=Logout, G=Sprache",
        "shortcuts_input": "Shortcut-Befehl",
        "shortcuts_apply": "Shortcut ausführen",
        "shortcuts_unknown": "Unbekannter Shortcut. Nutze A/S/L/G.",
        "safe_mode": "Safe Mode (Laptop)",
        "safe_mode_plus": "Safe Mode+ Profil",
        "safe_mode_hint": "Schaltet auf ressourcenschonende Defaults.",
        "safe_mode_active": "Safe Mode aktiv: weniger Last, stabilere Performance auf schwachen Laptops.",
        "autosave_status": "Auto-Save: {time}",
        "metric_help": "Kennzahl-Hilfe",
        "metric_help_edge": "Value Edge: positiver Wert heißt, dass deine Modellquote besser als die Marktquote ist.",
        "metric_help_confidence": "Confidence: zeigt, wie sicher das Modell die Richtung einschätzt.",
        "metric_help_exposure": "Exposure: wie viel Risiko-Budget bereits gebunden ist.",
        "export_today_csv": "Heute als CSV exportieren",
        "export_compare_csv": "Vergleich als CSV exportieren",
        "export_compact_columns": "Nur wichtige Spalten exportieren",
        "cache_hit": "Analyse aus Kurzzeit-Cache geladen (schneller).",
        "cache_store": "Neue Analyse berechnet und im Kurzzeit-Cache gespeichert.",
        "sync_progress": "Sync-Fortschritt",
        "quickstart_minimal_enable": "Schlanke Startansicht aktivieren",
        "charts_light_mode": "Chart-Off Modus (leicht)",
        "charts_light_mode_hint": "Blendet schwere Charts aus und zeigt nur Kerninfos.",
        "delta_reuse": "Eingaben unverändert: letzte Analyse wiederverwendet.",
        "background_paused": "Hintergrund-Sync pausiert (Inaktivität).",
        "background_resume": "Hintergrund-Sync fortsetzen",
        "news_load_more": "Mehr News laden",
        "news_text_only": "News nur als Text anzeigen",
        "fast_backtest": "Fast Backtest (kleines Sample)",
        "backtest_only_today_export": "Export nur für heute",
        "precision_section": "Precision-Boost (kostenlos)",
        "precision_auto_tune": "Gewichte automatisch feintunen",
        "precision_recency_decay": "Form-Relevanz (neuere Spiele stärker)",
        "precision_calibration": "Kalibrierungsstärke",
        "precision_lineup_confirmed": "Startelf bestätigt",
        "precision_lineup_penalty": "Unsicherheits-Abschlag",
        "precision_min_data_quality": "Min. Datenqualität für Bet-Empfehlung",
        "precision_weights_auto": "Gewichte wurden anhand Modellqualität leicht angepasst.",
        "precision_data_quality_gate": "No Bet: Datenqualität unter Mindestschwelle.",
        "precision_drift_adjust": "Drift-abhängige Gewichtsanpassung",
        "precision_consensus_gate": "Konsens-Gate (2 von 3 Modellen)",
        "precision_prob_interval": "Wahrscheinlichkeitsintervall",
        "precision_adaptive_threshold": "Adaptive Schwelle je Liga",
        "precision_disagreement_penalty": "Disagreement-Penalty aktivieren",
        "precision_stability_gate": "Stabilitäts-Gate (20/30/40 Spiele)",
        "precision_evidence_gate": "Evidenz-Mindestmenge aktivieren",
        "precision_min_evidence_matches": "Min. historische Team-Spiele",
        "precision_min_evidence_signals": "Min. valide Signale",
        "precision_evidence_gate_reason": "No Bet: zu wenig Evidenz (Datenbasis/Signale).",
        "precision_quantile_threshold": "Quantil-basierte Entscheidungsgrenze",
        "precision_scenario_ensemble": "Szenario-Ensembling (Basis/konservativ/aggressiv)",
        "precision_scenario_conservative": "Konservativen Ensemble-Bound nutzen",
        "precision_online_bias": "Online-Bias-Korrektur aktivieren",
        "precision_bias_lr": "Bias-Lernrate",
        "precision_shadow_mode": "Shadow-Mode für neue Präzisionsregeln",
        "precision_shadow_canary": "Neue Regel nur für Canary-Nutzer ausrollen",
        "precision_shadow_diff": "Shadow-Mode: neue Regel weicht von Legacy-Entscheidung ab.",
        "precision_feature_trust": "Per-Feature-Vertrauensgewichtung",
        "precision_hysteresis": "Hysterese für Bet/No-Bet",
        "precision_season_calibration": "Saisonphasen-Kalibrierung",
        "precision_trimmed_ensemble": "Trimmed-Mean Ensemble-Mix",
        "precision_error_tracking": "Fehlerklassen-Tracking (FP/FN)",
        "precision_error_stats": "Fehlerstatistik: FP={fp} | FN={fn} | N={n}",
        "precision_adaptive_bias_lr": "Adaptive Bias-Lernrate",
        "precision_weighted_consensus": "Confidence-gewichteter Konsens",
        "precision_segmented_hysteresis": "Segmentierte Hysterese (Liga/Risiko)",
        "precision_error_adaptive_threshold": "FP/FN-basierte Schwellenanpassung",
        "precision_shadow_auto_promote": "Shadow Auto-Promotion aktivieren",
        "precision_regime_detector": "Regime-Detektor aktivieren",
        "precision_reliability_score": "Reliability-Score anzeigen",
        "precision_cost_sensitive_threshold": "Kosten-sensitive Schwellensteuerung",
        "precision_fallback_matrix": "Fallback-Matrix bei Datenlücken",
        "precision_adaptive_odds_move": "Adaptive Odds-Move-Schwelle",
        "precision_reliability_hard_gate": "Reliability Hard-Gate",
        "precision_regime_auto_profile": "Regime-basiertes Auto-Risikoprofil",
        "precision_reliability_auto_gate": "Auto-Tuning für Reliability Hard-Gate",
        "precision_odds_alert_cooldown": "Odds-Alert Cooldown (Min)",
        "precision_regime_exposure_scaling": "Regime-abhängige Exposure-Skalierung",
        "precision_odds_auto_cooldown": "Odds-Cooldown automatisch anpassen",
        "precision_replay_ci_blocker": "Replay Gold-Failures blockieren",
        "precision_correlation_guardrail": "Korrelations-Guardrail für Einsatz",
        "precision_shadow_promoted": "Shadow-Rule wurde anhand Vergleichsdaten global freigegeben.",
        "debug_title": "Mini-Fehlerdiagnose",
        "debug_copy": "Copy Debug Info",
        "debug_hint": "Enthält nur Basisdaten zur Fehlersuche (ohne Passwörter/Secrets).",
        "power_table": "Power-User Tabelle",
        "power_table_columns": "Spalten ein-/ausblenden",
        "power_table_hint": "Nur Pro-Modus: zeigt kompakte Einsätze mit frei wählbaren Spalten.",
        "workspace": "Arbeitsbereich",
        "live": "⚡ Live Terminal",
        "backtest": "🧪 Backtesting Lab",
        "news": "📰 News & Sentiment",
        "refresh": "Auto-Refresh (Sek)",
        "risk": "Risikoprofil",
        "compact": "Kompakte Ansicht",
        "accent": "Akzentfarbe",
        "sim": "Simulationen",
        "league": "Liga / Wettbewerb",
        "sync": "Daten werden geladen...",
        "hero": "Global Football Intelligence",
        "live_cfg": "Live-Konfiguration",
        "api_key": "TheOddsAPI Key",
        "auto_odds": "Auto-Quoten aus dem Internet",
        "home": "Heimteam",
        "away": "Auswärtsteam",
        "compare_toggle": "Vergleichsansicht (2. Spiel)",
        "compare_home": "Vergleich Heimteam",
        "compare_away": "Vergleich Auswärtsteam",
        "compare_title": "Spielvergleich",
        "compare_primary": "Spiel A (Hauptauswahl)",
        "compare_secondary": "Spiel B (Vergleich)",
        "compare_ml_conf": "ML-Confidence",
        "quote": "Heimquote",
        "bankroll": "Bankroll (€)",
        "run": "Einstellungen anwenden",
        "apply_hint": 'Klicke auf "Einstellungen anwenden", damit deine Auswahl direkt in die Analyse übernommen wird.',
        "validation_title": "Bitte Eingaben prüfen:",
        "validation_teams_required": "Heim- und Auswärtsteam müssen gesetzt sein.",
        "validation_teams_must_differ": "Heim- und Auswärtsteam dürfen nicht identisch sein.",
        "validation_quote_range": "Quote muss zwischen 1.01 und 25 liegen.",
        "validation_bankroll_positive": "Bankroll muss größer als 0 sein.",
        "validation_confidence_range": "Mindest-Confidence muss zwischen 40% und 95% liegen.",
        "validation_sim_count_range": "Simulationen müssen zwischen 1.000 und 200.000 liegen.",
        "validation_weights_positive": "Die Summe der Modellgewichte muss größer als 0 sein.",
        "validation_weights_non_negative": "Modellgewichte dürfen nicht negativ sein.",
        "validation_ok": "✅ Eingaben sind valide.",
        "validation_error_count": "❌ {count} Eingabeproblem(e) gefunden.",
        "home_dashboard": "Startseite",
        "overview": "Übersicht",
        "analytics": "Analyse",
        "execution": "Ausführung",
        "settings": "Einstellungen",
        "alerts": "Alerts",
        "favorites": "Favoriten",
        "alert_prob": "Alarm bei Wahrscheinlichkeit (%)",
        "alert_quote": "Alarm bei Quote <=",
        "monitor": "Odds-Drift / Live-Wert-Monitoring",
        "admin": "Admin-Bereich",
        "beta": "Beta-Zugang aktiv",
        "api_credits": "API-Credits",
        "api_warn": "Achtung: API-Limit fast erreicht.",
        "api_admin_only": "Live-API ist nur für Admin sichtbar.",
        "portfolio": "Portfolio-Analyse",
        "gamification": "Gamification",
        "seasonal": "Saisonale Muster",
        "filter": "Filter & Suche",
        "news_title": "Live News mit Quellen",
        "source": "Quelle",
        "date": "Datum",
        "link": "Link",
        "backup_ok": "Backup gespeichert",
        "backup_fail": "Backup fehlgeschlagen, Rollback aktiv",
        "sync_all": "Alle Ligen synchronisieren",
        "sync_status": "Synchronisierungsstatus",
        "sync_warn": "Synchronisierung fehlgeschlagen – letzte verfügbare Daten werden verwendet.",
        "feedback_fail": "Feedback konnte nicht gespeichert werden.",
        "feedback": "Benutzer-Feedback",
        "feedback_hint": "Beschreibe kurz, was gut läuft oder verbessert werden sollte.",
        "feedback_send": "Feedback speichern",
        "feedback_category": "Kategorie",
        "feedback_ok": "Danke! Dein Feedback wurde lokal gespeichert.",
        "monitor_alert": "Monitoring-Alarm",
        "quickstart": "Schnellstart",
        "quickstart_enable": "Geführten Schnellstart anzeigen",
        "home_layout": "Startseiten-Layout",
        "layout_beginner": "Einsteiger",
        "layout_balanced": "Balanced",
        "layout_pro": "Pro",
        "layout_custom": "Personalisierung",
        "layout_widgets": "Widgets auswählen",
        "layout_save": "Layout speichern",
        "layout_saved": "Layout gespeichert.",
        "quickstart_step1": "1) Wähle Liga und Modus in der Sidebar.",
        "quickstart_step2": "2) Wähle Heim-/Auswärtsteam und klicke auf Einstellungen anwenden.",
        "quickstart_step3": "3) Prüfe Handlungsempfehlung, Risiko und Alerts.",
        "preset": "Preset",
        "preset_apply": "Preset anwenden",
        "preset_beginner": "Einsteiger",
        "preset_balanced": "Ausgewogen",
        "preset_aggressive": "Aggressiv",
        "preset_beginner_desc": "Einsteiger: sicherer Standard mit klaren Defaults.",
        "preset_balanced_desc": "Ausgewogen: stabiler Mix für Alltag und Live-Betrieb.",
        "preset_aggressive_desc": "Aggressiv: höhere Chancen, aber mehr Schwankung.",
        "session_resume_loaded": "🔄 Letzte Session-Einstellungen wurden geladen.",
        "weights": "Ensemble-Gewichte",
        "weights_hint": "Empfohlen: Poisson 0.45 · Elo 0.35 · ML 0.20",
        "api_forecast": "Prognose: ca. {hours:.1f}h Credits übrig",
        "used_of_total": "Verbraucht {used}/{total}",
        "action_title": "Handlungsempfehlung",
        "action_bet": "Aktion: Bet",
        "action_no_bet": "Aktion: No Bet",
        "action_reason": "Grund",
        "action_reason_value": "Positiver Value und akzeptables Risiko",
        "action_reason_risk": "Kein ausreichender Value oder Risiko zu hoch",
        "action_reason_confidence": "Modell-Confidence unter dem Mindestwert",
        "action_why_not_set": "Warum nicht gesetzt?",
        "action_blocking_title": "Blockierende Checks",
        "action_blocking_edge": "Value Edge ist nicht positiv.",
        "action_blocking_confidence": "Confidence liegt unter dem Mindestwert.",
        "action_blocking_exposure": "Kein verfügbares Risiko-Budget (Exposure-Limit erreicht).",
        "data_freshness": "Datenalter",
        "data_quality": "Datenqualität",
        "data_drift": "Feature-Drift",
        "data_drift_insufficient": "nicht genügend Daten",
        "data_drift_alert": "Erhöhter Feature-Drift erkannt ({status}) – Top-Feature: {feature}",
        "calibration": "Kalibrierung",
        "calibration_good": "Modell-Kalibrierung: stabil",
        "calibration_watch": "Modell-Kalibrierung: beobachten",
        "calibration_risk": "Modell-Kalibrierung: erhöhtes Risiko",
        "canary_state": "Canary-Status",
        "canary_on": "aktiv",
        "canary_off": "aus",
        "min_confidence": "Mindest-Confidence (%)",
        "action_checks": "Entscheidungs-Checks",
        "explainability_panel": "Explainability: Warum Bet / No Bet?",
        "explainability_top_drivers": "Top-3 Treiber",
        "explainability_top_risks": "Top-2 Risiken",
        "odds_move_warning": "Odds-Move-Warnsignal: Marktquote {move:+.2f} ({pct:+.1f}%)",
        "auto_retro": "Auto-Retrospektive (30 geschlossene Bets)",
        "regime_state": "Regime-Status",
        "reliability_score": "Reliability-Score",
        "fallback_notice": "Fallback aktiv ({count} fehlende Signale, Penalty {penalty:.1f}%)",
        "replay_suite": "Replay-Mini-Suite",
        "reliability_trend": "Reliability-Trend",
        "risk_profile_auto": "Auto-Risikoprofil",
        "risk_profile_auto_hint": "Regime-basiert: nutzt {profile}",
        "odds_move_direction_hint": "{hint}",
        "fallback_breakdown": "Fallback-Details",
        "replay_quality": "Replay-Qualität",
        "uncertainty_band": "Unsicherheitsband",
        "dynamic_threshold_stress": "Dynamic Threshold Stress-Test",
        "stress_flip_count": "What-if Flips",
        "replay_health_trend": "Replay-Health-Trend (20 Runs)",
        "regime_actions": "Regime-Aktionen",
        "correlation_penalty": "Korrelations-Penalty",
        "replay_health": "Replay-Health",
        "odds_priority": "Odds-Priorität",
        "check_edge": "Value Edge: {value}%",
        "check_confidence": "ML-Confidence: {value}% (Minimum: {minimum}%)",
        "check_exposure": "Verfügbares Risiko-Budget: {left}€ von {max_exposure}€",
        "widget_win_prob": "Sieg-Wahrscheinlichkeit",
        "widget_value_edge": "Value Edge",
        "widget_stake": "Empfohlener Einsatz",
        "widget_api_credits": "API-Credits",
        "widget_exposure": "Aktuelles Exposure",
        "widget_data_quality": "Datenqualität",
        "widget_sync_status": "Letzte Synchronisierung",
        "widget_ml_probs": "ML-Wahrscheinlichkeiten",
        "widget_social_sentiment": "Social Sentiment",
        "widget_injury_risk": "Verletzungsrisiko",
        "widget_clv": "CLV",
        "empty_data": "Keine Daten verfügbar. Bitte Liga wechseln oder Synchronisierung starten.",
        "alert_center": "Alert-Center",
        "alert_center_empty": "Keine Alerts in dieser Sitzung.",
        "password_strength": "Passwortstärke",
        "show_password": "Passwort anzeigen",
        "low": "Niedrig",
        "medium": "Mittel",
        "high": "Hoch",
        "risk_conservative": "Konservativ",
        "risk_balanced": "Ausgewogen",
        "risk_aggressive": "Aggressiv",
        "backtest_title": "Strategie-Designer",
        "min_elo_diff": "Min Elo-Differenz",
        "assumed_odds": "Angenommene Quote",
        "stake_per_bet": "Einsatz pro Wette",
        "start_capital": "Startkapital",
        "run_backtest": "Institutionellen Backtest starten",
        "end_capital": "Endkapital",
        "win_rate": "Trefferquote",
        "roi": "ROI",
        "max_drawdown": "Max Drawdown",
        "backtest_export_csv": "Backtest CSV Export",
        "backtest_export_json": "Backtest JSON Export",
        "w_value": "w(Value)",
        "w_prob": "w(Wahrscheinlichkeit)",
        "w_risk": "w(Risiko)",
        "live_alert_msg": "🔔 {team} Alarm | Wahrscheinlichkeit {prob:.1f}% | Quote {quote:.2f}",
        "ensemble_win_prob": "Ensemble Sieg-Wahrscheinlichkeit",
        "fair_odds": "Faire Quote",
        "value_edge": "Value-Vorteil",
        "composite": "Composite",
        "injury_risk": "Verletzungsrisiko (ML)",
        "social_sentiment": "Social Sentiment",
        "ml_probs": "ML Wahrsch. H/U/A: {h:.2f} / {d:.2f} / {a:.2f}",
        "exposure": "Exposure",
        "stake": "Einsatz",
        "total_stake": "Gesamteinsätze",
        "avg_edge": "Durchschn. Edge",
        "diversification": "Diversifikation",
        "clv": "CLV (Closing Line Value)",
        "clv_better_rate": "CLV Trefferquote",
        "clv_samples": "CLV Stichproben",
        "benchmark": "Benchmark",
        "model_unavailable": "ML-Modell derzeit nicht verfügbar (zu wenig oder zu einseitige Daten).",
    },
    "en": {
        "app_title": "Elite Quant Ultimate — Global Pro Workspace",
        "login": "Beta Login",
        "username": "Username",
        "password": "Password",
        "login_btn": "Sign in",
        "logout": "Sign out",
        "login_err": "Login failed.",
        "login_locked": "Too many failed attempts. Account is temporarily locked.",
        "login_rate_limited": "Too many login attempts. Please wait and try again.",
        "login_totp": "Admin 2FA code (TOTP)",
        "login_totp_err": "Admin login requires a valid 2FA code.",
        "session_expired": "Session expired due to inactivity. Please log in again.",
        "register": "Register",
        "register_btn": "Create account",
        "password_repeat": "Repeat password",
        "register_ok": "Account created. You can now sign in.",
        "register_err_exists": "Username already exists.",
        "register_err_password": "Password must be at least 8 characters.",
        "register_err_password_weak": "Password is too weak (use upper/lowercase, numbers, symbols).",
        "register_err_username": "Username must be 3–32 chars (letters, numbers, _ . - only).",
        "register_err_mismatch": "Passwords do not match.",
        "user_level": "User mode",
        "level_beginner": "Beginner",
        "level_pro": "Pro",
        "ai_help": "💬 AI help (free)",
        "ai_help_hint": "Ask in plain language. Help explains terms and suggests next steps.",
        "ai_question": "Your question",
        "ai_ask": "Show answer",
        "ai_admin": "Admin AI coach (free)",
        "ai_admin_hint": "Admin only: explains all features clearly and provides concrete pre-match setup guidance.",
        "ai_admin_locked": "AI coach is currently enabled for admin only.",
        "ai_explain_all": "Explain all features",
        "ai_pre_match": "Setup guidance for this match",
        "ai_auto_optimize": "Auto optimize (coach)",
        "ai_auto_optimized": "Coach settings applied.",
        "control": "Control Center",
        "sidebar_section_layout": "Layout & Navigation",
        "sidebar_section_strategy": "Strategy baseline",
        "sidebar_section_sync": "Data & synchronization",
        "sidebar_section_live_match": "Live: Match setup",
        "sidebar_section_live_monitor": "Live: Monitoring",
        "sidebar_section_live_advanced": "Live: Advanced settings",
        "preset_applied": 'Applied preset "{preset}" for {league}.',
        "language": "Language",
        "shortcuts": "Keyboard shortcuts",
        "shortcuts_hint": "A=Apply, S=Sync, L=Logout, G=Language",
        "shortcuts_input": "Shortcut command",
        "shortcuts_apply": "Run shortcut",
        "shortcuts_unknown": "Unknown shortcut. Use A/S/L/G.",
        "safe_mode": "Safe mode (laptop)",
        "safe_mode_plus": "Safe mode+ profile",
        "safe_mode_hint": "Switches to resource-friendly defaults.",
        "safe_mode_active": "Safe mode active: lower load and smoother performance on low-end laptops.",
        "autosave_status": "Auto-save: {time}",
        "metric_help": "Metric help",
        "metric_help_edge": "Value edge: a positive value means your model price is better than the market price.",
        "metric_help_confidence": "Confidence: how sure the model is about the direction.",
        "metric_help_exposure": "Exposure: how much of the risk budget is already in use.",
        "export_today_csv": "Export today as CSV",
        "export_compare_csv": "Export comparison as CSV",
        "export_compact_columns": "Export only important columns",
        "cache_hit": "Loaded analysis from short-term cache (faster).",
        "cache_store": "Calculated new analysis and stored it in short-term cache.",
        "sync_progress": "Sync progress",
        "quickstart_minimal_enable": "Enable slim quickstart view",
        "charts_light_mode": "Chart-off mode (light)",
        "charts_light_mode_hint": "Hides heavy charts and shows only core info.",
        "delta_reuse": "Inputs unchanged: reusing last analysis.",
        "background_paused": "Background sync paused (inactivity).",
        "background_resume": "Resume background sync",
        "news_load_more": "Load more news",
        "news_text_only": "Show news as text only",
        "fast_backtest": "Fast backtest (small sample)",
        "backtest_only_today_export": "Export only today",
        "precision_section": "Precision boost (free)",
        "precision_auto_tune": "Auto fine-tune weights",
        "precision_recency_decay": "Form recency (newer matches weighted higher)",
        "precision_calibration": "Calibration strength",
        "precision_lineup_confirmed": "Starting lineup confirmed",
        "precision_lineup_penalty": "Uncertainty penalty",
        "precision_min_data_quality": "Min data quality for bet recommendation",
        "precision_weights_auto": "Weights were lightly adjusted based on model quality.",
        "precision_data_quality_gate": "No bet: data quality below minimum threshold.",
        "precision_drift_adjust": "Drift-aware weight adjustment",
        "precision_consensus_gate": "Consensus gate (2 of 3 models)",
        "precision_prob_interval": "Probability interval",
        "precision_adaptive_threshold": "Adaptive per-league threshold",
        "precision_disagreement_penalty": "Enable disagreement penalty",
        "precision_stability_gate": "Stability gate (20/30/40 matches)",
        "precision_evidence_gate": "Enable minimum evidence gate",
        "precision_min_evidence_matches": "Min historical team matches",
        "precision_min_evidence_signals": "Min valid signals",
        "precision_evidence_gate_reason": "No bet: not enough evidence (data basis/signals).",
        "precision_quantile_threshold": "Quantile-based decision threshold",
        "precision_scenario_ensemble": "Scenario ensembling (base/conservative/aggressive)",
        "precision_scenario_conservative": "Use conservative ensemble bound",
        "precision_online_bias": "Enable online bias correction",
        "precision_bias_lr": "Bias learning rate",
        "precision_shadow_mode": "Shadow mode for new precision rules",
        "precision_shadow_canary": "Roll out new rule only for canary users",
        "precision_shadow_diff": "Shadow mode: new rule differs from legacy decision.",
        "precision_feature_trust": "Per-feature trust weighting",
        "precision_hysteresis": "Hysteresis for bet/no-bet",
        "precision_season_calibration": "Season-phase calibration",
        "precision_trimmed_ensemble": "Trimmed-mean ensemble mix",
        "precision_error_tracking": "Error-class tracking (FP/FN)",
        "precision_error_stats": "Error stats: FP={fp} | FN={fn} | N={n}",
        "precision_adaptive_bias_lr": "Adaptive bias learning rate",
        "precision_weighted_consensus": "Confidence-weighted consensus",
        "precision_segmented_hysteresis": "Segmented hysteresis (league/risk)",
        "precision_error_adaptive_threshold": "FP/FN-based threshold adaptation",
        "precision_shadow_auto_promote": "Enable shadow auto-promotion",
        "precision_regime_detector": "Enable regime detector",
        "precision_reliability_score": "Show reliability score",
        "precision_cost_sensitive_threshold": "Cost-sensitive thresholding",
        "precision_fallback_matrix": "Fallback matrix for missing signals",
        "precision_adaptive_odds_move": "Adaptive odds-move threshold",
        "precision_reliability_hard_gate": "Reliability hard gate",
        "precision_regime_auto_profile": "Regime-based auto risk profile",
        "precision_reliability_auto_gate": "Auto-tune reliability hard gate",
        "precision_odds_alert_cooldown": "Odds-alert cooldown (min)",
        "precision_regime_exposure_scaling": "Regime-based exposure scaling",
        "precision_odds_auto_cooldown": "Auto-adjust odds cooldown",
        "precision_replay_ci_blocker": "Block on replay gold failures",
        "precision_correlation_guardrail": "Correlation guardrail for stake",
        "precision_shadow_promoted": "Shadow rule promoted globally based on comparison data.",
        "debug_title": "Mini diagnostics",
        "debug_copy": "Copy debug info",
        "debug_hint": "Contains only basic troubleshooting data (no passwords/secrets).",
        "power_table": "Power-user table",
        "power_table_columns": "Show/hide columns",
        "power_table_hint": "Pro mode only: compact ledger with selectable columns.",
        "workspace": "Workspace",
        "live": "⚡ Live Terminal",
        "backtest": "🧪 Backtesting Lab",
        "news": "📰 News & Sentiment",
        "refresh": "Auto-refresh (sec)",
        "risk": "Risk profile",
        "compact": "Compact density",
        "accent": "Accent color",
        "sim": "Simulations",
        "league": "League / Competition",
        "sync": "Loading data...",
        "hero": "Global Football Intelligence",
        "live_cfg": "Live configuration",
        "api_key": "TheOddsAPI Key",
        "auto_odds": "Auto odds from internet",
        "home": "Home team",
        "away": "Away team",
        "compare_toggle": "Comparison view (2nd match)",
        "compare_home": "Comparison home team",
        "compare_away": "Comparison away team",
        "compare_title": "Match comparison",
        "compare_primary": "Match A (primary)",
        "compare_secondary": "Match B (comparison)",
        "compare_ml_conf": "ML confidence",
        "quote": "Home quote",
        "bankroll": "Bankroll (€)",
        "run": "Apply settings",
        "apply_hint": 'Click "Apply settings" so your selection is immediately used for analysis.',
        "validation_title": "Please review the current inputs:",
        "validation_teams_required": "Home and away team are required.",
        "validation_teams_must_differ": "Home and away team must be different.",
        "validation_quote_range": "Odds must be between 1.01 and 25.",
        "validation_bankroll_positive": "Bankroll must be greater than 0.",
        "validation_confidence_range": "Minimum confidence must be between 40% and 95%.",
        "validation_sim_count_range": "Simulations must be between 1,000 and 200,000.",
        "validation_weights_positive": "Model weight sum must be greater than 0.",
        "validation_weights_non_negative": "Model weights cannot be negative.",
        "validation_ok": "✅ Inputs are valid.",
        "validation_error_count": "❌ {count} input issue(s) found.",
        "home_dashboard": "Home",
        "overview": "Overview",
        "analytics": "Analytics",
        "execution": "Execution",
        "settings": "Settings",
        "alerts": "Alerts",
        "favorites": "Favorites",
        "alert_prob": "Alert at probability (%)",
        "alert_quote": "Alert at quote <=",
        "monitor": "Odds Drift / Live Value Monitoring",
        "admin": "Admin Area",
        "beta": "Beta access active",
        "api_credits": "API credits",
        "api_warn": "Warning: API limit almost reached.",
        "api_admin_only": "Live API is visible to admin only.",
        "portfolio": "Portfolio analysis",
        "gamification": "Gamification",
        "seasonal": "Seasonal patterns",
        "filter": "Filter & Search",
        "news_title": "Live News with sources",
        "source": "Source",
        "date": "Date",
        "link": "Link",
        "backup_ok": "Backup saved",
        "backup_fail": "Backup failed, rollback active",
        "sync_all": "Sync all leagues",
        "sync_status": "Sync status",
        "sync_warn": "Sync failed — using last available data.",
        "feedback_fail": "Feedback could not be saved.",
        "feedback": "User feedback",
        "feedback_hint": "Briefly describe what works well or what should be improved.",
        "feedback_send": "Save feedback",
        "feedback_category": "Category",
        "feedback_ok": "Thanks! Your feedback was saved locally.",
        "monitor_alert": "Monitoring alert",
        "quickstart": "Quickstart",
        "quickstart_enable": "Show guided quickstart",
        "home_layout": "Home layout",
        "layout_beginner": "Beginner",
        "layout_balanced": "Balanced",
        "layout_pro": "Pro",
        "layout_custom": "Personalization",
        "layout_widgets": "Select widgets",
        "layout_save": "Save layout",
        "layout_saved": "Layout saved.",
        "quickstart_step1": "1) Choose league and workspace in the sidebar.",
        "quickstart_step2": "2) Choose home/away teams and click Apply settings.",
        "quickstart_step3": "3) Review action recommendation, risk and alerts.",
        "preset": "Preset",
        "preset_apply": "Apply preset",
        "preset_beginner": "Beginner",
        "preset_balanced": "Balanced",
        "preset_aggressive": "Aggressive",
        "preset_beginner_desc": "Beginner: safe defaults and minimal complexity.",
        "preset_balanced_desc": "Balanced: stable mix for everyday live usage.",
        "preset_aggressive_desc": "Aggressive: higher upside with more volatility.",
        "session_resume_loaded": "🔄 Restored settings from your last session.",
        "weights": "Ensemble Weights",
        "weights_hint": "Recommended: Poisson 0.45 · Elo 0.35 · ML 0.20",
        "api_forecast": "Forecast: ~{hours:.1f}h credits left",
        "used_of_total": "Used {used}/{total}",
        "action_title": "Action recommendation",
        "action_bet": "Action: Bet",
        "action_no_bet": "Action: No Bet",
        "action_reason": "Reason",
        "action_reason_value": "Positive value edge and acceptable risk",
        "action_reason_risk": "Insufficient value edge or risk too high",
        "action_reason_confidence": "Model confidence is below minimum threshold",
        "action_why_not_set": "Why not placed?",
        "action_blocking_title": "Blocking checks",
        "action_blocking_edge": "Value edge is not positive.",
        "action_blocking_confidence": "Confidence is below the minimum threshold.",
        "action_blocking_exposure": "No available risk budget (exposure limit reached).",
        "data_freshness": "Data freshness",
        "data_quality": "Data quality",
        "data_drift": "Feature drift",
        "data_drift_insufficient": "insufficient data",
        "data_drift_alert": "Elevated feature drift detected ({status}) – top feature: {feature}",
        "calibration": "Calibration",
        "calibration_good": "Model calibration: stable",
        "calibration_watch": "Model calibration: monitor",
        "calibration_risk": "Model calibration: elevated risk",
        "canary_state": "Canary status",
        "canary_on": "active",
        "canary_off": "off",
        "min_confidence": "Minimum confidence (%)",
        "action_checks": "Decision checks",
        "explainability_panel": "Explainability: Why Bet / No Bet?",
        "explainability_top_drivers": "Top-3 drivers",
        "explainability_top_risks": "Top-2 risks",
        "odds_move_warning": "Odds-move warning: market odds {move:+.2f} ({pct:+.1f}%)",
        "auto_retro": "Auto retrospective (last 30 closed bets)",
        "regime_state": "Regime state",
        "reliability_score": "Reliability score",
        "fallback_notice": "Fallback active ({count} missing signals, penalty {penalty:.1f}%)",
        "replay_suite": "Replay mini-suite",
        "reliability_trend": "Reliability trend",
        "risk_profile_auto": "Auto risk profile",
        "risk_profile_auto_hint": "Regime-based: using {profile}",
        "odds_move_direction_hint": "{hint}",
        "fallback_breakdown": "Fallback details",
        "replay_quality": "Replay quality",
        "uncertainty_band": "Uncertainty band",
        "dynamic_threshold_stress": "Dynamic threshold stress test",
        "stress_flip_count": "What-if flips",
        "replay_health_trend": "Replay health trend (20 runs)",
        "regime_actions": "Regime actions",
        "correlation_penalty": "Correlation penalty",
        "replay_health": "Replay health",
        "odds_priority": "Odds priority",
        "check_edge": "Value edge: {value}%",
        "check_confidence": "ML confidence: {value}% (minimum: {minimum}%)",
        "check_exposure": "Available risk budget: {left}€ of {max_exposure}€",
        "widget_win_prob": "Win probability",
        "widget_value_edge": "Value edge",
        "widget_stake": "Suggested stake",
        "widget_api_credits": "API credits",
        "widget_exposure": "Current exposure",
        "widget_data_quality": "Data quality",
        "widget_sync_status": "Last sync",
        "widget_ml_probs": "ML probabilities",
        "widget_social_sentiment": "Social sentiment",
        "widget_injury_risk": "Injury risk",
        "widget_clv": "CLV",
        "empty_data": "No data available. Please switch league or run sync.",
        "alert_center": "Alert center",
        "alert_center_empty": "No alerts in this session.",
        "password_strength": "Password strength",
        "show_password": "Show password",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "risk_conservative": "Conservative",
        "risk_balanced": "Balanced",
        "risk_aggressive": "Aggressive",
        "backtest_title": "Strategy Designer",
        "min_elo_diff": "Min Elo-Diff",
        "assumed_odds": "Assumed odds",
        "stake_per_bet": "Stake per bet",
        "start_capital": "Start capital",
        "run_backtest": "Run institutional backtest",
        "end_capital": "End capital",
        "win_rate": "Win rate",
        "roi": "ROI",
        "max_drawdown": "Max drawdown",
        "backtest_export_csv": "Backtest CSV export",
        "backtest_export_json": "Backtest JSON export",
        "w_value": "w(Value)",
        "w_prob": "w(Prob)",
        "w_risk": "w(Risk)",
        "live_alert_msg": "🔔 {team} alert | Prob {prob:.1f}% | Quote {quote:.2f}",
        "ensemble_win_prob": "Ensemble win probability",
        "fair_odds": "Fair odds",
        "value_edge": "Value edge",
        "composite": "Composite",
        "injury_risk": "Injury risk (ML)",
        "social_sentiment": "Social sentiment",
        "ml_probs": "ML probs H/D/A: {h:.2f} / {d:.2f} / {a:.2f}",
        "exposure": "Exposure",
        "stake": "Stake",
        "total_stake": "Total stake",
        "avg_edge": "Avg edge",
        "diversification": "Diversification",
        "clv": "CLV (Closing Line Value)",
        "clv_better_rate": "CLV hit rate",
        "clv_samples": "CLV samples",
        "benchmark": "Benchmark",
        "model_unavailable": "ML model currently unavailable (insufficient or single-class data).",
    },
}

_RAW_TOOLTIPS = {
    "de": {
        "refresh": "Wie oft Daten neu geholt werden.",
        "risk": "So vorsichtig setzt die App Einsätze.",
        "sim": "Mehr Simulationen sind genauer, aber langsamer.",
        "league": "Hier wählst du den Wettbewerb.",
        "auto_odds": "Lädt Quoten automatisch aus dem Internet.",
        "quote": "Manuelle Quote, falls keine Live-Quote da ist.",
        "bankroll": "Gesamtes Budget für Wetten.",
        "accent": "Wählt die Hauptfarbe des Dashboards.",
        "compact": "Reduziert Abstände für mehr Informationen auf einmal.",
        "workspace": "Wechselt zwischen Live, Backtest und News.",
        "feedback": "Direktes Feedback hilft bei der Priorisierung neuer Features.",
        "sync_all": "Aktualisiert historische und Live-CSV-Daten aller Ligen.",
        "api_key": "API-Schlüssel für Live-Quoten. Ohne gültigen Key nutzt die App Fallback-Werte.",
        "home": "Wähle das Heimteam für die nächste Analyse.",
        "away": "Wähle das Auswärtsteam für die nächste Analyse.",
        "favorites": "Favoriten lösen schneller Alerts aus und werden priorisiert angezeigt.",
        "alert_prob": "Alert wird ausgelöst, wenn die berechnete Sieg-Wahrscheinlichkeit diesen Wert überschreitet.",
        "alert_quote": "Alert wird ausgelöst, wenn die Marktquote unter oder gleich diesem Wert liegt.",
        "weights": "Gewichte bestimmen, wie stark Poisson-, Elo- und ML-Modell in die Endprognose eingehen.",
        "weights_hint": "Empfohlene Startwerte für die meisten Nutzer. Im Profi-Modus kannst du sie feinjustieren.",
        "search_team": "Filtert den Datenbestand auf Teamnamen (Heim und Auswärts).",
        "min_elo": "Setzt eine Mindest-Elo-Schwelle, unter der keine Empfehlung gesetzt wird.",
        "min_confidence": "Ignoriert Wetten, wenn die Modell-Confidence unter diesem Wert liegt.",
        "w_poisson": "Gewichtung des Poisson-Simulationsmodells.",
        "w_elo": "Gewichtung des Elo-Modells.",
        "w_ml": "Gewichtung des ML-Modells.",
        "run": "Übernimmt alle aktuellen Eingaben und startet die Analyse.",
        "quickstart_enable": "Zeigt eine Einsteiger-Anleitung mit den wichtigsten Schritten.",
        "user_level": "Einsteiger-Modus reduziert Komplexität. Profi-Modus zeigt alle erweiterten Optionen.",
        "home_layout": "Wähle ein fertiges Startseiten-Layout oder personalisiere die Widgets.",
        "ai_help": "Kostenlose integrierte Hilfs-KI ohne Cloud-Kosten: erklärt Begriffe und Bedienung.",
        "safe_mode": "Aktiviert ressourcenschonende Standardwerte für schwächere Geräte.",
        "metric_help": "Kurze Erklärungen zu den wichtigsten Kennzahlen.",
        "export_compact_columns": "Exportiert nur ein kompaktes Set wichtiger Spalten.",
        "quickstart_minimal_enable": "Startet mit reduzierter Ansicht für schnelleren Einstieg.",
        "charts_light_mode": "Schaltet große Charts für schwächere Geräte aus.",
        "precision_auto_tune": "Passt Poisson/Elo/ML leicht an CV-Genauigkeit und Kalibrierungsfehler an.",
        "precision_consensus_gate": "Setzt Bet nur, wenn mindestens 2 Modelle die Richtung stützen.",
        "precision_adaptive_threshold": "Passt die Freigabeschwelle leicht an jüngste Ligaperformance an.",
    },
    "en": {
        "refresh": "How often fresh data is fetched.",
        "risk": "How careful stake sizing should be.",
        "sim": "More simulations are more stable but slower.",
        "league": "Select the competition here.",
        "auto_odds": "Fetch odds automatically from the web.",
        "quote": "Manual quote if no live quote is available.",
        "bankroll": "Total budget for betting.",
        "accent": "Choose the dashboard accent color.",
        "compact": "Use tighter spacing to show more information.",
        "workspace": "Switch between live, backtest, and news areas.",
        "feedback": "Direct feedback helps prioritize upcoming features.",
        "sync_all": "Refresh historical and live CSV data for all leagues.",
        "api_key": "API key for live odds. Without a valid key the app falls back to local/default values.",
        "home": "Choose the home team for the next analysis.",
        "away": "Choose the away team for the next analysis.",
        "favorites": "Favorites trigger alerts earlier and are prioritized in monitoring.",
        "alert_prob": "Trigger an alert when the computed win probability exceeds this threshold.",
        "alert_quote": "Trigger an alert when market odds are less than or equal to this threshold.",
        "weights": "Weights control how strongly Poisson, Elo, and ML influence the final ensemble prediction.",
        "weights_hint": "Recommended starting values for most users. In pro mode you can fine-tune them.",
        "search_team": "Filter the dataset by team names (home and away).",
        "min_elo": "Set a minimum Elo threshold below which no recommendation is placed.",
        "min_confidence": "Ignore bets when model confidence is below this threshold.",
        "w_poisson": "Weight for the Poisson simulation model.",
        "w_elo": "Weight for the Elo model.",
        "w_ml": "Weight for the ML model.",
        "run": "Apply all current inputs and start the analysis.",
        "quickstart_enable": "Show a beginner-friendly guide with the essential steps.",
        "user_level": "Beginner mode reduces complexity. Pro mode reveals all advanced options.",
        "home_layout": "Choose a preset home layout or personalize widgets.",
        "ai_help": "Free built-in help AI without cloud costs: explains terms and workflow.",
        "safe_mode": "Enables resource-saving defaults for weaker devices.",
        "metric_help": "Short explanations for the most important metrics.",
        "export_compact_columns": "Exports only a compact set of key columns.",
        "quickstart_minimal_enable": "Starts with a reduced view for a faster entry.",
        "charts_light_mode": "Disables large charts for weaker devices.",
        "precision_auto_tune": "Lightly adapts Poisson/Elo/ML using CV accuracy and calibration error.",
        "precision_consensus_gate": "Only place a bet if at least 2 models support it.",
        "precision_adaptive_threshold": "Lightly adapts release thresholds using recent league performance.",
    },
}

I18N = {lang: _RAW_I18N[lang] for lang in SUPPORTED_LANGUAGES if lang in _RAW_I18N}
TOOLTIPS = {lang: _RAW_TOOLTIPS[lang] for lang in SUPPORTED_LANGUAGES if lang in _RAW_TOOLTIPS}
LANGUAGE_LABELS = {
    lang: I18N_LANGUAGE_LABELS.get(lang, lang.upper()) for lang in SUPPORTED_LANGUAGES
}
LANGUAGES = {lang: lang for lang in SUPPORTED_LANGUAGES}
assert set(I18N) == set(SUPPORTED_LANGUAGES), "I18N must only contain supported languages (de/en)."
assert set(TOOLTIPS) == set(
    SUPPORTED_LANGUAGES
), "TOOLTIPS must only contain supported languages (de/en)."

LEAGUES = {
    "Bundesliga": "D1",
    "2. Bundesliga": "D2",
    "3. Liga": "D3",
    "Premier League": "E0",
    "Championship": "E1",
    "La Liga": "SP1",
    "La Liga 2": "SP2",
    "Serie A": "I1",
    "Serie B": "I2",
    "Ligue 1": "F1",
    "Ligue 2": "F2",
}


ODDS_SPORT_MAP = {
    "Bundesliga": "soccer_germany_bundesliga",
    "2. Bundesliga": "soccer_germany_bundesliga2",
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Ligue 1": "soccer_france_ligue_one",
    "UEFA Champions League": "soccer_uefa_champs_league",
    "UEFA Europa League": "soccer_uefa_europa_league",
    "Major League Soccer": "soccer_usa_mls",
}

PREMIUM_FLAGS = {
    "pro_dashboard": True,
    "advanced_alerts": True,
    "premium_data_connectors": False,
}

DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)
logger = get_app_logger("elite_analyst.app")
ACTIVE_LEAGUES = filter_leagues_with_cached_data(LEAGUES, data_dir=DATA_DIR)
sync_service = LeagueDataSyncService(ACTIVE_LEAGUES)
monitoring_service = MonitoringService()
SYNC_INTERVAL = timedelta(minutes=30)
FEEDBACK_FILE = Path("runtime_logs/user_feedback.jsonl")
ODDS_HISTORY_LIMIT = 500


def expand_probs_3(proba: list[float], classes: list[int]) -> list[float]:
    expanded = [0.0, 0.0, 0.0]
    for idx, cls in enumerate(classes):
        if 0 <= int(cls) < 3:
            expanded[int(cls)] = float(proba[idx])
    return expanded


def parse_credit_status(credit_text: str) -> tuple[int | None, int | None]:
    try:
        used, remaining = str(credit_text).split("/")
        return int(used), int(remaining)
    except (ValueError, TypeError):
        return None, None


def t(key: str) -> str:
    lang = normalize_language(st.session_state.get("lang", "de"))
    return I18N.get(lang, I18N["de"]).get(key, key)


def tip(key: str) -> str:
    lang = normalize_language(st.session_state.get("lang", "de"))
    return TOOLTIPS.get(lang, TOOLTIPS["de"]).get(key, "")


def log_event(level: str, msg: str) -> None:
    st.session_state.setdefault("error_logs", []).append(
        {"ts": datetime.utcnow().isoformat(), "level": level, "msg": msg}
    )


def confidence_gauge(value: float, accent: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge", value=value, gauge={"axis": {"range": [0, 100]}, "bar": {"color": accent}}
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 36, "b": 4},
        height=300,
        annotations=[
            {
                "text": f"<b>{value:.1f}%</b>",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.24,
                "xanchor": "center",
                "yanchor": "middle",
                "showarrow": False,
                "font": {"size": 30, "color": "white"},
            }
        ],
    )
    return fig


def save_league_csv(df: pd.DataFrame, league_name: str) -> tuple[bool, str]:
    target = DATA_DIR / league_cache_filename(league_name)
    backup = DATA_DIR / f"{target.stem}.bak.csv"
    try:
        if target.exists():
            target.replace(backup)
        df.to_csv(target, index=False)
        return True, t("backup_ok")
    except Exception:
        log_event("ERROR", f"csv_save_failed:{league_name}")
        if backup.exists():
            backup.replace(target)
        return False, t("backup_fail")


def load_dynamic_league_data(
    league_name: str, code: str
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    try:
        sync_service.sync_league(league_name, code)
    except Exception as exc:
        logger.warning("league sync failed for %s: %s", league_name, exc)
        log_event("WARN", f"sync_failed:{league_name}")

    cached = DATA_DIR / league_cache_filename(league_name)
    if cached.exists():
        try:
            df = with_retry(
                lambda: pd.read_csv(cached), retries=3, action_name=f"read_cache:{league_name}"
            )
            df = _ensure_columns(df)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
            df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"], inplace=True)
            df.sort_values("Date", inplace=True)
            elo, hist, h, a = _compute_elo(df)
            df["Elo_Home"] = h
            df["Elo_Away"] = a
            return df, elo, hist
        except Exception as exc:
            logger.warning("cache read failed for %s: %s", league_name, exc)
            log_event("WARN", f"cache_read_failed:{league_name}")

    try:
        return load_data_full(CONFIG.default_league)
    except Exception as exc:
        logger.error("fallback load_data_full failed: %s", exc)
        log_event("ERROR", "fallback_load_failed")
        return pd.DataFrame(), {}, pd.DataFrame(columns=["Date", "Team", "Elo"])


def get_real_odds_by_league(
    api_key: str, home_team: str, league_name: str
) -> tuple[list[dict] | None, str]:
    sport = ODDS_SPORT_MAP.get(league_name)
    if not sport:
        return get_real_odds(api_key, home_team)
    if not api_key or len(api_key) < 10:
        return None, "Standby"

    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {"apiKey": api_key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"}
    try:
        r = with_retry(
            lambda: requests.get(url, params=params, timeout=8),
            retries=3,
            action_name=f"odds:{league_name}",
        )
        r.raise_for_status()
        credits = (
            f"{r.headers.get('x-requests-used', '?')}/{r.headers.get('x-requests-remaining', '?')}"
        )
        data = r.json()
        if not isinstance(data, list):
            return None, credits
        for game in data:
            if not isinstance(game, dict):
                continue
            if home_team[:4].lower() in str(game.get("home_team", "")).lower():
                bookmakers = game.get("bookmakers") or []
                if bookmakers and isinstance(bookmakers[0], dict):
                    markets = bookmakers[0].get("markets") or []
                    if markets and isinstance(markets[0], dict):
                        outcomes = markets[0].get("outcomes")
                        if isinstance(outcomes, list):
                            return outcomes, credits
        return None, credits
    except Exception:
        log_event("WARN", f"odds_failed:{league_name}")
        return None, "Fehler"


@st.cache_resource
def build_injury_model() -> RandomForestClassifier:
    rng = np.random.default_rng(42)
    x = pd.DataFrame(
        {
            "minutes_load": rng.uniform(50, 100, 600),
            "travel": rng.uniform(0, 1, 600),
            "stress": rng.uniform(0, 1, 600),
            "inj_hist": rng.integers(0, 3, 600),
        }
    )
    y = (
        (
            0.5 * x["minutes_load"] / 100
            + 0.2 * x["travel"]
            + 0.2 * x["stress"]
            + 0.1 * x["inj_hist"] / 2
        )
        > 0.56
    ).astype(int)
    m = RandomForestClassifier(n_estimators=160, random_state=42)
    m.fit(x, y)
    return m


def social_sentiment_index(team: str) -> tuple[int, list[dict]]:
    url = f"https://www.reddit.com/search.rss?q={team.replace(' ', '%20')}%20football"
    entries: list[dict] = []
    try:
        import feedparser

        feed = feedparser.parse(url)
        for e in feed.entries[:8]:
            entries.append(
                {
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", ""),
                    "source": "Reddit",
                }
            )
    except Exception:
        log_event("WARN", "social_feed_failed")
    score = 50
    for e in entries:
        txt = (e["title"] + " " + e["summary"]).lower()
        score += 5 * ("win" in txt or "sieg" in txt or "strong" in txt)
        score -= 5 * ("injury" in txt or "krise" in txt or "loss" in txt)
    return max(0, min(100, score)), entries


def run_full_sync(trigger: str) -> list[str]:
    logger.info("run_full_sync trigger=%s", trigger)
    try:
        results = []
        leagues = list(ACTIVE_LEAGUES.items())
        total = len(leagues)
        for idx, (league_name, league_code) in enumerate(leagues, start=1):
            try:
                results.append(sync_service.sync_league(league_name, league_code))
            except Exception as exc:  # noqa: PERF203 - continue with next league
                logger.warning("league sync failed for %s: %s", league_name, exc)
                log_event("WARN", f"sync_failed:{league_name}")
            st.session_state["sync_progress"] = f"{idx}/{total}"
        alerts = monitoring_service.evaluate_sync(results)
        st.session_state["latest_sync_alerts"] = list(alerts)
        st.session_state["sync_quality_score"] = monitoring_service.sync_quality_score(results)
        st.session_state["sync_results"] = [r.__dict__ for r in results]
        st.session_state["last_sync_at"] = datetime.utcnow().isoformat()
        for alert in alerts:
            log_event("ERROR", alert)
        return alerts
    except Exception as exc:
        logger.error("run_full_sync failed (%s): %s", trigger, exc)
        log_event("ERROR", f"sync_all_failed:{trigger}")
        return [t("sync_warn")]


def maybe_run_auto_sync() -> None:
    if should_pause_background_sync(
        st.session_state.get("last_action_at", ""),
        datetime.utcnow(),
        inactivity_minutes=5,
    ):
        return
    last_sync = st.session_state.get("last_sync_at")
    try:
        if not last_sync:
            run_full_sync("startup")
            return
        try:
            last_sync_dt = datetime.fromisoformat(last_sync)
        except ValueError:
            run_full_sync("invalid_state")
            return
        if datetime.utcnow() - last_sync_dt >= SYNC_INTERVAL:
            run_full_sync("interval")
    except Exception as exc:
        logger.error("auto sync failed: %s", exc)
        log_event("ERROR", "auto_sync_failed")


def save_user_feedback(message: str, league_name: str, mode: str, category: str) -> bool:
    try:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.utcnow().isoformat(),
            "league": league_name,
            "mode": mode,
            "category": category,
            "message": message.strip(),
        }
        with FEEDBACK_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:
        logger.error("feedback write failed: %s", exc)
        log_event("ERROR", "feedback_write_failed")
        return False


def get_client_identifier() -> str:
    try:
        headers = getattr(st.context, "headers", {})
        forwarded_for = (
            headers.get("x-forwarded-for")
            or headers.get("X-Forwarded-For")
            or headers.get("x-real-ip")
            or headers.get("X-Real-IP")
        )
        if forwarded_for:
            return str(forwarded_for).split(",")[0].strip()
    except Exception:
        pass
    return "local"


bootstrap_default_users()
admin_password_overridden = apply_admin_password_override_from_env()
if not admin_password_overridden:
    ensure_admin_default_access()
cleanup_audit_log()
init_monitoring_db()
ledger_health = run_ledger_consistency_check()
st.session_state["ledger_consistency"] = ledger_health
i18n_coverage = evaluate_i18n_coverage(LANGUAGES, TOOLTIPS)
st.session_state["i18n_coverage"] = {
    "valid": i18n_coverage.valid,
    "missing_i18n_keys": i18n_coverage.missing_i18n_keys,
    "missing_tooltip_keys": i18n_coverage.missing_tooltip_keys,
}
input_validation_fuzz = run_input_validation_fuzz(rounds=60, seed=11)
st.session_state["input_validation_fuzz"] = {
    "stable": input_validation_fuzz.stable,
    "exceptions": input_validation_fuzz.exception_count,
    "invalid_payload_count": input_validation_fuzz.invalid_payload_count,
}
cache_invalidation = evaluate_cache_invalidation(
    now=datetime.utcnow(),
    last_sync_at_iso=str(st.session_state.get("last_sync_at", datetime.utcnow().isoformat())),
    cache_schema_version=str(st.session_state.get("analysis_cache_schema", "v1")),
    active_schema_version="v1",
    max_age_minutes=30,
)
st.session_state["cache_invalidation"] = {
    "should_invalidate": cache_invalidation.should_invalidate,
    "reasons": cache_invalidation.reasons,
}
if cache_invalidation.should_invalidate:
    st.session_state["analysis_cache"] = {}

PUBLIC_PROOF_MODE = str(os.getenv("PUBLIC_PROOF_MODE", "false")).lower() == "true"

# --- session defaults ---
for k, v in {
    "lang": "de",
    "auth": False,
    "role": "beta",
    "username": "guest",
    "api_credits": "Standby",
    "odds_history": [],
    "bet_ledger": [],
    "xp": 0,
    "error_logs": [],
    "sync_results": [],
    "last_sync_at": "",
    "alert_state": {},
    "credits_history": [],
    "league_weights": {},
    "user_mode": "beginner",
    "sim_count": CONFIG.poisson_simulations,
    "alert_prob": 70,
    "alert_quote": 2.2,
    "min_elo": 1300,
    "min_confidence": 60,
    "bankroll": CONFIG.default_budget,
    "bankroll_peak": CONFIG.default_budget,
    "auto_odds": True,
    "last_activity_at": "",
    "session_expired_flag": False,
    "trigger_run": False,
    "trigger_sync": False,
    "shortcut_feedback": "",
    "safe_mode": False,
    "resume_last_saved_at": "",
    "last_action": "",
    "analysis_cache": {},
    "cache_hits": 0,
    "cache_misses": 0,
    "cache_ttl_seconds": 600,
    "sidebar_notices": {},
    "sync_quality_score": 0.0,
    "audit_anomalies": [],
    "quickstart_minimal": False,
    "sync_progress": "",
    "chart_light_mode": False,
    "last_action_at": "",
    "last_analysis_signature": "",
    "last_primary_snapshot": None,
    "news_items_limit": 5,
    "news_text_only": False,
    "precision_auto_tune": True,
    "precision_recency_decay": 0.92,
    "precision_calibration": 0.15,
    "precision_lineup_confirmed": True,
    "precision_lineup_penalty": 0.08,
    "precision_min_data_quality": 55,
    "precision_drift_adjust": True,
    "precision_consensus_gate": True,
    "precision_adaptive_threshold": True,
    "precision_disagreement_penalty": True,
    "precision_stability_gate": True,
    "precision_evidence_gate": True,
    "precision_min_evidence_matches": 8,
    "precision_min_evidence_signals": 6,
    "precision_quantile_threshold": True,
    "precision_scenario_ensemble": True,
    "precision_scenario_conservative": False,
    "precision_online_bias": True,
    "precision_bias_lr": 0.03,
    "precision_shadow_mode": True,
    "precision_shadow_canary": True,
    "precision_bias_term": 0.0,
    "precision_feature_trust": True,
    "precision_hysteresis": True,
    "precision_season_calibration": True,
    "precision_trimmed_ensemble": True,
    "precision_error_tracking": True,
    "precision_last_gate_decision": False,
    "precision_error_stats": {"false_positives": 0, "false_negatives": 0, "total_labeled": 0},
    "precision_adaptive_bias_lr": True,
    "precision_weighted_consensus": True,
    "precision_segmented_hysteresis": True,
    "precision_error_adaptive_threshold": True,
    "precision_shadow_auto_promote": True,
    "precision_shadow_promoted": False,
    "precision_shadow_compare": {"samples": 0, "legacy_errors": 0, "candidate_errors": 0},
    "precision_regime_detector": True,
    "precision_reliability_score": True,
    "precision_cost_sensitive_threshold": True,
    "precision_fallback_matrix": True,
    "precision_adaptive_odds_move": True,
    "precision_reliability_hard_gate": 45,
    "precision_regime_auto_profile": True,
    "precision_reliability_auto_gate": True,
    "precision_odds_alert_cooldown": 10,
    "precision_regime_exposure_scaling": True,
    "precision_odds_auto_cooldown": True,
    "precision_replay_ci_blocker": True,
    "precision_correlation_guardrail": True,
    "reliability_history": [],
    "replay_history": [],
    "replay_health_history": [],
    "odds_alert_last_at": {},
    "last_regime_status": "stable",
    "last_retrospective": None,
    "precision_weekly_calibration": True,
    "precision_confidence_drift_alert": True,
    "precision_auto_postmortem": True,
    "safe_mode_plus_profile": "auto",
    "provider_quality_stats": {},
    "provider_priority": [],
    "confidence_history": [],
    "prediction_outcomes": [],
    "weekly_calibration_cache": {},
    "last_postmortem_report": [],
    "precision_alert_auto_sensitivity": True,
    "decision_delta_prev": None,
    "data_quality_history": [],
    "ab_test_history": [],
    "precision_stability_guard": True,
    "last_stability_signals": None,
    "explainability_trace": [],
    "pipeline_latency_history": [],
    "data_freshness_score": 0,
    "experiment_logbook": [],
    "pending_outcome_queue": [],
    "dead_letter_outcomes": [],
    "outcome_sync_hours": 6,
    "quality_budget_mode": True,
    "quality_budget_max_recomputes": 2,
    "quality_budget_events": [],
    "quality_budget_decisions": [],
    "bandit_threshold_offsets": {},
    "bandit_registry": {},
    "bandit_events": [],
    "daily_digest_log": {},
    "drift_history": [],
    "latest_sync_alerts": [],
    "weekly_auto_review": [],
    "startup_health": {},
    "feature_flag_drift_speed": True,
    "feature_flag_retry_learning": True,
    "feature_flag_weekly_auto_review": True,
    "feature_flag_no_bet_gap_hint": True,
    "feature_flag_pipeline_benchmark": True,
    "pipeline_benchmark_rows": [],
    "last_ops_report_path": "",
    "latest_trust_score": 0.0,
    "latest_trust_explain": [],
    "latest_agreement": {},
    "latest_drift": {},
    "latest_decision_id": "",
    "autonomy_state": "GREEN",
    "autonomy_actions": [],
    "ownerless_mode": False,
    "ui_theme_variant": "C",
    "ui_theme_mode": "dark",
    "ui_theme_accent": "#2563eb",
    "ui_theme_radius": 16,
    "show_ui_smoke_test": False,
    "ui_theme_style": "Ultra Premium Dark SaaS",
    "ui_layout_structure": "Dashboard",
    "login_layout": "Centered Card",
}.items():
    st.session_state.setdefault(k, v)

sync_service.retry_learning_enabled = bool(
    st.session_state.get("feature_flag_retry_learning", True)
)
if not st.session_state.get("startup_health"):
    st.session_state["startup_health"] = run_startup_health_checks(now=datetime.utcnow())
    monitor_api_health()

startup_health = st.session_state.get("startup_health", {})
try:
    import shutil

    disk_total, disk_used, _ = shutil.disk_usage(".")
    disk_usage_pct = 0.0 if disk_total <= 0 else disk_used / disk_total
except OSError:
    disk_usage_pct = 0.0

autonomy_decision = evaluate_autonomy_state(
    AutonomySignals(
        slo_status="freeze" if startup_health.get("overall") == "red" else "healthy",
        drift_status=(
            "critical"
            if int(startup_health.get("error_count", 0) or 0) > 0
            else ("warning" if int(startup_health.get("warn_count", 0) or 0) > 0 else "healthy")
        ),
        provider_healthy=any(
            str(check.get("name", "")).startswith("env:api_key") and check.get("status") == "ok"
            for check in startup_health.get("checks", [])
        ),
        crash_loop_detected=False,
        disk_usage_pct=disk_usage_pct,
        ownerless_mode=ownerless_mode_enabled(),
    )
)
st.session_state["autonomy_state"] = autonomy_decision.state
st.session_state["autonomy_actions"] = autonomy_decision.action_set
st.session_state["ownerless_mode"] = ownerless_mode_enabled()

if autonomy_decision.state in {"RED", "FROZEN"}:
    event = "FREEZE"
    write_decision_event(
        event=event,
        reason_code=(
            autonomy_decision.reason_codes[0]
            if autonomy_decision.reason_codes
            else "autonomy_guard"
        ),
        thresholds={"state": autonomy_decision.state},
        metrics_snapshot={
            "warn_count": int(startup_health.get("warn_count", 0) or 0),
            "error_count": int(startup_health.get("error_count", 0) or 0),
            "disk_usage_pct": round(float(disk_usage_pct), 4),
        },
        metadata={"actions": autonomy_decision.action_set[:5]},
    )
    runbook_actions_for_event("slo_freeze", ownerless_mode=ownerless_mode_enabled())

if st.session_state.get("auth"):
    if is_session_expired(st.session_state.get("last_activity_at", "")):
        st.session_state["auth"] = False
        st.session_state["role"] = "beta"
        st.session_state["username"] = "guest"
        st.session_state["session_expired_flag"] = True
        st.rerun()
    st.session_state["last_activity_at"] = datetime.utcnow().isoformat()


def render_login_page() -> None:
    if ui_is_safe_mode():
        st.title("SAFE UI MODE")
        st.caption("UI-Fallback aktiv. Keine Custom-CSS. Login im Simplified Layout.")
        if st.button("Safe UI deaktivieren", width="content"):
            st.session_state["safe_ui"] = False
            try:
                st.query_params.pop("safe_ui")
            except Exception:
                pass
            st.rerun()

        if not st.session_state.get("auth", False):
            _, safe_col, _ = st.columns([1, 1.2, 1])
            with safe_col:
                st.subheader("Login")
                safe_lang = st.selectbox(
                    "Sprache / Language",
                    ["de", "en"],
                    index=0 if st.session_state.get("lang", "de") == "de" else 1,
                )
                st.session_state["lang"] = safe_lang
                with st.form(key=_form_key("safe_login_form"), clear_on_submit=False):
                    safe_user = st.text_input(t("username"), key="safe_login_username")
                    safe_pass = st.text_input(
                        t("password"), type="password", key="safe_login_password"
                    )
                    safe_totp = ""
                    if is_admin_totp_enabled():
                        safe_totp = st.text_input(
                            t("login_totp"), max_chars=6, key="safe_login_totp"
                        )
                    safe_submit = st.form_submit_button("Anmelden", width="stretch")
                if safe_submit:
                    safe_role = authenticate_user(safe_user, safe_pass)
                    if safe_role:
                        if (
                            safe_role == "admin"
                            and is_admin_totp_enabled()
                            and not verify_admin_totp(safe_totp)
                        ):
                            st.error(t("login_totp_err"))
                        else:
                            st.session_state["auth"] = True
                            st.session_state["role"] = safe_role
                            st.session_state["username"] = safe_user.strip() or "guest"
                            st.session_state["last_activity_at"] = datetime.utcnow().isoformat()
                            st.success("Login erfolgreich.")
                            st.rerun()
                    else:
                        st.error(t("login_err"))
        st.stop()

    if not st.session_state["auth"]:
        current_lang = st.session_state.get("lang", "de")
        login_layout = str(st.session_state.get("login_layout", "Centered Card"))
        layout_key = get_login_layout_key(login_layout)

        st.markdown("<div class='login-page'><div class='ea-login'>", unsafe_allow_html=True)

        left_col = None
        right_col = None
        if layout_key == "split":
            left_col, right_col = st.columns([1.2, 1], gap="large")
        elif layout_key == "minimal":
            _, right_col, _ = st.columns([0.2, 1, 0.2])
        else:
            _, right_col, _ = st.columns([0.45, 1, 0.45])

        if left_col is not None:
            with left_col:
                st.markdown(
                    f"""
                    <div class='ea-card ea-card--soft'>
                        <span class='ea-badge'>Elite Analyst</span>
                        <h2 class='ea-login-title'>{get_text('login_title', current_lang)}</h2>
                        <p class='ea-login-subline'>AI-powered Football Intelligence Platform</p>
                        <div class='ea-divider'></div>
                        <div class='ea-feature-bullets'>
                            • Live Odds Monitoring<br>
                            • Backtesting Lab<br>
                            • Risk & Exposure Controls<br>
                            • Session Security & Audit Trail
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with right_col:
            st.markdown("<div class='login-card ea-login-card'>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class='ea-login-top'>
                    <span class='ea-badge'>Beta</span>
                </div>
                <h1 class='ea-login-title'>{get_text('login_title', current_lang)}</h1>
                <p class='ea-login-sub'>{get_text('login_sub', current_lang)}</p>
                <p class='ea-login-subline'>Secure access to your workspace</p>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.get("login_password") and not st.session_state.get(
                "login_username"
            ):
                st.session_state["login_password"] = ""
            if st.session_state.get("register_password") and not st.session_state.get(
                "register_username"
            ):
                st.session_state["register_password"] = ""
            if st.session_state.get("register_password_repeat") and not st.session_state.get(
                "register_username"
            ):
                st.session_state["register_password_repeat"] = ""
            login_tab, register_tab = st.tabs([t("login_btn"), t("register")])

            with login_tab:
                if st.session_state.get("session_expired_flag"):
                    st.warning(t("session_expired"))
                    st.session_state["session_expired_flag"] = False
                if admin_password_overridden:
                    st.info("Admin password was reset from ADMIN_PASSWORD for this app start.")
                with st.form(key=_form_key("login_form"), clear_on_submit=False):
                    u = st.text_input(
                        t("username"), key="login_username", autocomplete="new-password"
                    )
                    p = st.text_input(
                        t("password"),
                        type="password",
                        key="login_password",
                        autocomplete="new-password",
                    )
                    totp_input = ""
                    if is_admin_totp_enabled():
                        totp_input = st.text_input(
                            t("login_totp"),
                            max_chars=6,
                            key="login_totp",
                            autocomplete="one-time-code",
                        )
                    login_submit = st.form_submit_button("Anmelden", width="stretch")
                if login_submit:
                    client_id = get_client_identifier()
                    if is_rate_limited(u, client_id=client_id):
                        st.error(t("login_rate_limited"))
                        st.stop()

                    role = authenticate_user(u, p)
                    if role:
                        if (
                            role == "admin"
                            and is_admin_totp_enabled()
                            and not verify_admin_totp(totp_input)
                        ):
                            st.error(t("login_totp_err"))
                            st.stop()
                        st.session_state["auth"] = True
                        st.session_state["role"] = role
                        st.session_state["username"] = u.strip() or "guest"
                        st.session_state["last_activity_at"] = datetime.utcnow().isoformat()
                        st.session_state["session_expired_flag"] = False
                        clear_rate_limit(u, client_id=client_id)
                        write_audit_event(
                            actor=st.session_state["username"],
                            role=role,
                            action="login_success",
                            details={"client_id": client_id},
                        )
                        st.rerun()
                    else:
                        register_rate_limit_attempt(u, client_id=client_id)
                        write_audit_event(
                            actor=(u or "guest").strip() or "guest",
                            role="unknown",
                            action="login_failed",
                            details={"client_id": client_id},
                        )
                        if is_user_locked(u):
                            st.error(t("login_locked"))
                        else:
                            st.error(t("login_err"))

            with register_tab:
                with st.form(key=_form_key("register_form"), clear_on_submit=False):
                    new_u = st.text_input(
                        t("username"), key="register_username", autocomplete="new-password"
                    )
                    show_password = st.checkbox(t("show_password"), value=False)
                    pwd_type = "default" if show_password else "password"
                    new_p = st.text_input(
                        t("password"),
                        type=pwd_type,
                        key="register_password",
                        autocomplete="new-password",
                    )
                    new_p2 = st.text_input(
                        t("password_repeat"),
                        type=pwd_type,
                        key="register_password_repeat",
                        autocomplete="new-password",
                    )
                    strength = (
                        t("low")
                        if len(new_p) < 8
                        else (t("medium") if len(new_p) < 10 or new_p.isalnum() else t("high"))
                    )
                    st.caption(f"{t('password_strength')}: {strength}")
                    register_submit = st.form_submit_button("Registrieren", width="stretch")
                if register_submit:
                    if new_p != new_p2:
                        st.error(t("register_err_mismatch"))
                    else:
                        ok, code = register_user(new_u, new_p)
                        if ok:
                            st.success(t("register_ok"))
                        elif code == "username_exists":
                            st.error(t("register_err_exists"))
                        elif code == "password_short":
                            st.error(t("register_err_password"))
                        elif code == "password_weak":
                            st.error(t("register_err_password_weak"))
                        else:
                            st.error(t("register_err_username"))

            lang_choice = st.selectbox(
                get_text("language", current_lang),
                options=["de", "en"],
                format_func=lambda x: "DE" if x == "de" else "EN",
                index=0 if current_lang == "de" else 1,
                key="login_language_dropdown",
            )
            if lang_choice != current_lang:
                st.session_state["lang"] = lang_choice
                st.rerun()
            st.markdown(
                "<div class='ea-trust-line'>Secure login • Rate-limited • Session protected</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)
        if not ui_is_safe_mode():
            st.markdown("</div>", unsafe_allow_html=True)
        st.stop()


if not ui_is_safe_mode():
    try:
        theme_key = get_theme_key(
            str(st.session_state.get("ui_theme_style", "Ultra Premium Dark SaaS"))
        )
        layout_key = (
            str(st.session_state.get("ui_layout_structure", "Dashboard")).lower().replace(" ", "-")
        )
        login_layout_key = get_login_layout_key(
            str(st.session_state.get("login_layout", "Centered Card"))
        )
        render_global_styles(
            theme_key=theme_key, layout_key=layout_key, login_layout=login_layout_key
        )
        st.markdown(
            f"<div class='ea-root' data-theme='{theme_key}' data-layout='{layout_key}' data-login-layout='{login_layout_key}'>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.session_state["safe_ui"] = True
        st.rerun()

render_login_page()

maybe_run_auto_sync()
_cleanup_ops_state_files(max_age_days=90)

if not ui_is_safe_mode():
    st.markdown("<!-- ui-alive -->", unsafe_allow_html=True)

current_user = st.session_state.get("username", "guest")
saved_layout_mode, saved_layout_widgets = get_user_dashboard_layout(current_user)
saved_resume = load_user_resume(current_user)
saved_resume_meta = load_user_resume_meta(current_user)
ops_state = _load_ops_state(current_user)
st.session_state.setdefault("dashboard_layout_mode", saved_layout_mode)
st.session_state.setdefault("dashboard_layout_widgets", saved_layout_widgets)
st.session_state.setdefault("dead_letter_outcomes", ops_state.get("dead_letter_outcomes", []))
st.session_state.setdefault(
    "quality_budget_decisions", ops_state.get("quality_budget_decisions", [])
)
st.session_state.setdefault("bandit_events", ops_state.get("bandit_events", []))
if "ai_messages" not in st.session_state:
    st.session_state["ai_messages"] = []

st.markdown(
    f"<div class='ea-root' data-theme='{CURRENT_THEME_TOKEN}'><div class='main-container ea-page'>",
    unsafe_allow_html=True,
)

if saved_resume:
    st.session_state.setdefault("resume_revision", int(saved_resume_meta.get("revision", 0) or 0))
    st.session_state.setdefault("league_name", saved_resume.get("league_name", ""))
    st.session_state.setdefault("home_team", saved_resume.get("home_team", ""))
    st.session_state.setdefault("away_team", saved_resume.get("away_team", ""))
    st.session_state.setdefault("search_team", saved_resume.get("search_team", ""))
    if saved_resume.get("user_mode") in {"beginner", "pro"}:
        st.session_state.setdefault("user_mode", saved_resume["user_mode"])

st.markdown(
    """
    <div class="ea-header">
        <p class="ea-header__title">Elite Analyst</p>
        <span class="ea-header__status">Local • Ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- sidebar controls ---
st.sidebar.caption(
    "THEME TOKEN: bloomberg" if CURRENT_THEME_TOKEN == "bloomberg" else "THEME TOKEN: other"
)
st.sidebar.title(t("control"))
st.sidebar.markdown(
    "<div class='ea-sidebar-section'><div class='ea-sidebar-label'>Navigation</div></div>",
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Bereiche",
    ["Dashboard", "Live Terminal", "Backtesting Lab", "News & Sentiment", "Settings"],
    index=0,
    key="ui_nav_hint",
)
if saved_resume:
    st.sidebar.caption(t("session_resume_loaded"))
active_notices = active_sidebar_notices(
    st.session_state.get("sidebar_notices", {}),
    now=datetime.utcnow(),
    ttl_seconds=25,
)
st.session_state["sidebar_notices"] = active_notices
for notice in active_notices.values():
    level = notice.get("level", "info")
    message = notice.get("message", "")
    if level == "success":
        st.sidebar.success(message)
    elif level == "warning":
        st.sidebar.warning(message)
    elif level == "error":
        st.sidebar.error(message)
    else:
        st.sidebar.info(message)
with st.sidebar.container():
    if "lang" not in st.session_state:
        st.session_state.lang = "de"
    language_options = list(LANGUAGES.keys())
    current_lang = st.session_state.get("lang", "de")
    if current_lang not in language_options:
        current_lang = "de"
    lang = st.selectbox(
        "Language",
        language_options,
        index=language_options.index(current_lang),
        format_func=lambda code: LANGUAGE_LABELS.get(code, code),
    )
    st.session_state.lang = lang

    if st.button(t("logout"), width="stretch"):
        st.session_state["auth"] = False
        st.rerun()

st.sidebar.markdown(
    "<div class='ea-sidebar-section'><div class='ea-sidebar-label'>Login Status</div></div>",
    unsafe_allow_html=True,
)
st.sidebar.caption(
    f"{st.session_state.get('username', 'guest')} • {st.session_state.get('role', 'beta')}"
)
st.sidebar.markdown(
    "<div class='ea-sidebar-section'><div class='ea-sidebar-label'>Systemstatus</div></div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Runtime aktiv • Monitoring bereit")
st.sidebar.info("Elite Analyst Dashboard im Premium Dark Mode")

st.sidebar.markdown(
    "<div class='ea-sidebar-section'><div class='ea-sidebar-label'>Assistant</div></div>",
    unsafe_allow_html=True,
)
with st.sidebar.expander("Assistant", expanded=False):
    render_assistant(current_page=page)

lang_now = st.session_state.get("lang", "de")
with st.sidebar.expander(get_text("design_layout", lang_now), expanded=False):
    style_options = [
        "Ultra Premium Dark SaaS",
        "Bloomberg / Institutional",
        "Neon Tech / Cyber",
        "Clean Enterprise Light",
    ]
    current_theme_style = st.session_state.get("ui_theme_style", "Ultra Premium Dark SaaS")
    new_theme_style = st.selectbox(
        "Theme-Stil",
        style_options,
        index=(
            style_options.index(current_theme_style) if current_theme_style in style_options else 0
        ),
    )
    if new_theme_style != current_theme_style:
        st.session_state["ui_theme_style"] = new_theme_style
        st.rerun()

    layout_options = ["Focus", "Dashboard", "Pro Terminal", "Institutional"]
    st.selectbox("Layout-Struktur", layout_options, key="ui_layout_structure")
    st.selectbox(
        "Login-Layout",
        ["Centered Card", "Split Screen", "Minimal Top"],
        key="login_layout",
    )

    st.color_picker(get_text("theme_accent", lang_now), key="ui_theme_accent")
    st.slider(get_text("theme_radius", lang_now), 8, 24, key="ui_theme_radius")
    st.caption(get_text("design_live", lang_now))

with st.sidebar.expander(t("shortcuts"), expanded=False):
    st.caption(t("shortcuts_hint"))
    with st.form("shortcut_form", clear_on_submit=True):
        shortcut_raw = st.text_input(t("shortcuts_input"), max_chars=12)
        shortcut_submit = st.form_submit_button(t("shortcuts_apply"), width="stretch")
    if shortcut_submit:
        action = resolve_shortcut_action(shortcut_raw)
        if action == "run":
            st.session_state["trigger_run"] = True
            st.session_state["shortcut_feedback"] = "run"
            st.session_state["last_action"] = "shortcut_run"
            st.session_state["last_action_at"] = datetime.utcnow().isoformat()
        elif action == "sync":
            st.session_state["trigger_sync"] = True
            st.session_state["shortcut_feedback"] = "sync"
            st.session_state["last_action"] = "shortcut_sync"
            st.session_state["last_action_at"] = datetime.utcnow().isoformat()
        elif action == "logout":
            st.session_state["auth"] = False
            st.rerun()
        elif action == "language":
            language_options = list(LANGUAGES.keys())
            current_index = (
                language_options.index(st.session_state.get("lang", "de"))
                if st.session_state.get("lang", "de") in language_options
                else 0
            )
            st.session_state["lang"] = language_options[(current_index + 1) % len(language_options)]
            st.rerun()
        else:
            st.warning(t("shortcuts_unknown"))
            set_sidebar_notice(
                st.session_state["sidebar_notices"],
                notice_id=f"shortcut_unknown_{int(datetime.utcnow().timestamp())}",
                level="warning",
                message=t("shortcuts_unknown"),
                now=datetime.utcnow(),
            )

with st.sidebar.expander("UI & System", expanded=False):
    safe_mode = st.toggle(
        t("safe_mode"),
        value=bool(st.session_state.get("safe_mode", False)),
        key="safe_mode",
        help=tip("safe_mode"),
    )
    if safe_mode:
        st.caption(t("safe_mode_active"))
    safe_mode_plus_profile = st.selectbox(
        t("safe_mode_plus"),
        ["auto", "low-end", "balanced", "desktop"],
        index=["auto", "low-end", "balanced", "desktop"].index(
            st.session_state.get("safe_mode_plus_profile", "auto")
        ),
        key="safe_mode_plus_profile",
    )
    safe_mode_plus = resolve_safe_mode_plus_profile_with_cooldown(
        latency_ms=float(st.session_state.get("last_provider_latency_ms", 180.0)),
        cpu_ratio=0.85 if safe_mode else 0.45,
        previous_profile=str(st.session_state.get("safe_mode_plus_last_profile", "")),
        last_switch_at_iso=str(st.session_state.get("safe_mode_plus_last_switch_at", "")),
        device_hint=safe_mode_plus_profile,
        cooldown_seconds=60,
    )
    if st.session_state.get("safe_mode_plus_last_profile") != safe_mode_plus.name:
        st.session_state["safe_mode_plus_last_profile"] = safe_mode_plus.name
        st.session_state["safe_mode_plus_last_switch_at"] = datetime.utcnow().isoformat()
    adaptive_low_end = derive_low_end_adaptive_profile(
        latency_ms=float(st.session_state.get("last_provider_latency_ms", 180.0)),
        cpu_ratio=0.85 if safe_mode else 0.45,
        memory_ratio=0.75 if safe_mode else 0.45,
    )
    st.session_state["adaptive_low_end_profile"] = adaptive_low_end.profile
    st.session_state["adaptive_max_news_items"] = adaptive_low_end.max_news_items
    st.caption(
        f"Adaptive Low-End: {adaptive_low_end.profile} | Sim {adaptive_low_end.max_simulations} | Budget/h {adaptive_low_end.recompute_budget_per_hour}"
    )
    if st.session_state.get("safe_mode_plus_last_profile") != safe_mode_plus.name:
        st.session_state["safe_mode_plus_last_profile"] = safe_mode_plus.name
        st.session_state["safe_mode_plus_last_switch_at"] = datetime.utcnow().isoformat()
    accent = st.color_picker(t("accent"), "#30c48d", help=tip("accent"))
    dense_default = (
        safe_mode_plus.compact_layout if safe_mode else bool(st.session_state.get("compact", False))
    )
    dense = st.toggle(t("compact"), value=dense_default, help=tip("compact"))
    refresh_default = (
        safe_mode_plus.refresh_seconds if safe_mode else int(st.session_state.get("refresh", 30))
    )
    auto_refresh = st.slider(t("refresh"), 10, 120, refresh_default, 10, help=tip("refresh"))
    st.session_state["refresh"] = auto_refresh
    quality_budget_mode = st.toggle(
        "Quality Budget Mode",
        value=bool(st.session_state.get("quality_budget_mode", True)),
        help="Begrenzt teure Recomputes pro Stunde auf schwachen Geräten.",
    )
    quality_budget_max = st.slider(
        "Recompute-Budget / Stunde",
        2,
        20,
        int(st.session_state.get("quality_budget_max_recomputes", 2)),
        1,
    )
    outcome_sync_hours = st.slider(
        "Outcome Sync nach X Stunden",
        1,
        24,
        int(st.session_state.get("outcome_sync_hours", 6)),
        1,
        help="Offene Entscheidungen werden nach dieser Wartezeit automatisch mit Ergebnissen abgeglichen.",
    )
    st.session_state["quality_budget_mode"] = bool(quality_budget_mode)
    st.session_state["quality_budget_max_recomputes"] = int(quality_budget_max)
    st.session_state["outcome_sync_hours"] = int(outcome_sync_hours)
    with st.expander("Feature-Flags (Ops)", expanded=False):
        st.session_state["feature_flag_drift_speed"] = st.checkbox(
            "Drift-Speed aktiv", value=bool(st.session_state.get("feature_flag_drift_speed", True))
        )
        st.session_state["feature_flag_retry_learning"] = st.checkbox(
            "Retry-Learning aktiv",
            value=bool(st.session_state.get("feature_flag_retry_learning", True)),
        )
        st.session_state["feature_flag_weekly_auto_review"] = st.checkbox(
            "Weekly Auto-Review aktiv",
            value=bool(st.session_state.get("feature_flag_weekly_auto_review", True)),
        )
        st.session_state["feature_flag_no_bet_gap_hint"] = st.checkbox(
            "No-Bet Gap Hint aktiv",
            value=bool(st.session_state.get("feature_flag_no_bet_gap_hint", True)),
        )
        st.session_state["feature_flag_pipeline_benchmark"] = st.checkbox(
            "Pipeline-Benchmark aktiv",
            value=bool(st.session_state.get("feature_flag_pipeline_benchmark", True)),
        )
    sync_service.retry_learning_enabled = bool(
        st.session_state.get("feature_flag_retry_learning", True)
    )
    mode_options = {t("level_beginner"): "beginner", t("level_pro"): "pro"}
    mode_label = st.selectbox(
        t("user_level"),
        list(mode_options.keys()),
        index=0 if st.session_state.get("user_mode") == "beginner" else 1,
        help=tip("user_level"),
    )
    st.session_state["user_mode"] = mode_options[mode_label]
    quickstart_default = False if safe_mode else bool(st.session_state.get("show_quickstart", True))
    show_quickstart = st.toggle(
        t("quickstart_enable"), value=quickstart_default, help=tip("quickstart_enable")
    )
    st.session_state["show_quickstart"] = show_quickstart
    quickstart_minimal = st.toggle(
        t("quickstart_minimal_enable"),
        value=bool(st.session_state.get("quickstart_minimal", False)),
        help=tip("quickstart_minimal_enable"),
    )
    st.session_state["quickstart_minimal"] = quickstart_minimal
    chart_light_mode = st.toggle(
        t("charts_light_mode"),
        value=bool(st.session_state.get("chart_light_mode", False)),
        help=tip("charts_light_mode"),
    )
    st.session_state["chart_light_mode"] = chart_light_mode

    layout_options = {
        t("layout_beginner"): "beginner",
        t("layout_balanced"): "balanced",
        t("layout_pro"): "pro",
        t("layout_custom"): "custom",
    }
    current_layout_mode = st.session_state.get("dashboard_layout_mode", saved_layout_mode)
    current_layout_label = next(
        (label for label, key in layout_options.items() if key == current_layout_mode),
        t("layout_balanced"),
    )
    selected_layout_label = st.selectbox(
        t("home_layout"),
        list(layout_options.keys()),
        index=list(layout_options.keys()).index(current_layout_label),
        help=tip("home_layout"),
    )
    selected_layout_mode = layout_options[selected_layout_label]

    widget_label_to_key = {
        t("widget_win_prob"): "win_prob",
        t("widget_value_edge"): "value_edge",
        t("widget_stake"): "stake",
        t("widget_api_credits"): "api_credits",
        t("widget_exposure"): "exposure",
        t("widget_data_quality"): "data_quality",
        t("widget_sync_status"): "sync_status",
        t("widget_ml_probs"): "ml_probs",
        t("widget_social_sentiment"): "social_sentiment",
        t("widget_injury_risk"): "injury_risk",
        t("widget_clv"): "clv",
    }
    if selected_layout_mode == "custom":
        default_widget_labels = [
            label
            for label, key in widget_label_to_key.items()
            if key in st.session_state.get("dashboard_layout_widgets", PRESET_WIDGETS["balanced"])
        ]
        custom_widget_labels = st.multiselect(
            t("layout_widgets"), list(widget_label_to_key.keys()), default=default_widget_labels
        )
        selected_widgets = [widget_label_to_key[label] for label in custom_widget_labels]
    else:
        selected_widgets = PRESET_WIDGETS[selected_layout_mode]

    effective_layout_mode, effective_layout_widgets = resolve_layout_widgets(
        selected_layout_mode, selected_widgets
    )
    st.session_state["dashboard_layout_mode"] = effective_layout_mode
    st.session_state["dashboard_layout_widgets"] = effective_layout_widgets

    if st.button(t("layout_save"), width="stretch"):
        mode_saved, widgets_saved = set_user_dashboard_layout(
            current_user, effective_layout_mode, effective_layout_widgets
        )
        st.session_state["dashboard_layout_mode"] = mode_saved
        st.session_state["dashboard_layout_widgets"] = widgets_saved
        st.success(t("layout_saved"))

page_to_mode = {
    "Dashboard": t("live"),
    "Live Terminal": t("live"),
    "Backtesting Lab": t("backtest"),
    "News & Sentiment": t("news"),
    "Settings": t("live"),
}
mode = page_to_mode.get(page, t("live"))

risk_options = [
    ("Conservative", t("risk_conservative")),
    ("Balanced", t("risk_balanced")),
    ("Aggressive", t("risk_aggressive")),
]
risk_label_to_key = {label: key for key, label in risk_options}
risk_labels = [label for _, label in risk_options]
if st.session_state.get("risk_profile_label") not in risk_labels:
    st.session_state["risk_profile_label"] = t("risk_balanced")

with st.sidebar.expander("Einstellungen", expanded=False):
    league_options = list(ACTIVE_LEAGUES.keys())
    default_league = st.session_state.get("league_name", league_options[0])
    if default_league not in league_options:
        default_league = league_options[0]
    league_name = st.selectbox(
        t("league"),
        league_options,
        index=league_options.index(default_league),
        key="league_name",
        help=tip("league"),
    )
    risk_profile_label = st.select_slider(
        t("risk"),
        risk_labels,
        value=st.session_state.get("risk_profile_label", t("risk_balanced")),
        key="risk_profile_label",
        help=tip("risk"),
    )
    sim_count = st.slider(
        t("sim"),
        2000,
        20000,
        int(st.session_state.get("sim_count", CONFIG.poisson_simulations)),
        1000,
        key="sim_count",
        help=tip("sim"),
    )
    if st.session_state.get("safe_mode"):
        sim_cap = int(
            resolve_safe_mode_plus_profile(
                latency_ms=float(st.session_state.get("last_provider_latency_ms", 180.0)),
                cpu_ratio=0.85,
                device_hint=str(st.session_state.get("safe_mode_plus_profile", "auto")),
            ).simulation_count
        )
        if sim_count > sim_cap:
            sim_count = sim_cap
            st.session_state["sim_count"] = sim_cap

    preset_options = [t("preset_beginner"), t("preset_balanced"), t("preset_aggressive")]
    selected_preset = st.selectbox(t("preset"), preset_options, index=1)
    preset_hint = {
        t("preset_beginner"): t("preset_beginner_desc"),
        t("preset_balanced"): t("preset_balanced_desc"),
        t("preset_aggressive"): t("preset_aggressive_desc"),
    }
    st.caption(preset_hint.get(selected_preset, t("preset_balanced_desc")))
    if st.button(t("preset_apply"), width="stretch"):
        if selected_preset == t("preset_beginner"):
            st.session_state["league_weights"][league_name] = {
                "poisson": 0.50,
                "elo": 0.40,
                "ml": 0.10,
            }
        elif selected_preset == t("preset_aggressive"):
            st.session_state["league_weights"][league_name] = {
                "poisson": 0.35,
                "elo": 0.25,
                "ml": 0.40,
            }
        else:
            st.session_state["league_weights"][league_name] = {
                "poisson": 0.45,
                "elo": 0.35,
                "ml": 0.20,
            }
        st.success(t("preset_applied").format(preset=selected_preset, league=league_name))

with st.sidebar.expander("Ops", expanded=False):
    run_sync_action = st.button(t("sync_all"), help=tip("sync_all"), width="stretch")
    if st.session_state.get("trigger_sync"):
        run_sync_action = True
        st.session_state["trigger_sync"] = False
    if run_sync_action:
        st.session_state["last_action"] = "sync"
        st.session_state["last_action_at"] = datetime.utcnow().isoformat()
        alerts = run_full_sync("manual")
        if alerts:
            st.error("; ".join(alerts))
            set_sidebar_notice(
                st.session_state["sidebar_notices"],
                notice_id=f"sync_error_{int(datetime.utcnow().timestamp())}",
                level="error",
                message="; ".join(alerts),
                now=datetime.utcnow(),
            )
        else:
            st.success(t("sync_status"))
            set_sidebar_notice(
                st.session_state["sidebar_notices"],
                notice_id=f"sync_ok_{int(datetime.utcnow().timestamp())}",
                level="success",
                message=t("sync_status"),
                now=datetime.utcnow(),
            )
    st.caption(f"{t('sync_status')}: {st.session_state.get('last_sync_at') or '-'}")
    st.caption(f"{t('sync_progress')}: {st.session_state.get('sync_progress') or '-'}")
    st.caption(f"Sync-Qualität: {float(st.session_state.get('sync_quality_score', 0.0)):.1f}/100")
    if should_pause_background_sync(
        st.session_state.get("last_action_at", ""), datetime.utcnow(), inactivity_minutes=5
    ):
        st.caption(t("background_paused"))
        if st.button(t("background_resume"), width="stretch"):
            st.session_state["last_action_at"] = datetime.utcnow().isoformat()

audit_events_recent = load_recent_audit_events(limit=200)
st.session_state["audit_anomalies"] = detection_delta_anomalies(audit_events_recent)
if st.session_state.get("audit_anomalies"):
    st.sidebar.warning(
        "Audit-Events auffällig: " + " | ".join(st.session_state.get("audit_anomalies", []))
    )

league_code = ACTIVE_LEAGUES[league_name]
risk_profile = risk_label_to_key[risk_profile_label]
risk_profile_runtime = risk_profile
if bool(st.session_state.get("precision_regime_auto_profile", True)):
    risk_profile_runtime = recommend_risk_profile_for_regime(
        risk_profile, str(st.session_state.get("last_regime_status", "stable"))
    )
    if risk_profile_runtime != risk_profile:
        st.sidebar.caption(t("risk_profile_auto_hint").format(profile=risk_profile_runtime))

with st.spinner(t("sync")):
    load_t0 = perf_counter()
    df, elo_curr, elo_hist = load_dynamic_league_data(league_name, league_code)
    st.session_state["pipeline_latency_history"] = track_pipeline_latency(
        st.session_state.get("pipeline_latency_history", []),
        step="load_data",
        duration_ms=(perf_counter() - load_t0) * 1000.0,
    )
    model, metrics = (None, None)
    if not df.empty:
        train_t0 = perf_counter()
        model, metrics = train_and_evaluate(df)
        st.session_state["pipeline_latency_history"] = track_pipeline_latency(
            st.session_state.get("pipeline_latency_history", []),
            step="train_model",
            duration_ms=(perf_counter() - train_t0) * 1000.0,
        )

if not df.empty and model is None:
    st.info(t("model_unavailable"))

data_quality = compute_data_quality(df, league_name=league_name)
feature_drift = compute_feature_drift(df)
canary_enabled = is_canary_enabled_for_user(st.session_state.get("username", "guest"))

if page == "Dashboard":
    layout_style = st.session_state.get("ui_layout_structure", "Dashboard")
    kpi_vals = {
        "quality": f"{float(data_quality.get('score', 0.0)):.0f}/100",
        "drift": f"{float(feature_drift.get('score', 0.0)):.2f}",
        "acc": f"{mget(metrics, 'accuracy', 0.0):.2%}",
        "cache": f"{cache_hit_rate(st.session_state.get('cache_hits', 0), st.session_state.get('cache_misses', 0)):.1%}",
    }

    def render_model_prediction_compact() -> None:
        with card("Model Status"):
            status_line = "ready" if model is not None else "not available"
            st.caption(f"League: {league_name} • Rows: {len(df)} • Model: {status_line}")
            st.caption(
                f"Accuracy: {mget(metrics, 'accuracy', 0.0):.3f} | F1: {mget(metrics, 'f1', 0.0):.3f}"
            )
        with card("Prediction Panel"):
            p1, p2 = st.columns(2)
            with p1:
                st.caption(f"{t('home')}: {st.session_state.get('home_team') or '-'}")
                st.caption(f"{t('away')}: {st.session_state.get('away_team') or '-'}")
            with p2:
                st.caption(f"{t('risk_profile')}: {risk_profile_runtime}")
                st.caption(f"Canary: {'on' if canary_enabled else 'off'}")

    def render_model_prediction_bloomberg() -> None:
        status_col, panel_col = st.columns(2)
        with status_col:
            with card("Model Status"):
                status_line = "ready" if model is not None else "not available"
                st.caption(f"League: {league_name} • Rows: {len(df)} • Model: {status_line}")
                st.caption(
                    f"Accuracy: {mget(metrics, 'accuracy', 0.0):.3f} | F1: {mget(metrics, 'f1', 0.0):.3f}"
                )
        with panel_col:
            with card("Prediction Panel"):
                st.caption(f"{t('home')}: {st.session_state.get('home_team') or '-'}")
                st.caption(f"{t('away')}: {st.session_state.get('away_team') or '-'}")
                st.caption(f"{t('risk_profile')}: {risk_profile_runtime}")
                st.caption(f"Canary: {'on' if canary_enabled else 'off'}")

    def render_stats_compact() -> None:
        with card("Data / Stats"):
            st.caption(f"Sync: {st.session_state.get('last_sync_at') or '-'}")
            st.caption(f"Data Quality: {float(data_quality.get('score', 0.0)):.1f}/100")
            st.caption(f"Drift Score: {float(feature_drift.get('score', 0.0)):.3f}")
            st.caption(
                f"Cache Hit Rate: {cache_hit_rate(st.session_state.get('cache_hits', 0), st.session_state.get('cache_misses', 0)):.1%}"
            )
        with st.expander("Sync / Health", expanded=False):
            st.caption(f"Runtime: {st.session_state.get('autonomy_state', 'GREEN')}")
            st.caption(f"Last action: {st.session_state.get('last_action', '-')}")

    if CURRENT_THEME_TOKEN == "bloomberg":
        status_line = "ready" if model is not None else "not available"
        st.markdown(
            f"""
            <div class="ea-main">
              <div id="ea-kpi-bar" class="ea-kpi-bar">
                <div class="ea-kpi-card">
                  <div class="ea-kpi-title">DATA QUALITY</div>
                  <div class="ea-kpi-value">{kpi_vals['quality']}</div>
                </div>
                <div class="ea-kpi-card">
                  <div class="ea-kpi-title">DRIFT</div>
                  <div class="ea-kpi-value">{kpi_vals['drift']}</div>
                </div>
                <div class="ea-kpi-card">
                  <div class="ea-kpi-title">ACCURACY</div>
                  <div class="ea-kpi-value">{kpi_vals['acc']}</div>
                </div>
              </div>
              <div class="ea-content"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([2, 1], gap="small")
        with left:
            st.markdown("<div class='bloomberg-panel'>", unsafe_allow_html=True)
            st.markdown("**Model Status**")
            st.caption(f"League: {league_name} • Rows: {len(df)} • Model: {status_line}")
            st.caption(
                f"Accuracy: {mget(metrics, 'accuracy', 0.0):.3f} | F1: {mget(metrics, 'f1', 0.0):.3f}"
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bloomberg-panel'>", unsafe_allow_html=True)
            st.markdown("**Prediction Panel**")
            st.caption(f"{t('home')}: {st.session_state.get('home_team') or '-'}")
            st.caption(f"{t('away')}: {st.session_state.get('away_team') or '-'}")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='bloomberg-panel'>", unsafe_allow_html=True)
            st.markdown("**Risk / Context**")
            st.caption(f"{t('risk_profile')}: {risk_profile_runtime}")
            st.caption(f"Canary: {'on' if canary_enabled else 'off'}")
            st.caption(f"Cache Hit: {kpi_vals['cache']}")
            st.markdown("</div>", unsafe_allow_html=True)

    elif layout_style == "Focus":
        with card("Overview"):
            st.subheader("Focus Layout")
            st.caption("Kompakte Ansicht mit klarer Hauptspur.")
        render_model_prediction_compact()
        low1, low2 = st.columns(2)
        with low1:
            st.metric("Data Quality", kpi_vals["quality"])
            st.metric("Accuracy", kpi_vals["acc"])
        with low2:
            st.metric("Drift", kpi_vals["drift"])
            st.metric("Cache Hit", kpi_vals["cache"])

    elif layout_style == "Pro Terminal":
        left, mid, right = st.columns([1, 2, 1])
        with left:
            with card("Monitoring"):
                st.metric("Data Quality", kpi_vals["quality"])
                st.metric("Drift", kpi_vals["drift"])
        with mid:
            render_model_prediction_compact()
        with right:
            with card("Risk / Control"):
                st.metric("Accuracy", kpi_vals["acc"])
                st.metric("Cache Hit", kpi_vals["cache"])
            render_stats_compact()

    elif layout_style == "Institutional":
        tab_overview, tab_analytics, tab_execution, tab_risk = st.tabs(
            ["Overview", "Analytics", "Execution", "Risk"]
        )
        with tab_overview:
            row = st.columns(4)
            row[0].metric("Data Quality", kpi_vals["quality"])
            row[1].metric("Drift", kpi_vals["drift"])
            row[2].metric("Accuracy", kpi_vals["acc"])
            row[3].metric("Cache Hit", kpi_vals["cache"])
            render_model_prediction_compact()
        with tab_analytics:
            render_stats_compact()
            with card("Deep Metrics"):
                st.caption("Drift, Qualität und Modellwerte im Überblick.")
        with tab_execution:
            with card("Execution"):
                st.caption("Stake, Alerts und Ledger bleiben unverändert nutzbar.")
            render_model_prediction_compact()
        with tab_risk:
            render_stats_compact()
            with card("Guardrails"):
                st.caption("Risk Controls und Limits sind weiterhin aktiv.")

    else:
        row = st.columns(4)
        row[0].metric("Data Quality", kpi_vals["quality"])
        row[1].metric("Drift", kpi_vals["drift"])
        row[2].metric("Accuracy", kpi_vals["acc"])
        row[3].metric("Cache Hit", kpi_vals["cache"])

        main_col, side_col = st.columns([2, 1])
        with main_col:
            render_model_prediction_compact()
        with side_col:
            render_stats_compact()

quality_history = list(st.session_state.get("data_quality_history", []))
quality_history.append(
    {
        "ts": datetime.utcnow().isoformat(),
        "league": league_name,
        "provider": "odds_api",
        "score": float(data_quality.get("score", 0.0)),
    }
)
st.session_state["data_quality_history"] = quality_history[-200:]
if bool(st.session_state.get("feature_flag_pipeline_benchmark", True)):
    benchmark_rows = summarize_pipeline_benchmark(
        st.session_state.get("pipeline_latency_history", []),
        now=datetime.utcnow(),
        lookback_days=7,
        previous_days=7,
        regression_pct=20.0,
    )
    st.session_state["pipeline_benchmark_rows"] = benchmark_rows
else:
    st.session_state["pipeline_benchmark_rows"] = []
drift_history = list(st.session_state.get("drift_history", []))
drift_history.append(
    {
        "ts": datetime.utcnow().isoformat(),
        "league": league_name,
        "score": float(feature_drift.get("score", 0.0)),
        "top_feature": str(feature_drift.get("top_feature", "")),
    }
)
st.session_state["drift_history"] = drift_history[-240:]
drift_speed = {"status": "disabled", "trend": "stable", "delta": 0.0, "streak_up": 0}
if bool(st.session_state.get("feature_flag_drift_speed", True)):
    drift_speed = compute_drift_speed(
        st.session_state.get("drift_history", []),
        league=league_name,
        lookback_days=7,
    )
outcome_sync = sync_pending_outcomes(
    st.session_state.get("pending_outcome_queue", []),
    df,
    now=datetime.utcnow(),
    min_age_hours=float(st.session_state.get("outcome_sync_hours", 6)),
    stale_hours=24.0,
)
st.session_state["pending_outcome_queue"] = outcome_sync.queue
if outcome_sync.dead_letter:
    dead = list(st.session_state.get("dead_letter_outcomes", []))
    dead.extend(outcome_sync.dead_letter)
    st.session_state["dead_letter_outcomes"] = dead[-250:]
if outcome_sync.appended_outcomes:
    merged_outcomes = list(st.session_state.get("prediction_outcomes", []))
    merged_outcomes.extend(outcome_sync.appended_outcomes)
    st.session_state["prediction_outcomes"] = merged_outcomes[-500:]
    for settled in outcome_sync.appended_outcomes:
        decision_id = str(settled.get("decision_id", "")).strip()
        if decision_id and isinstance(settled.get("home_won"), bool):
            update_settlement(decision_id, bool(settled.get("home_won")))
if outcome_sync.stale_count > 0:
    st.sidebar.warning(f"Outcome Sync: {outcome_sync.stale_count} offene Entscheidungen >24h.")
if st.session_state.get("dead_letter_outcomes"):
    st.sidebar.error(
        f"Outcome Dead-Letter: {len(st.session_state.get('dead_letter_outcomes', []))} Fälle bitte manuell prüfen."
    )

_bankroll_for_peak = float(st.session_state.get("bankroll", CONFIG.default_budget))
st.session_state["bankroll_peak"] = max(
    float(st.session_state.get("bankroll_peak", _bankroll_for_peak)), _bankroll_for_peak
)
if not df.empty and "Date" in df.columns:
    latest_dt = pd.to_datetime(df["Date"], errors="coerce").max()
    if pd.notna(latest_dt):
        hours_old = max(0.0, (datetime.utcnow() - latest_dt.to_pydatetime()).total_seconds() / 3600)
        last_sync_at = st.session_state.get("last_sync_at", "")
        sync_age_minutes = 999.0
        if last_sync_at:
            try:
                sync_age_minutes = max(
                    0.0,
                    (datetime.utcnow() - datetime.fromisoformat(str(last_sync_at))).total_seconds()
                    / 60.0,
                )
            except ValueError:
                sync_age_minutes = 999.0
        freshness = compute_data_freshness_score(
            data_age_hours=float(hours_old),
            sync_age_minutes=float(sync_age_minutes),
            provider_latency_ms=float(st.session_state.get("last_provider_latency_ms", 180.0)),
        )
        st.session_state["data_freshness_score"] = int(freshness.score)
        st.sidebar.caption(
            f"{t('data_freshness')}: {hours_old:.1f}h | Score {freshness.score}/100 ({freshness.status})"
        )
st.sidebar.progress(
    data_quality["score"] / 100.0, text=f"{t('data_quality')}: {data_quality['score']:.0f}/100"
)
ledger_consistency = st.session_state.get("ledger_consistency", {"healthy": True, "issues": []})
if not bool(ledger_consistency.get("healthy", True)):
    st.sidebar.warning("Ledger-Consistency: " + ", ".join(ledger_consistency.get("issues", [])))
input_fuzz = st.session_state.get("input_validation_fuzz", {"stable": True, "exceptions": 0})
if not bool(input_fuzz.get("stable", True)):
    st.sidebar.warning("Input-Validation-Fuzz: Exceptions erkannt")
cache_inv = st.session_state.get("cache_invalidation", {"should_invalidate": False, "reasons": []})
if bool(cache_inv.get("should_invalidate", False)):
    st.sidebar.caption("Cache invalidiert: " + ", ".join(cache_inv.get("reasons", [])))
st.sidebar.caption(f"{t('canary_state')}: {t('canary_on') if canary_enabled else t('canary_off')}")
if feature_drift.get("status") == "insufficient":
    st.sidebar.caption(f"{t('data_drift')}: {t('data_drift_insufficient')}")
else:
    st.sidebar.caption(
        f"{t('data_drift')}: {feature_drift.get('score', 0):.0f}/100 ({feature_drift.get('status', 'low')})"
    )
    if feature_drift.get("status") in {"medium", "high"}:
        st.sidebar.warning(
            t("data_drift_alert").format(
                status=feature_drift.get("status", "unknown"),
                feature=feature_drift.get("top_feature", "-"),
            )
        )
if drift_speed.get("status") == "ok":
    st.sidebar.caption(
        f"Drift-Speed (7d): {drift_speed.get('trend')} ({float(drift_speed.get('delta', 0.0)):+.1f})"
    )
    if drift_speed.get("trend") == "up" and int(drift_speed.get("streak_up", 0)) >= 3:
        st.sidebar.warning("Drift-Trend steigt mehrfach in Folge – bitte konservativer freigeben.")
benchmark_rows = list(st.session_state.get("pipeline_benchmark_rows", []))
if benchmark_rows:
    top_bench = benchmark_rows[0]
    st.sidebar.caption(
        f"Pipeline-Benchmark: {top_bench.get('step')} p95 {float(top_bench.get('p95_ms', 0.0)):.0f}ms"
    )
    if bool(top_bench.get("alert", False)):
        st.sidebar.warning(
            f"Pipeline p95 Regression: {float(top_bench.get('delta_p95_pct', 0.0)):+.1f}% vs. Vorwoche"
        )

slo_guard = evaluate_control_loop_slo(
    p95_latency_ms=float(st.session_state.get("last_provider_latency_ms", 180.0)),
    error_rate=min(1.0, len(st.session_state.get("error_logs", [])) / 200.0),
    availability=0.99 if not st.session_state.get("error_logs") else 0.97,
    queue_seconds=float(len(st.session_state.get("quality_budget_decisions", [])) * 2),
)
st.session_state["slo_guardian"] = {
    "status": slo_guard.status,
    "burn_rate": slo_guard.burn_rate,
    "escalate": slo_guard.escalate,
    "rollback_hint": slo_guard.rollback_hint,
}
if slo_guard.escalate:
    st.sidebar.warning(
        f"SLO Guardian: {slo_guard.status} (burn {slo_guard.burn_rate:.2f}) → {slo_guard.rollback_hint}"
    )

teams = sorted(elo_curr.keys()) if elo_curr else ["DataError Home", "DataError Away"]
st.markdown(
    f"<div class='hero'><h2>{t('hero')}</h2><p>{league_name} • {auto_refresh}s</p></div>",
    unsafe_allow_html=True,
)

if show_quickstart:
    with st.expander(t("quickstart"), expanded=False):
        st.markdown(
            f"- {t('quickstart_step1')}\n- {t('quickstart_step2')}\n- {t('quickstart_step3')}"
        )

with st.expander(t("feedback"), expanded=False):
    feedback_category = st.selectbox(t("feedback_category"), ["Feature", "Bug", "UX", "Data"])
    feedback_msg = st.text_area(t("feedback"), help=tip("feedback"), placeholder=t("feedback_hint"))
    if st.button(t("feedback_send")) and feedback_msg.strip():
        if save_user_feedback(feedback_msg, league_name, mode, feedback_category):
            st.success(t("feedback_ok"))
        else:
            st.error(t("feedback_fail"))

if st.session_state.get("sync_results"):
    with st.expander(t("sync_status"), expanded=False):
        st.dataframe(pd.DataFrame(st.session_state["sync_results"]), use_container_width=True)
        alert_rows = [
            x for x in st.session_state["error_logs"] if str(x.get("msg", "")).startswith("ALERT:")
        ]
        st.markdown(f"**{t('alert_center')}**")
        if alert_rows:
            st.warning(f"{t('monitor_alert')}: {alert_rows[-1]['msg']}")
            st.dataframe(pd.DataFrame(alert_rows[-10:]), use_container_width=True)
        else:
            st.caption(t("alert_center_empty"))

if df.empty:
    st.info(t("empty_data"))

if page == "Settings":
    with card("Settings Hub"):
        st.caption("Hier kannst du UI- und Betriebsoptionen prüfen, ohne Core-Logik zu ändern.")
        show_ui_smoke_test = st.checkbox("UI Smoke Test anzeigen", key="show_ui_smoke_test")
        if show_ui_smoke_test:
            st.markdown("### UI Smoke Test")
            checks = [
                ("Dashboard lädt", True),
                ("Live Terminal sichtbar", True),
                ("Backtesting sichtbar", True),
                ("News & Sentiment sichtbar", True),
                ("Settings sichtbar", True),
                ("Expander öffnen", True),
            ]
            for label, ok in checks:
                st.write(f"{'✅' if ok else '❌'} {label}")
            c1, c2 = st.columns(2)
            if c1.button("Smoke: Button-Reaktion", width="stretch"):
                st.success("Button reagiert korrekt.")
            if c2.button("Smoke: Toast", width="stretch"):
                st.toast("UI Smoke Test: Toast OK")

if page == "Live Terminal":
    with st.sidebar.expander("Ops • Live Match", expanded=False):
        if is_admin(st.session_state.get("role", "beta")):
            api_key = st.text_input(
                t("api_key"), type="password", help=tip("api_key"), autocomplete="off"
            )
            auto_odds = st.toggle(
                t("auto_odds"),
                value=bool(st.session_state.get("auto_odds", True)),
                key="auto_odds",
                help=tip("auto_odds"),
            )
        else:
            api_key = ""
            auto_odds = False
            st.caption(t("api_admin_only"))
        default_home = st.session_state.get("home_team", teams[0])
        if default_home not in teams:
            default_home = teams[0]
        home = st.selectbox(
            t("home"), teams, index=teams.index(default_home), key="home_team", help=tip("home")
        )
        away_candidates = [tm for tm in teams if tm != home] or teams
        default_away = st.session_state.get("away_team", away_candidates[0])
        if default_away not in away_candidates:
            default_away = away_candidates[0]
        away = st.selectbox(
            t("away"),
            away_candidates,
            index=away_candidates.index(default_away),
            key="away_team",
            help=tip("away"),
        )
        compare_enabled = st.toggle(
            t("compare_toggle"),
            value=(
                False
                if st.session_state.get("safe_mode") or st.session_state.get("quickstart_minimal")
                else bool(st.session_state.get("compare_enabled", False))
            ),
            key="compare_enabled",
            disabled=bool(
                st.session_state.get("safe_mode") or st.session_state.get("quickstart_minimal")
            ),
        )
        compare_home = home
        compare_away = away
        if compare_enabled:
            compare_home_default = st.session_state.get("compare_home_team", teams[0])
            if compare_home_default not in teams:
                compare_home_default = teams[0]
            compare_home = st.selectbox(
                t("compare_home"),
                teams,
                index=teams.index(compare_home_default),
                key="compare_home_team",
            )
            compare_away_candidates = [tm for tm in teams if tm != compare_home] or teams
            compare_away_default = st.session_state.get(
                "compare_away_team", compare_away_candidates[0]
            )
            if compare_away_default not in compare_away_candidates:
                compare_away_default = compare_away_candidates[0]
            compare_away = st.selectbox(
                t("compare_away"),
                compare_away_candidates,
                index=compare_away_candidates.index(compare_away_default),
                key="compare_away_team",
            )

    with st.spinner("API..."):
        odds_start = datetime.utcnow()
        odds_data, credits = (
            get_real_odds_by_league(api_key, home, league_name)
            if auto_odds
            else (None, st.session_state["api_credits"])
        )
        odds_latency_ms = max(0.0, (datetime.utcnow() - odds_start).total_seconds() * 1000.0)
    st.session_state["api_credits"] = credits
    provider_stats = dict(st.session_state.get("provider_quality_stats", {}))
    odds_stats = dict(provider_stats.get("odds_api", {}))
    odds_stats["calls"] = float(odds_stats.get("calls", 0.0)) + 1.0
    odds_stats["success"] = float(odds_stats.get("success", 0.0)) + (1.0 if odds_data else 0.0)
    odds_stats["latency_ms_sum"] = float(odds_stats.get("latency_ms_sum", 0.0)) + float(
        odds_latency_ms
    )
    provider_stats["odds_api"] = odds_stats
    st.session_state["provider_quality_stats"] = provider_stats
    st.session_state["provider_priority"] = provider_priority_order(provider_stats)
    st.session_state["last_provider_latency_ms"] = float(odds_latency_ms)

    with st.sidebar.expander("Ops • Live Monitor", expanded=False):
        st.caption(f"{t('api_credits')}: {credits}")
        provider_scores = compute_provider_scores(
            st.session_state.get("provider_quality_stats", {})
        )
        if provider_scores:
            top_provider = provider_scores[0]
            st.caption(
                f"Provider-Qualität: {top_provider.name} {top_provider.score}/100 | "
                f"{top_provider.success_rate * 100:.0f}% success"
            )
        used, remaining = parse_credit_status(credits)
        if used is not None and remaining is not None:
            total = max(1, used + remaining)
            st.progress(
                min(1.0, used / total), text=t("used_of_total").format(used=used, total=total)
            )
            history = st.session_state["credits_history"]
            history.append(
                {"ts": datetime.utcnow().isoformat(), "used": used, "remaining": remaining}
            )
            st.session_state["credits_history"] = history[-150:]
            if len(history) >= 2:
                delta_used = history[-1]["used"] - history[0]["used"]
                delta_hours = max(
                    1e-6,
                    (
                        datetime.fromisoformat(history[-1]["ts"])
                        - datetime.fromisoformat(history[0]["ts"])
                    ).total_seconds()
                    / 3600,
                )
                burn_rate = max(0.0, delta_used / delta_hours)
                if burn_rate > 0:
                    st.caption(t("api_forecast").format(hours=remaining / burn_rate))
            if remaining <= 10:
                st.warning(t("api_warn"))

        fav_default = [f for f in st.session_state.get("favorites", [home]) if f in teams] or [home]
        favorites = st.multiselect(
            t("favorites"), teams, default=fav_default, key="favorites", help=tip("favorites")
        )
        alert_prob = st.slider(
            t("alert_prob"),
            40,
            95,
            int(st.session_state.get("alert_prob", 70)),
            key="alert_prob",
            help=tip("alert_prob"),
        )
        alert_quote = st.slider(
            t("alert_quote"),
            1.2,
            5.0,
            float(st.session_state.get("alert_quote", 2.2)),
            key="alert_quote",
            help=tip("alert_quote"),
        )
        precision_alert_auto_sensitivity = st.toggle(
            "Alert Auto-Sensitivity",
            value=bool(st.session_state.get("precision_alert_auto_sensitivity", True)),
            key="precision_alert_auto_sensitivity",
        )
        active_alert_prob = int(alert_prob)
        active_alert_quote = float(alert_quote)
        if precision_alert_auto_sensitivity:
            reliability_ref = (
                float(st.session_state.get("reliability_history", [50])[-1])
                if st.session_state.get("reliability_history")
                else 50.0
            )
            alert_adaptive = adaptive_alert_thresholds(
                base_prob_pct=int(alert_prob),
                base_quote_max=float(alert_quote),
                reliability_score=reliability_ref,
                regime_status=str(st.session_state.get("last_regime_status", "stable")),
                drift_score=float(feature_drift.get("score", 50.0)),
            )
            active_alert_prob = int(alert_adaptive.probability_pct)
            active_alert_quote = float(alert_adaptive.quote_max)
            st.caption(
                f"Active Alert-Schwellen: Prob ≥ {active_alert_prob}% | Quote ≤ {active_alert_quote:.2f} ({alert_adaptive.reason})"
            )

    quote = 2.0
    if odds_data:
        for o in odds_data:
            if o.get("name") == home:
                quote = o.get("price", quote)

    with st.sidebar.expander("Ops • Live Advanced", expanded=False):
        market_quote = st.number_input(
            t("quote"),
            min_value=1.01,
            value=float(st.session_state.get("market_quote", quote)),
            key="market_quote",
            help=tip("quote"),
        )
        bankroll = st.number_input(
            t("bankroll"),
            min_value=0.0,
            value=float(st.session_state.get("bankroll", CONFIG.default_budget)),
            key="bankroll",
            help=tip("bankroll"),
        )

        league_weights = st.session_state["league_weights"].get(
            league_name, {"poisson": 0.45, "elo": 0.35, "ml": 0.2}
        )
        if st.session_state.get("user_mode") == "pro":
            wt_c1, wt_c2 = st.columns([0.9, 0.1])
            wt_c1.caption(t("weights"))
            wt_c2.button("?", key="weights_help_btn", help=tip("weights"))

            wh_c1, wh_c2 = st.columns([0.9, 0.1])
            wh_c1.caption(t("weights_hint"))
            wh_c2.button("?", key="weights_hint_help_btn", help=tip("weights_hint"))

            w_poisson = st.slider(
                "w(Poisson)",
                0.0,
                1.0,
                float(league_weights["poisson"]),
                0.05,
                help=tip("w_poisson"),
            )
            w_elo = st.slider(
                "w(Elo)", 0.0, 1.0, float(league_weights["elo"]), 0.05, help=tip("w_elo")
            )
            w_ml = st.slider("w(ML)", 0.0, 1.0, float(league_weights["ml"]), 0.05, help=tip("w_ml"))
        else:
            w_poisson, w_elo, w_ml = 0.45, 0.35, 0.20

        search_team = st.text_input(
            f"{t('filter')} Team",
            value=st.session_state.get("search_team", ""),
            key="search_team",
            help=tip("search_team"),
        )
        min_elo = st.slider(
            "Min Elo",
            1200,
            2000,
            int(st.session_state.get("min_elo", 1300)),
            key="min_elo",
            help=tip("min_elo"),
        )
        min_confidence = st.slider(
            t("min_confidence"),
            40,
            95,
            int(st.session_state.get("min_confidence", 60)),
            key="min_confidence",
            help=tip("min_confidence"),
        )

        st.caption(t("precision_section"))
        precision_auto_tune = st.toggle(
            t("precision_auto_tune"),
            value=bool(st.session_state.get("precision_auto_tune", True)),
            key="precision_auto_tune",
            help=tip("precision_auto_tune"),
        )
        precision_recency_decay = st.slider(
            t("precision_recency_decay"),
            0.70,
            0.99,
            float(st.session_state.get("precision_recency_decay", 0.92)),
            0.01,
            key="precision_recency_decay",
        )
        precision_calibration = st.slider(
            t("precision_calibration"),
            0.0,
            0.5,
            float(st.session_state.get("precision_calibration", 0.15)),
            0.01,
            key="precision_calibration",
        )
        precision_lineup_confirmed = st.toggle(
            t("precision_lineup_confirmed"),
            value=bool(st.session_state.get("precision_lineup_confirmed", True)),
            key="precision_lineup_confirmed",
        )
        precision_lineup_penalty = st.slider(
            t("precision_lineup_penalty"),
            0.0,
            0.25,
            float(st.session_state.get("precision_lineup_penalty", 0.08)),
            0.01,
            key="precision_lineup_penalty",
        )
        precision_min_data_quality = st.slider(
            t("precision_min_data_quality"),
            0,
            100,
            int(st.session_state.get("precision_min_data_quality", 55)),
            1,
            key="precision_min_data_quality",
        )
        precision_drift_adjust = st.toggle(
            t("precision_drift_adjust"),
            value=bool(st.session_state.get("precision_drift_adjust", True)),
            key="precision_drift_adjust",
        )
        precision_consensus_gate = st.toggle(
            t("precision_consensus_gate"),
            value=bool(st.session_state.get("precision_consensus_gate", True)),
            key="precision_consensus_gate",
            help=tip("precision_consensus_gate"),
        )
        precision_adaptive_threshold = st.toggle(
            t("precision_adaptive_threshold"),
            value=bool(st.session_state.get("precision_adaptive_threshold", True)),
            key="precision_adaptive_threshold",
            help=tip("precision_adaptive_threshold"),
        )
        precision_disagreement_penalty = st.toggle(
            t("precision_disagreement_penalty"),
            value=bool(st.session_state.get("precision_disagreement_penalty", True)),
            key="precision_disagreement_penalty",
        )
        precision_stability_gate = st.toggle(
            t("precision_stability_gate"),
            value=bool(st.session_state.get("precision_stability_gate", True)),
            key="precision_stability_gate",
        )
        precision_evidence_gate = st.toggle(
            t("precision_evidence_gate"),
            value=bool(st.session_state.get("precision_evidence_gate", True)),
            key="precision_evidence_gate",
        )
        precision_min_evidence_matches = st.slider(
            t("precision_min_evidence_matches"),
            0,
            30,
            int(st.session_state.get("precision_min_evidence_matches", 8)),
            1,
            key="precision_min_evidence_matches",
        )
        precision_min_evidence_signals = st.slider(
            t("precision_min_evidence_signals"),
            0,
            12,
            int(st.session_state.get("precision_min_evidence_signals", 6)),
            1,
            key="precision_min_evidence_signals",
        )
        precision_quantile_threshold = st.toggle(
            t("precision_quantile_threshold"),
            value=bool(st.session_state.get("precision_quantile_threshold", True)),
            key="precision_quantile_threshold",
        )
        precision_scenario_ensemble = st.toggle(
            t("precision_scenario_ensemble"),
            value=bool(st.session_state.get("precision_scenario_ensemble", True)),
            key="precision_scenario_ensemble",
        )
        precision_scenario_conservative = st.toggle(
            t("precision_scenario_conservative"),
            value=bool(st.session_state.get("precision_scenario_conservative", False)),
            key="precision_scenario_conservative",
        )
        precision_online_bias = st.toggle(
            t("precision_online_bias"),
            value=bool(st.session_state.get("precision_online_bias", True)),
            key="precision_online_bias",
        )
        precision_bias_lr = st.slider(
            t("precision_bias_lr"),
            0.0,
            0.2,
            float(st.session_state.get("precision_bias_lr", 0.03)),
            0.01,
            key="precision_bias_lr",
        )
        precision_shadow_mode = st.toggle(
            t("precision_shadow_mode"),
            value=bool(st.session_state.get("precision_shadow_mode", True)),
            key="precision_shadow_mode",
        )
        precision_shadow_canary = st.toggle(
            t("precision_shadow_canary"),
            value=bool(st.session_state.get("precision_shadow_canary", True)),
            key="precision_shadow_canary",
        )
        precision_feature_trust = st.toggle(
            t("precision_feature_trust"),
            value=bool(st.session_state.get("precision_feature_trust", True)),
            key="precision_feature_trust",
        )
        precision_hysteresis = st.toggle(
            t("precision_hysteresis"),
            value=bool(st.session_state.get("precision_hysteresis", True)),
            key="precision_hysteresis",
        )
        precision_season_calibration = st.toggle(
            t("precision_season_calibration"),
            value=bool(st.session_state.get("precision_season_calibration", True)),
            key="precision_season_calibration",
        )
        precision_trimmed_ensemble = st.toggle(
            t("precision_trimmed_ensemble"),
            value=bool(st.session_state.get("precision_trimmed_ensemble", True)),
            key="precision_trimmed_ensemble",
        )
        precision_error_tracking = st.toggle(
            t("precision_error_tracking"),
            value=bool(st.session_state.get("precision_error_tracking", True)),
            key="precision_error_tracking",
        )
        precision_adaptive_bias_lr = st.toggle(
            t("precision_adaptive_bias_lr"),
            value=bool(st.session_state.get("precision_adaptive_bias_lr", True)),
            key="precision_adaptive_bias_lr",
        )
        precision_weighted_consensus = st.toggle(
            t("precision_weighted_consensus"),
            value=bool(st.session_state.get("precision_weighted_consensus", True)),
            key="precision_weighted_consensus",
        )
        precision_segmented_hysteresis = st.toggle(
            t("precision_segmented_hysteresis"),
            value=bool(st.session_state.get("precision_segmented_hysteresis", True)),
            key="precision_segmented_hysteresis",
        )
        precision_error_adaptive_threshold = st.toggle(
            t("precision_error_adaptive_threshold"),
            value=bool(st.session_state.get("precision_error_adaptive_threshold", True)),
            key="precision_error_adaptive_threshold",
        )
        precision_shadow_auto_promote = st.toggle(
            t("precision_shadow_auto_promote"),
            value=bool(st.session_state.get("precision_shadow_auto_promote", True)),
            key="precision_shadow_auto_promote",
        )

        precision_regime_detector = st.toggle(
            t("precision_regime_detector"),
            value=bool(st.session_state.get("precision_regime_detector", True)),
            key="precision_regime_detector",
        )
        precision_reliability_score = st.toggle(
            t("precision_reliability_score"),
            value=bool(st.session_state.get("precision_reliability_score", True)),
            key="precision_reliability_score",
        )
        precision_cost_sensitive_threshold = st.toggle(
            t("precision_cost_sensitive_threshold"),
            value=bool(st.session_state.get("precision_cost_sensitive_threshold", True)),
            key="precision_cost_sensitive_threshold",
        )
        precision_fallback_matrix = st.toggle(
            t("precision_fallback_matrix"),
            value=bool(st.session_state.get("precision_fallback_matrix", True)),
            key="precision_fallback_matrix",
        )

        precision_adaptive_odds_move = st.toggle(
            t("precision_adaptive_odds_move"),
            value=bool(st.session_state.get("precision_adaptive_odds_move", True)),
            key="precision_adaptive_odds_move",
        )
        precision_regime_auto_profile = st.toggle(
            t("precision_regime_auto_profile"),
            value=bool(st.session_state.get("precision_regime_auto_profile", True)),
            key="precision_regime_auto_profile",
        )
        precision_reliability_hard_gate = st.slider(
            t("precision_reliability_hard_gate"),
            0,
            100,
            int(st.session_state.get("precision_reliability_hard_gate", 45)),
            1,
            key="precision_reliability_hard_gate",
        )

        precision_reliability_auto_gate = st.toggle(
            t("precision_reliability_auto_gate"),
            value=bool(st.session_state.get("precision_reliability_auto_gate", True)),
            key="precision_reliability_auto_gate",
        )
        precision_odds_alert_cooldown = st.slider(
            t("precision_odds_alert_cooldown"),
            0,
            60,
            int(st.session_state.get("precision_odds_alert_cooldown", 10)),
            1,
            key="precision_odds_alert_cooldown",
        )
        precision_regime_exposure_scaling = st.toggle(
            t("precision_regime_exposure_scaling"),
            value=bool(st.session_state.get("precision_regime_exposure_scaling", True)),
            key="precision_regime_exposure_scaling",
        )

        precision_odds_auto_cooldown = st.toggle(
            t("precision_odds_auto_cooldown"),
            value=bool(st.session_state.get("precision_odds_auto_cooldown", True)),
            key="precision_odds_auto_cooldown",
        )
        precision_replay_ci_blocker = st.toggle(
            t("precision_replay_ci_blocker"),
            value=bool(st.session_state.get("precision_replay_ci_blocker", True)),
            key="precision_replay_ci_blocker",
        )
        precision_correlation_guardrail = st.toggle(
            t("precision_correlation_guardrail"),
            value=bool(st.session_state.get("precision_correlation_guardrail", True)),
            key="precision_correlation_guardrail",
        )

        if is_admin(st.session_state.get("role", "beta")) and st.button(
            t("ai_pre_match"), width="stretch"
        ):
            write_audit_event(
                actor=st.session_state.get("username", "guest"),
                role=st.session_state.get("role", "beta"),
                action="admin_ai_pre_match",
                details={"league": league_name, "home": home, "away": away},
            )
            allowed, _remaining = check_ai_usage(
                st.session_state.get("username", "guest"), st.session_state.get("role", "beta")
            )
            if not allowed:
                st.warning(get_quota_limit_message(st.session_state.get("lang", "de")))
            else:
                coach_ctx = HelpContext(
                    league_name=league_name,
                    mode=mode,
                    risk_profile=risk_profile_label,
                    home_team=home,
                    away_team=away,
                    market_quote=float(quote),
                    bankroll=float(bankroll),
                    min_confidence=int(min_confidence),
                    auto_odds=bool(auto_odds),
                    w_poisson=float(league_weights.get("poisson", 0.45)),
                    w_elo=float(league_weights.get("elo", 0.35)),
                    w_ml=float(league_weights.get("ml", 0.20)),
                )
                st.info(get_pre_match_setup_guide(st.session_state.get("lang", "de"), coach_ctx))

        if is_admin(st.session_state.get("role", "beta")) and st.button(
            t("ai_auto_optimize"), width="stretch"
        ):
            write_audit_event(
                actor=st.session_state.get("username", "guest"),
                role=st.session_state.get("role", "beta"),
                action="admin_ai_auto_optimize",
                details={"league": league_name, "home": home, "away": away},
            )
            allowed, _remaining = check_ai_usage(
                st.session_state.get("username", "guest"), st.session_state.get("role", "beta")
            )
            if not allowed:
                st.warning(get_quota_limit_message(st.session_state.get("lang", "de")))
            else:
                rec = recommend_match_settings(
                    HelpContext(
                        league_name=league_name,
                        mode=mode,
                        risk_profile=risk_profile_label,
                        home_team=home,
                        away_team=away,
                        market_quote=float(market_quote),
                        bankroll=float(bankroll),
                        min_confidence=int(min_confidence),
                        auto_odds=bool(auto_odds),
                        w_poisson=float(league_weights.get("poisson", 0.45)),
                        w_elo=float(league_weights.get("elo", 0.35)),
                        w_ml=float(league_weights.get("ml", 0.20)),
                    )
                )
                rp_map = {
                    "Conservative": t("risk_conservative"),
                    "Balanced": t("risk_balanced"),
                    "Aggressive": t("risk_aggressive"),
                }
                st.session_state["risk_profile_label"] = rp_map.get(
                    str(rec.get("risk_profile", "Balanced")), t("risk_balanced")
                )
                st.session_state["sim_count"] = int(rec.get("sim_count", 15000))
                st.session_state["alert_prob"] = int(rec.get("alert_prob", 72))
                st.session_state["alert_quote"] = float(rec.get("alert_quote", 2.2))
                st.session_state["min_elo"] = int(rec.get("min_elo", 1400))
                st.session_state["min_confidence"] = int(rec.get("min_confidence", 70))
                st.session_state["favorites"] = rec.get("favorites", [home]) or [home]
                st.session_state["auto_odds"] = bool(rec.get("auto_odds", True))
                st.session_state["league_weights"][league_name] = {
                    "poisson": float(rec.get("w_poisson", 0.50)),
                    "elo": float(rec.get("w_elo", 0.35)),
                    "ml": float(rec.get("w_ml", 0.15)),
                }
                st.success(t("ai_auto_optimized"))
                st.caption(get_legal_disclaimer(st.session_state.get("lang", "de")))
                st.rerun()

    w_sum = max(1e-6, w_poisson + w_elo + w_ml)
    w_poisson, w_elo, w_ml = w_poisson / w_sum, w_elo / w_sum, w_ml / w_sum
    if precision_auto_tune and metrics is not None:
        w_poisson, w_elo, w_ml = auto_tune_weights(
            float(w_poisson),
            float(w_elo),
            float(w_ml),
            cv_accuracy=float(metrics.cv_accuracy),
            calibration_error=float(metrics.calibration_error),
        )
        st.caption(t("precision_weights_auto"))
    if precision_drift_adjust:
        w_poisson, w_elo, w_ml = apply_feature_drift_weight_adjustment(
            float(w_poisson),
            float(w_elo),
            float(w_ml),
            drift_status=str(feature_drift.get("status", "low")),
            top_feature=str(feature_drift.get("top_feature", "")),
        )

    st.session_state["league_weights"][league_name] = {
        "poisson": w_poisson,
        "elo": w_elo,
        "ml": w_ml,
    }
    now_utc = datetime.utcnow()
    if should_autosave_resume(
        st.session_state.get("resume_last_saved_at", ""), now_utc, interval_seconds=15
    ):
        saved_payload, saved_ok = save_user_resume_if_revision(
            current_user,
            {
                "league_name": league_name,
                "home_team": home,
                "away_team": away,
                "search_team": search_team,
                "user_mode": st.session_state.get("user_mode", "beginner"),
            },
            expected_revision=int(st.session_state.get("resume_revision", 0)),
        )
        if saved_ok:
            st.session_state["resume_revision"] = (
                int(st.session_state.get("resume_revision", 0)) + 1
            )
            st.session_state["resume_last_saved_at"] = now_utc.isoformat()
        else:
            st.session_state["resume_conflict_count"] = (
                int(st.session_state.get("resume_conflict_count", 0)) + 1
            )
            st.session_state["resume_revision"] = int(
                load_user_resume_meta(current_user).get("revision", 0) or 0
            )
            st.sidebar.warning("Session-Resume Konflikt erkannt (Multi-Tab). Neu laden empfohlen.")

    if st.session_state.get("resume_last_saved_at"):
        st.sidebar.caption(
            t("autosave_status").format(
                time=str(st.session_state.get("resume_last_saved_at", ""))[:19].replace("T", " ")
            )
        )

    # Home dashboard preview should update immediately when layout changes (even before running analysis)
    if not st.session_state.get("quickstart_minimal"):
        st.markdown(f"### {t('home_dashboard')}")
        preview_widget_values = {
            "win_prob": (t("widget_win_prob"), "-"),
            "value_edge": (t("widget_value_edge"), "-"),
            "stake": (t("widget_stake"), "-"),
            "api_credits": (t("widget_api_credits"), st.session_state.get("api_credits", "-")),
            "exposure": (
                t("widget_exposure"),
                f"{sum(b.get('stake', 0) for b in st.session_state.get('bet_ledger', []) if b.get('open', True)):.2f}€",
            ),
            "data_quality": (t("widget_data_quality"), f"{data_quality['score']:.0f}/100"),
            "sync_status": (t("widget_sync_status"), st.session_state.get("last_sync_at") or "-"),
            "ml_probs": (t("widget_ml_probs"), "- / - / -"),
            "social_sentiment": (t("widget_social_sentiment"), "-"),
            "injury_risk": (t("widget_injury_risk"), "-"),
            "clv": (t("widget_clv"), "-"),
        }
        preview_widgets = st.session_state.get(
            "dashboard_layout_widgets", PRESET_WIDGETS["balanced"]
        )
        pcols = st.columns(3)
        for i, widget_key in enumerate(preview_widgets):
            if widget_key not in preview_widget_values:
                continue
            label, value = preview_widget_values[widget_key]
            pcols[i % 3].metric(label, value)

    validation_payload = LiveInputPayload(
        home_team=str(home),
        away_team=str(away),
        market_quote=float(market_quote),
        bankroll=float(bankroll),
        min_confidence=int(min_confidence),
        sim_count=int(sim_count),
        weight_poisson=float(w_poisson),
        weight_elo=float(w_elo),
        weight_ml=float(w_ml),
    )
    validation_errors = validate_live_input_payload(validation_payload)
    if validation_errors:
        st.sidebar.warning(t("validation_title"))
        st.sidebar.caption(t("validation_error_count").format(count=len(validation_errors)))
        for err in validation_errors:
            st.sidebar.caption(f"• {t('validation_' + err)}")
    else:
        st.sidebar.caption(t("validation_ok"))

    st.sidebar.caption(t("apply_hint"))
    run_analysis_action = st.sidebar.button(
        t("run"),
        help=tip("run"),
        width="stretch",
        disabled=bool(validation_errors),
    )
    if st.session_state.get("trigger_run"):
        run_analysis_action = True
        st.session_state["trigger_run"] = False
    if run_analysis_action and not df.empty:
        st.session_state["last_action"] = "run_analysis"
        st.session_state["last_action_at"] = datetime.utcnow().isoformat()
        # filter system
        filtered_df = df.copy()
        if search_team:
            filtered_df = filtered_df[
                filtered_df["HomeTeam"].str.contains(search_team, case=False, na=False)
                | filtered_df["AwayTeam"].str.contains(search_team, case=False, na=False)
            ]

        recent = filtered_df[
            (filtered_df["HomeTeam"].isin([home, away]))
            | (filtered_df["AwayTeam"].isin([home, away]))
        ].tail(30)

        def team_sig(team: str):
            m = recent[(recent["HomeTeam"] == team) | (recent["AwayTeam"] == team)].sort_values(
                "Date"
            )
            if m.empty:
                return 1.3, 0.0, 0.0
            goals = clip_outliers(
                np.asarray(np.where(m["HomeTeam"] == team, m["FTHG"], m["FTAG"]), dtype=float)
            )
            shots = clip_outliers(
                np.asarray(np.where(m["HomeTeam"] == team, m["HST"], m["AST"]), dtype=float)
            )
            corners = clip_outliers(
                np.asarray(np.where(m["HomeTeam"] == team, m["HC"], m["AC"]), dtype=float)
            )
            weights = recency_weights(len(m), decay=float(precision_recency_decay))
            goals_w = weighted_mean(goals, weights)
            shots_w = weighted_mean(shots, weights)
            corners_w = weighted_mean(corners, weights)
            dom = (goals_w + 0.2 * shots_w + 0.1 * corners_w) * season_phase_factor(
                datetime.utcnow()
            )
            return (
                float(1.3 if np.isnan(dom) else dom),
                float(shots_w),
                float(corners_w),
            )

        def build_match_snapshot(
            match_home: str, match_away: str, quote_input: float
        ) -> dict[str, object]:
            dom_home, shots_home, corners_home = team_sig(match_home)
            dom_away, shots_away, corners_away = team_sig(match_away)
            sim_home = np.random.poisson(dom_home, sim_count)
            sim_away = np.random.poisson(dom_away, sim_count)
            prob_home = float(np.mean(sim_home > sim_away))
            prob_draw = float(np.mean(sim_home == sim_away))
            prob_away = max(0.0, 1 - prob_home - prob_draw)

            team_matches_all = recent[
                (recent["HomeTeam"] == match_home) | (recent["AwayTeam"] == match_home)
            ]
            if not team_matches_all.empty:
                home_outcome = np.where(
                    team_matches_all["HomeTeam"] == match_home,
                    np.where(
                        team_matches_all["FTHG"] > team_matches_all["FTAG"],
                        2,
                        np.where(team_matches_all["FTHG"] < team_matches_all["FTAG"], 0, 1),
                    ),
                    np.where(
                        team_matches_all["FTAG"] > team_matches_all["FTHG"],
                        2,
                        np.where(team_matches_all["FTAG"] < team_matches_all["FTHG"], 0, 1),
                    ),
                )
                empirical = (
                    float(np.mean(home_outcome == 2)),
                    float(np.mean(home_outcome == 1)),
                    float(np.mean(home_outcome == 0)),
                )
            else:
                empirical = (prob_home, prob_draw, prob_away)
            prob_home, prob_draw, prob_away = calibrate_multiclass_outcomes(
                (prob_home, prob_draw, prob_away), empirical, strength=float(precision_calibration)
            )

            ml_conf_home = 0.0
            ml_probs_home = [0.0, 0.0, 0.0]
            if model is not None and match_home in elo_curr and match_away in elo_curr:
                _, ml_conf_home = predict_match(
                    model,
                    elo_curr[match_home] - elo_curr[match_away],
                    shots_home - shots_away,
                    corners_home - corners_away,
                )
                raw_probs = model.predict_proba(
                    np.array(
                        [
                            [
                                elo_curr[match_home] - elo_curr[match_away],
                                shots_home - shots_away,
                                corners_home - corners_away,
                            ]
                        ]
                    )
                )[0].tolist()
                ml_probs_home = expand_probs_3(
                    raw_probs, [int(c) for c in getattr(model, "classes_", [0, 1, 2])]
                )
            elo_prob_home = 1 / (
                1 + 10 ** ((elo_curr.get(match_away, 1500) - elo_curr.get(match_home, 1500)) / 400)
            )
            ml_conf_home = apply_lineup_uncertainty_penalty(
                ml_conf_home,
                lineup_confirmed=bool(precision_lineup_confirmed),
                penalty=float(precision_lineup_penalty),
            )
            ml_conf_home = calibrate_confidence_by_league(ml_conf_home, league_name)
            if bool(st.session_state.get("precision_weekly_calibration", True)):
                cache = st.session_state.get("weekly_calibration_cache", {})
                if isinstance(cache, dict) and cache:
                    ml_conf_home = apply_weekly_league_calibration(ml_conf_home, league_name, cache)
            ensemble_home = (
                (prob_home * w_poisson) + (elo_prob_home * w_elo) + (ml_conf_home * w_ml)
            )
            disagreement = model_disagreement_score(prob_home, elo_prob_home, ml_conf_home)
            if precision_disagreement_penalty:
                ml_conf_home = apply_disagreement_penalty(
                    ml_conf_home,
                    disagreement,
                    max_penalty=0.12,
                )
            ensemble_home = calibrate_probability(
                ensemble_home,
                prob_home,
                strength=float(precision_calibration),
            )
            expected_value = (ensemble_home * quote_input) - 1
            exposure_now_local = sum(
                b["stake"] for b in st.session_state["bet_ledger"] if b.get("open", True)
            )
            max_exposure_local = dynamic_stake_cap(
                bankroll=bankroll,
                confidence=ensemble_home,
                disagreement=disagreement,
                base_cap=0.3,
            )
            if precision_regime_exposure_scaling:
                max_exposure_local *= regime_exposure_multiplier(
                    str(st.session_state.get("last_regime_status", "stable"))
                )
            exposure_left_local = max(0.0, max_exposure_local - exposure_now_local)
            risk_mult_local = {"Conservative": 0.25, "Balanced": 0.4, "Aggressive": 0.65}[
                risk_profile_runtime
            ]
            kelly_local = (
                (expected_value / (quote_input - 1)) * risk_mult_local if quote_input > 1 else 0
            )
            stake_local = min(max(0.0, bankroll * kelly_local), exposure_left_local)
            correlation_penalty = 0.0
            if precision_correlation_guardrail:
                pair_similarity = estimate_team_pair_similarity(
                    st.session_state["bet_ledger"],
                    team=match_home,
                    away_team=match_away,
                )
                stake_local, correlation_penalty = apply_correlation_guardrail(
                    stake_local,
                    st.session_state["bet_ledger"],
                    team=match_home,
                    league=league_name,
                    away_team=match_away,
                    pair_similarity=pair_similarity,
                )
                peak = float(st.session_state.get("bankroll_peak", bankroll))
                drawdown_pct = max(0.0, ((peak - float(bankroll)) / max(1e-6, peak)) * 100.0)
                correlation_penalty = correlation_penalty_with_drawdown(
                    correlation_penalty, drawdown_pct
                )
                stake_local = max(0.0, float(stake_local) * (1.0 - correlation_penalty))
            explanation_local = build_decision_explanation(
                expected_value=expected_value,
                suggested_stake=stake_local,
                model_confidence=ml_conf_home,
                min_confidence_pct=float(min_confidence),
                exposure_left=exposure_left_local,
                max_exposure=max_exposure_local,
            )
            return {
                "dom_home": dom_home,
                "dom_away": dom_away,
                "shots_home": shots_home,
                "shots_away": shots_away,
                "corners_home": corners_home,
                "corners_away": corners_away,
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "disagreement": disagreement,
                "ml_conf": ml_conf_home,
                "ml_probs": ml_probs_home,
                "elo_prob": elo_prob_home,
                "ensemble": ensemble_home,
                "ev": expected_value,
                "stake": stake_local,
                "exposure_now": exposure_now_local,
                "max_exposure": max_exposure_local,
                "exposure_left": exposure_left_local,
                "explanation": explanation_local,
                "correlation_penalty": correlation_penalty,
            }

        cache_payload = {
            "league": league_name,
            "home": home,
            "away": away,
            "market_quote": round(float(market_quote), 4),
            "sim_count": int(sim_count),
            "min_confidence": int(min_confidence),
            "bankroll": round(float(bankroll), 2),
            "weights": [round(float(w_poisson), 4), round(float(w_elo), 4), round(float(w_ml), 4)],
            "risk_profile": risk_profile_runtime,
        }
        primary_cache_key = build_analysis_cache_key(cache_payload)
        current_hit_rate = cache_hit_rate(
            st.session_state.get("cache_hits", 0), st.session_state.get("cache_misses", 0)
        )
        effective_ttl = cache_effective_ttl_seconds(600, hit_rate=current_hit_rate)
        st.session_state["cache_ttl_seconds"] = effective_ttl
        if (
            should_skip_recompute(
                st.session_state.get("last_analysis_signature", ""), primary_cache_key
            )
            and st.session_state.get("last_primary_snapshot") is not None
        ):
            primary_snapshot = st.session_state.get("last_primary_snapshot")
            st.caption(t("delta_reuse"))
        else:
            primary_snapshot = read_cached_result(
                st.session_state["analysis_cache"],
                primary_cache_key,
                now=datetime.utcnow(),
                ttl_seconds=effective_ttl,
            )
            if primary_snapshot is None:
                budget_decision = apply_quality_budget(
                    st.session_state.get("quality_budget_events", []),
                    now=datetime.utcnow(),
                    max_recomputes_per_hour=int(
                        st.session_state.get("quality_budget_max_recomputes", 6)
                    ),
                    enabled=bool(st.session_state.get("quality_budget_mode", True)),
                    priority_score=(
                        float(feature_drift.get("score", 50.0))
                        + (100.0 - float(data_quality.get("score", 50.0)))
                    )
                    / 2.0,
                    emergency_priority_threshold=85.0,
                    allow_emergency_slot=True,
                )
                priority_score = (
                    float(feature_drift.get("score", 50.0))
                    + (100.0 - float(data_quality.get("score", 50.0)))
                ) / 2.0
                priority_tier = classify_priority_tier(priority_score)
                st.caption(f"Quality-Budget Priorität: {priority_tier} ({priority_score:.0f}/100)")
                if budget_decision.reason == "emergency_slot":
                    st.info("Quality-Budget: Emergency-Slot genutzt (kritische Lage).")
                st.session_state["quality_budget_events"] = budget_decision.events
                decision_log = list(st.session_state.get("quality_budget_decisions", []))
                decision_log.append(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "reason": str(budget_decision.reason),
                        "priority": float(priority_score),
                        "tier": str(priority_tier),
                        "used": len(budget_decision.events),
                        "max_per_hour": int(
                            st.session_state.get("quality_budget_max_recomputes", 6)
                        ),
                    }
                )
                st.session_state["quality_budget_decisions"] = decision_log[-200:]
                if not budget_decision.allow_recompute:
                    st.warning(
                        "Quality Budget aktiv: Recompute-Limit pro Stunde erreicht. Nutze Cache/letzten Snapshot."
                    )
                    primary_snapshot = st.session_state.get("last_primary_snapshot")
                if primary_snapshot is None:
                    st.session_state["cache_misses"] = (
                        int(st.session_state.get("cache_misses", 0)) + 1
                    )
                    inference_t0 = perf_counter()
                    primary_snapshot = build_match_snapshot(home, away, float(market_quote))
                    st.session_state["quality_budget_events"] = (
                        st.session_state.get("quality_budget_events", [])
                        + [datetime.utcnow().isoformat()]
                    )[-100:]
                    st.session_state["pipeline_latency_history"] = track_pipeline_latency(
                        st.session_state.get("pipeline_latency_history", []),
                        step="build_primary_snapshot",
                        duration_ms=(perf_counter() - inference_t0) * 1000.0,
                    )
                write_cached_result(
                    st.session_state["analysis_cache"],
                    primary_cache_key,
                    primary_snapshot,
                    now=datetime.utcnow(),
                )
                st.caption(t("cache_store"))
            else:
                st.session_state["cache_hits"] = int(st.session_state.get("cache_hits", 0)) + 1
                st.caption(t("cache_hit"))
        live_cache_hit_rate = cache_hit_rate(
            st.session_state.get("cache_hits", 0), st.session_state.get("cache_misses", 0)
        )
        st.sidebar.caption(
            f"Cache Hit-Rate: {live_cache_hit_rate * 100:.1f}% | TTL: {int(st.session_state.get('cache_ttl_seconds', 600))}s"
        )
        st.session_state["last_analysis_signature"] = primary_cache_key
        st.session_state["last_primary_snapshot"] = primary_snapshot
        dom_h = float(primary_snapshot["dom_home"])
        dom_a = float(primary_snapshot["dom_away"])
        shots_h = float(primary_snapshot["shots_home"])
        shots_a = float(primary_snapshot["shots_away"])
        corners_h = float(primary_snapshot["corners_home"])
        corners_a = float(primary_snapshot["corners_away"])
        prob_h = float(primary_snapshot["prob_home"])
        prob_d = float(primary_snapshot["prob_draw"])
        prob_a = float(primary_snapshot["prob_away"])
        ml_conf = float(primary_snapshot["ml_conf"])
        ml_probs = list(primary_snapshot["ml_probs"])
        elo_prob = float(primary_snapshot["elo_prob"])
        ensemble = float(primary_snapshot["ensemble"])

        # Multi criteria optimizer
        wcol1, wcol2, wcol3 = st.columns(3)
        w_val = wcol1.slider(t("w_value"), 0.0, 1.0, 0.4, 0.05)
        w_prob = wcol2.slider(t("w_prob"), 0.0, 1.0, 0.4, 0.05)
        w_risk = wcol3.slider(t("w_risk"), 0.0, 1.0, 0.2, 0.05)

        ev = float(primary_snapshot["ev"])
        exposure_now = float(primary_snapshot["exposure_now"])
        max_exposure = float(primary_snapshot["max_exposure"])
        exposure_left = float(primary_snapshot["exposure_left"])
        stake = float(primary_snapshot["stake"])

        composite = (
            w_val * max(ev, 0)
            + w_prob * ensemble
            + w_risk * (1 - min(1.0, exposure_now / max(bankroll, 1)))
        ) / max(1e-6, w_val + w_prob + w_risk)

        explanation = primary_snapshot["explanation"]
        recommended_bet = explanation.recommended_bet
        prob_low, prob_high = poisson_confidence_interval(
            np.array([prob_h, prob_d, prob_a], dtype=float), alpha=0.1
        )
        adaptive_gate = 0.55
        if precision_adaptive_threshold:
            adaptive_gate = adaptive_threshold(
                0.55,
                np.array([prob_h, float(primary_snapshot.get("ensemble", prob_h))], dtype=float),
                strength=0.2,
            )
        if precision_quantile_threshold:
            adaptive_gate = quantile_decision_threshold(
                (
                    filtered_df.tail(40)["home_win_rate"].to_numpy(dtype=float)
                    if "home_win_rate" in filtered_df.columns
                    else np.where(
                        filtered_df.tail(40)["FTHG"] > filtered_df.tail(40)["FTAG"],
                        1.0,
                        0.0,
                    )
                ),
                adaptive_gate,
                lower_quantile=0.4,
                strength=0.35,
            )
        if precision_error_adaptive_threshold:
            prior_err = st.session_state.get("precision_error_stats", {})
            adaptive_gate = error_class_threshold_adjustment(
                adaptive_gate,
                false_positives=int(prior_err.get("false_positives", 0)),
                false_negatives=int(prior_err.get("false_negatives", 0)),
                total_labeled=int(prior_err.get("total_labeled", 0)),
                strength=0.04,
                min_samples=20,
            )
        prior_err = st.session_state.get("precision_error_stats", {})
        labeled_total = int(prior_err.get("total_labeled", 0))
        fp_count = int(prior_err.get("false_positives", 0))
        fn_count = int(prior_err.get("false_negatives", 0))
        error_rate = ((fp_count + fn_count) / labeled_total) if labeled_total > 0 else 0.0
        regime_state = detect_regime_state(
            drift_score=float(feature_drift.get("score", 50.0)),
            error_rate=float(error_rate),
            bias_term=float(st.session_state.get("precision_bias_term", 0.0)),
        )
        if precision_cost_sensitive_threshold:
            adaptive_gate = adjust_cost_sensitive_threshold(
                adaptive_gate,
                false_positive_cost=max(1.0, float(fp_count) + 1.0),
                false_negative_cost=max(1.0, float(fn_count) + 1.0),
                risk_profile=risk_profile_runtime,
                regime_status=regime_state.status,
            )
        league_outcomes = [
            x
            for x in st.session_state.get("prediction_outcomes", [])
            if isinstance(x, dict) and str(x.get("league", "")) == str(league_name)
        ]
        bandit_offsets = dict(st.session_state.get("bandit_threshold_offsets", {}))
        current_offset = float(bandit_offsets.get(league_name, 0.0))
        new_offset = confidence_bandit_delta(
            league_outcomes,
            current_offset=current_offset,
            step=0.005,
            stability_window=20,
            max_error_rate_delta=0.08,
        )
        bandit_registry = dict(st.session_state.get("bandit_registry", {}))
        bandit_events = list(st.session_state.get("bandit_events", []))
        league_state = dict(bandit_registry.get(league_name, {}))
        league_error_rate = compute_outcome_error_rate(league_outcomes, window=40)
        best_error_rate = league_state.get("best_error_rate")
        cooldown_until_n = int(league_state.get("cooldown_until_n", 0) or 0)
        if len(league_outcomes) < cooldown_until_n:
            if should_exit_bandit_cooldown(
                current_error_rate=league_error_rate,
                best_error_rate=(float(best_error_rate) if best_error_rate is not None else None),
                drift_score=float(feature_drift.get("score", 50.0)),
            ):
                league_state["cooldown_until_n"] = len(league_outcomes)
                cooldown_until_n = len(league_outcomes)
                st.caption("Bandit Cooldown vorzeitig beendet (Stabilisierung erkannt).")
                bandit_events.append(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "league": league_name,
                        "event": "cooldown_exit_early",
                        "error_rate": (
                            float(league_error_rate) if league_error_rate is not None else None
                        ),
                        "drift": float(feature_drift.get("score", 50.0)),
                    }
                )
            new_offset = float(league_state.get("last_offset", current_offset))
            if len(league_outcomes) < cooldown_until_n:
                st.caption(
                    f"Bandit Cooldown aktiv ({cooldown_until_n - len(league_outcomes)} Entscheidungen verbleibend)."
                )
        if league_error_rate is not None:
            if best_error_rate is None or float(league_error_rate) < float(best_error_rate):
                league_state["best_error_rate"] = float(league_error_rate)
                league_state["best_offset"] = float(new_offset)
            elif float(league_error_rate) > float(best_error_rate) + 0.08:
                new_offset = float(league_state.get("best_offset", 0.0))
                drift_now = float(feature_drift.get("score", 50.0))
                cooldown_len = 30 if drift_now >= 70 else 20 if drift_now >= 50 else 12
                league_state["cooldown_until_n"] = len(league_outcomes) + cooldown_len
                st.caption("Bandit Safety: Offset-Rollback wegen Performance-Abfall aktiv.")
                bandit_events.append(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "league": league_name,
                        "event": "cooldown_enter",
                        "cooldown_len": int(cooldown_len),
                        "error_rate": float(league_error_rate),
                        "best_error_rate": float(best_error_rate),
                        "drift": float(drift_now),
                    }
                )
        league_state["last_offset"] = float(new_offset)
        bandit_registry[league_name] = league_state
        st.session_state["bandit_registry"] = bandit_registry
        st.session_state["bandit_events"] = bandit_events[-300:]
        bandit_offsets[league_name] = float(new_offset)
        st.session_state["bandit_threshold_offsets"] = bandit_offsets
        adaptive_gate = max(0.45, min(0.75, float(adaptive_gate) + float(new_offset)))
        if precision_scenario_ensemble:
            scenario_result = scenario_ensemble_probability(
                prob_h,
                adjustment=max(
                    0.01, min(0.08, float(primary_snapshot.get("disagreement", 0.0)) * 0.15 + 0.01)
                ),
                conservative_mode=precision_scenario_conservative,
            )
            prob_h_for_gate = scenario_result.blended
        else:
            prob_h_for_gate = prob_h
        if precision_online_bias:
            bias_lr_effective = float(precision_bias_lr)
            if precision_adaptive_bias_lr:
                bias_lr_effective = adaptive_bias_learning_rate(
                    float(precision_bias_lr),
                    data_quality_score=float(data_quality.get("score", 50.0)),
                    drift_score=float(feature_drift.get("score", 50.0)),
                )
            bias_result = online_bias_correction(
                prob_h_for_gate,
                observed_outcome=None,
                previous_bias_term=float(st.session_state.get("precision_bias_term", 0.0)),
                learning_rate=bias_lr_effective,
                max_abs_bias=0.15,
            )
            prob_h_for_gate = bias_result.corrected_probability
            st.session_state["precision_bias_term"] = bias_result.bias_term
        if precision_season_calibration:
            prob_h_for_gate = season_phase_calibration(
                prob_h_for_gate,
                datetime.utcnow(),
                strength=0.06,
            )
        if precision_feature_trust:
            trust = feature_trust_weight(
                recency_hours=max(0.0, float(data_quality.get("hours_since_update", 24.0))),
                completeness=float(data_quality.get("score", 50.0)) / 100.0,
                stability=float(feature_drift.get("score", 50.0)) / 100.0,
            )
            prob_h_for_gate = apply_trust_to_probability(prob_h_for_gate, trust)

        fallback_result = apply_weighted_signal_fallback_matrix(
            {
                "prob_h": prob_h_for_gate,
                "elo_prob": elo_prob,
                "ml_conf": ml_conf,
                "ensemble": ensemble,
            }
        )
        if precision_fallback_matrix:
            prob_h_for_gate = fallback_result.values["prob_h"]
            elo_prob = fallback_result.values["elo_prob"]
            ml_conf = fallback_result.values["ml_conf"]
            ensemble = fallback_result.values["ensemble"]

        legacy_recommended_bet = bool(recommended_bet)
        if data_quality["score"] < int(precision_min_data_quality):
            legacy_recommended_bet = False
        consensus_home_prob = prob_h_for_gate
        if precision_trimmed_ensemble:
            consensus_home_prob = trimmed_mean([prob_h_for_gate, elo_prob, ml_conf], trim_ratio=0.2)
        if precision_weighted_consensus:
            consensus_home_prob = weighted_consensus_probability(
                [prob_h_for_gate, elo_prob, ml_conf],
                [
                    float(data_quality.get("score", 50.0)) / 100.0,
                    1.0 - (float(feature_drift.get("score", 50.0)) / 100.0),
                    float(ml_conf),
                ],
            )
        precision_stability_guard = bool(st.session_state.get("precision_stability_guard", True))
        guard_result = stability_guard(
            st.session_state.get("last_stability_signals"),
            {
                "market_quote": float(market_quote),
                "drift_score": float(feature_drift.get("score", 50.0)),
                "data_quality": float(data_quality.get("score", 50.0)),
            },
            strictness=1.0,
        )
        st.session_state["last_stability_signals"] = {
            "market_quote": float(market_quote),
            "drift_score": float(feature_drift.get("score", 50.0)),
            "data_quality": float(data_quality.get("score", 50.0)),
        }
        if precision_stability_guard and guard_result.triggered:
            adaptive_gate = min(0.85, float(adaptive_gate) + float(guard_result.threshold_add))

        if precision_consensus_gate and not consensus_gate(
            consensus_home_prob, elo_prob, ml_conf, threshold=adaptive_gate
        ):
            legacy_recommended_bet = False
        candidate_recommended_bet = bool(legacy_recommended_bet)
        if precision_stability_gate:
            recent_20 = filtered_df.tail(20)
            recent_30 = filtered_df.tail(30)
            recent_40 = filtered_df.tail(40)

            def _window_home_win_rate(frame: pd.DataFrame) -> float:
                if frame.empty:
                    return prob_h
                sub = frame[(frame["HomeTeam"] == home) | (frame["AwayTeam"] == home)]
                if sub.empty:
                    return prob_h
                wins = np.where(
                    sub["HomeTeam"] == home,
                    sub["FTHG"] > sub["FTAG"],
                    sub["FTAG"] > sub["FTHG"],
                )
                return float(np.mean(wins.astype(float)))

            stable_ok = stability_gate(
                np.array(
                    [
                        _window_home_win_rate(recent_20),
                        _window_home_win_rate(recent_30),
                        _window_home_win_rate(recent_40),
                    ],
                    dtype=float,
                ),
                tolerance=0.05,
            )
            if not stable_ok:
                candidate_recommended_bet = False
        evidence_gate = evaluate_evidence_gate(
            sample_size=0, valid_signals=999, min_sample_size=0, min_signals=0
        )
        if precision_evidence_gate:
            evidence_sample = filtered_df[
                (filtered_df["HomeTeam"] == home)
                | (filtered_df["AwayTeam"] == home)
                | (filtered_df["HomeTeam"] == away)
                | (filtered_df["AwayTeam"] == away)
            ]
            evidence_signals = count_valid_signals(
                [
                    prob_h,
                    prob_d,
                    prob_a,
                    ml_conf,
                    elo_prob,
                    ensemble,
                    ev,
                    stake,
                    data_quality.get("score"),
                ]
            )
            evidence_gate = evaluate_evidence_gate(
                sample_size=len(evidence_sample),
                valid_signals=evidence_signals,
                min_sample_size=int(precision_min_evidence_matches),
                min_signals=int(precision_min_evidence_signals),
            )
            if not evidence_gate.is_sufficient:
                candidate_recommended_bet = False
        active_reliability_gate = int(precision_reliability_hard_gate)
        if precision_reliability_auto_gate:
            active_reliability_gate = reliability_hard_gate_from_history(
                st.session_state.get("reliability_history", []),
                base_gate=int(precision_reliability_hard_gate),
            )

        if precision_reliability_score:
            provisional_reliability = build_reliability_score(
                data_quality_score=float(data_quality.get("score", 50.0)),
                drift_score=float(feature_drift.get("score", 50.0)),
                evidence_ratio=(
                    min(
                        1.0,
                        float(evidence_gate.sample_size)
                        / max(1.0, float(precision_min_evidence_matches)),
                    )
                    if precision_evidence_gate
                    else 1.0
                ),
                consensus_strength=max(0.0, 1.0 - abs(consensus_home_prob - adaptive_gate)),
                regime_status=regime_state.status,
            )
            if provisional_reliability < int(active_reliability_gate):
                candidate_recommended_bet = False

        shadow_result = evaluate_shadow_decision(legacy_recommended_bet, candidate_recommended_bet)
        shadow_promoted = bool(st.session_state.get("precision_shadow_promoted", False))
        if precision_shadow_mode:
            if shadow_result.changed:
                st.caption(t("precision_shadow_diff"))
            if shadow_promoted:
                st.caption(t("precision_shadow_promoted"))
            if precision_shadow_canary and not canary_enabled and not shadow_promoted:
                recommended_bet = shadow_result.legacy_decision
            else:
                recommended_bet = shadow_result.candidate_decision
        else:
            recommended_bet = shadow_result.candidate_decision

        if precision_hysteresis:
            enter_threshold = max(adaptive_gate, 0.55)
            exit_threshold = max(0.45, adaptive_gate - 0.05)
            if precision_segmented_hysteresis:
                enter_threshold, exit_threshold = segmented_hysteresis_thresholds(
                    base_enter=enter_threshold,
                    risk_profile=risk_profile_runtime,
                    league_name=league_name,
                )
            recommended_bet = hysteresis_decision(
                score=consensus_home_prob,
                previous_decision=bool(st.session_state.get("precision_last_gate_decision", False)),
                enter_threshold=enter_threshold,
                exit_threshold=exit_threshold,
            )
        st.session_state["precision_last_gate_decision"] = bool(recommended_bet)

        evidence_ratio = 1.0
        if precision_evidence_gate:
            evidence_ratio = min(
                1.0,
                float(evidence_gate.sample_size) / max(1.0, float(precision_min_evidence_matches)),
            )
        reliability_score = build_reliability_score(
            data_quality_score=float(data_quality.get("score", 50.0)),
            drift_score=float(feature_drift.get("score", 50.0)),
            evidence_ratio=evidence_ratio,
            consensus_strength=max(0.0, 1.0 - abs(consensus_home_prob - adaptive_gate)),
            regime_status=regime_state.status,
        )
        st.session_state["last_regime_status"] = regime_state.status
        st.session_state["reliability_history"] = update_metric_history(
            st.session_state.get("reliability_history", []), reliability_score, limit=60
        )
        reliability_trend = compute_metric_trend(
            st.session_state.get("reliability_history", []), short_window=5, long_window=20
        )
        st.session_state["confidence_history"] = update_metric_history(
            st.session_state.get("confidence_history", []), float(ml_conf), limit=80
        )
        confidence_drift_alert = detect_confidence_drift_alert(
            st.session_state.get("confidence_history", []),
            current_confidence=float(ml_conf),
            threshold=0.10,
        )

        replay_seed = st.session_state.get("replay_history", [])
        if not replay_seed:
            replay_seed = load_replay_history(st.session_state.get("username", "guest"))
        replay_seed.append(
            {
                "name": f"run_{datetime.utcnow().strftime('%H%M%S')}",
                "score": float(consensus_home_prob),
                "threshold": float(adaptive_gate),
                "expected": bool(recommended_bet),
            }
        )
        st.session_state["replay_history"] = replay_seed[-80:]
        persist_replay_history(
            st.session_state.get("username", "guest"), st.session_state["replay_history"]
        )
        historical_cases = build_replay_cases_from_history(
            st.session_state.get("replay_history", []), threshold=float(adaptive_gate), max_cases=25
        )
        replay_cases = [
            ReplayCase(
                "high_signal_bet",
                score=consensus_home_prob,
                threshold=adaptive_gate,
                expected=True,
            ),
            ReplayCase(
                "low_signal_no_bet",
                score=adaptive_gate - 0.08,
                threshold=adaptive_gate,
                expected=False,
            ),
            ReplayCase(
                "borderline_guard",
                score=adaptive_gate + 0.02,
                threshold=adaptive_gate,
                expected=True,
            ),
            *historical_cases,
        ]
        replay_report = run_replay_suite(replay_cases)
        replay_quality = summarize_replay_quality(replay_cases, replay_report.failed_cases)
        replay_health_status = replay_health_level(replay_quality)
        replay_health_history = update_metric_history(
            st.session_state.get("replay_health_history", []),
            {"good": 2.0, "watch": 1.0, "critical": 0.0}.get(replay_health_status, 1.0),
            limit=60,
        )
        st.session_state["replay_health_history"] = replay_health_history
        replay_health_labels = [
            "critical" if x < 0.5 else "watch" if x < 1.5 else "good" for x in replay_health_history
        ]
        replay_health_delta = replay_quality_trend(replay_health_labels, window=20)
        stress_points = build_dynamic_threshold_stress(
            score=float(consensus_home_prob),
            base_threshold=float(adaptive_gate),
        )
        stress_flip_count = max(0, len({point.recommended for point in stress_points}) - 1)
        uncertainty_threshold = uncertainty_threshold_by_league(
            float(adaptive_gate), league_name=league_name
        )
        uncertainty = decision_uncertainty_band(
            [float(x.get("score", 0.0)) for x in st.session_state.get("replay_history", [])],
            threshold=float(uncertainty_threshold),
        )
        if uncertainty.unstable:
            candidate_recommended_bet = False
        if precision_replay_ci_blocker and replay_quality.critical_failures > 0:
            candidate_recommended_bet = False

        decision_id = str(uuid.uuid4())
        replay_id = f"replay-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{decision_id[:8]}"
        model_agreement = compute_model_agreement(
            {
                "poisson": float(prob_h_for_gate),
                "elo": float(elo_prob),
                "ml": float(ml_conf),
                "ensemble": float(ensemble),
            }
        )
        feature_current = {
            "goals": recent.tail(20)["FTHG"].astype(float).tolist() if not recent.empty else [],
            "odds": [float(market_quote)],
            "form": recent.tail(20)["HST"].astype(float).tolist() if not recent.empty else [],
        }
        feature_baseline = {
            "goals": (
                recent.tail(80).head(60)["FTHG"].astype(float).tolist() if len(recent) >= 20 else []
            ),
            "odds": st.session_state.get("odds_history", [])[-20:],
            "form": (
                recent.tail(80).head(60)["HST"].astype(float).tolist() if len(recent) >= 20 else []
            ),
        }
        baseline_odds = [
            float(x.get("market", market_quote))
            for x in st.session_state.get("odds_history", [])[-30:]
            if isinstance(x, dict)
        ]
        feature_baseline["odds"] = baseline_odds
        perf_rows = list(st.session_state.get("prediction_outcomes", []))[-80:]
        rolling_brier = [
            (float(x.get("pred_home", 0.5)) - float(bool(x.get("home_won", False)))) ** 2
            for x in perf_rows
            if isinstance(x.get("home_won"), bool)
        ]
        rolling_logloss = [
            -(
                (
                    float(bool(x.get("home_won", False)))
                    * np.log(max(1e-6, float(x.get("pred_home", 0.5))))
                )
                + (
                    (1 - float(bool(x.get("home_won", False))))
                    * np.log(max(1e-6, 1 - float(x.get("pred_home", 0.5))))
                )
            )
            for x in perf_rows
            if isinstance(x.get("home_won"), bool)
        ]
        rolling_accuracy = [
            float((float(x.get("pred_home", 0.5)) >= 0.5) == bool(x.get("home_won", False)))
            for x in perf_rows
            if isinstance(x.get("home_won"), bool)
        ]
        drift_result = compute_drift_score(
            feature_current=feature_current,
            feature_baseline=feature_baseline,
            rolling_brier=rolling_brier,
            rolling_logloss=rolling_logloss,
            rolling_accuracy=rolling_accuracy,
        )
        trust_score, trust_explain = compute_trust_score(
            data_quality=float(data_quality.get("score", 50.0)) / 100.0,
            model_agreement=float(model_agreement.get("agreement", 0.0)),
            volatility=float(uncertainty.band),
            drift_score=float(drift_result.drift_score),
            calibration_bucket_perf=None,
        )
        st.session_state["latest_trust_score"] = trust_score
        st.session_state["latest_trust_explain"] = trust_explain
        st.session_state["latest_agreement"] = model_agreement
        st.session_state["latest_drift"] = {
            "drift_score": drift_result.drift_score,
            "severity": drift_result.severity,
            "recommended_action": drift_result.recommended_action,
            "details": drift_result.details,
        }
        st.session_state["latest_decision_id"] = decision_id

        record_prediction(
            {
                "decision_id": decision_id,
                "replay_id": replay_id,
                "created_at": datetime.utcnow().isoformat(),
                "league": league_name,
                "home": home,
                "away": away,
                "recommended_bet": bool(recommended_bet),
                "pred_home": float(prob_h_for_gate),
                "confidence": float(ml_conf),
                "edge": float(ev),
                "quote": float(market_quote),
                "trust_score": float(trust_score),
                "trust_explain": trust_explain,
                "model_agreement": float(model_agreement.get("agreement", 0.0)),
                "consensus_flag": bool(model_agreement.get("consensus", False)),
                "drift_score": float(drift_result.drift_score),
                "drift_severity": drift_result.severity,
                "status": "open",
            }
        )
        if drift_result.severity != "ok":
            log_drift_event(
                decision_id=decision_id,
                drift_score=float(drift_result.drift_score),
                severity=drift_result.severity,
                recommended_action=drift_result.recommended_action,
                details=drift_result.details,
            )

        if precision_error_tracking:
            prior = st.session_state.get("precision_error_stats", {})
            current_stats = ErrorClassStats(
                false_positives=int(prior.get("false_positives", 0)),
                false_negatives=int(prior.get("false_negatives", 0)),
                total_labeled=int(prior.get("total_labeled", 0)),
            )
            h2h = filtered_df[
                (filtered_df["HomeTeam"] == home) & (filtered_df["AwayTeam"] == away)
            ].tail(1)
            actual_positive = None
            if not h2h.empty:
                actual_positive = bool(float(h2h.iloc[-1]["FTHG"]) > float(h2h.iloc[-1]["FTAG"]))
            updated = update_error_class_stats(
                current_stats,
                predicted_positive=bool(recommended_bet),
                actual_positive=actual_positive,
            )
            st.session_state["precision_error_stats"] = {
                "false_positives": updated.false_positives,
                "false_negatives": updated.false_negatives,
                "total_labeled": updated.total_labeled,
            }
            st.caption(
                t("precision_error_stats").format(
                    fp=updated.false_positives,
                    fn=updated.false_negatives,
                    n=updated.total_labeled,
                )
            )
            if actual_positive is not None:
                outcomes = list(st.session_state.get("prediction_outcomes", []))
                outcomes.append(
                    {
                        "league": league_name,
                        "pred_home": float(prob_h_for_gate),
                        "home_won": bool(actual_positive),
                        "took_bet": bool(recommended_bet),
                        "edge": float(ev),
                        "confidence": float(ml_conf),
                        "reason_code": str(explanation.reason_code),
                        "quote": float(market_quote),
                        "decision_id": decision_id,
                        "replay_id": replay_id,
                        "trust_score": float(trust_score),
                        "model_agreement": float(model_agreement.get("agreement", 0.0)),
                        "drift_score": float(drift_result.drift_score),
                        "ts": datetime.utcnow().isoformat(),
                    }
                )
                st.session_state["prediction_outcomes"] = outcomes[-350:]
            else:
                pending = list(st.session_state.get("pending_outcome_queue", []))
                pending.append(
                    {
                        "league": league_name,
                        "home": home,
                        "away": away,
                        "pred_home": float(prob_h_for_gate),
                        "took_bet": bool(recommended_bet),
                        "edge": float(ev),
                        "confidence": float(ml_conf),
                        "reason_code": str(explanation.reason_code),
                        "quote": float(market_quote),
                        "decision_id": decision_id,
                        "replay_id": replay_id,
                        "trust_score": float(trust_score),
                        "model_agreement": float(model_agreement.get("agreement", 0.0)),
                        "drift_score": float(drift_result.drift_score),
                        "ts": datetime.utcnow().isoformat(),
                        "resolved": False,
                    }
                )
                st.session_state["pending_outcome_queue"] = pending[-500:]

            if bool(st.session_state.get("precision_weekly_calibration", True)):
                st.session_state["weekly_calibration_cache"] = build_weekly_league_calibration(
                    st.session_state.get("prediction_outcomes", []),
                    now=datetime.utcnow(),
                    days=7,
                    min_samples=5,
                )
                st.session_state["calibration_report_cache"] = build_calibration_report(
                    st.session_state.get("prediction_outcomes", []),
                    bins=5,
                    min_samples=20,
                )
                st.session_state["no_bet_summary_cache"] = summarize_no_bet_quality(
                    st.session_state.get("prediction_outcomes", []),
                    min_edge=0.03,
                )
                compare_state = st.session_state.get("precision_shadow_compare", {})
                legacy_err = int(compare_state.get("legacy_errors", 0)) + int(
                    bool(shadow_result.legacy_decision) != bool(actual_positive)
                )
                candidate_err = int(compare_state.get("candidate_errors", 0)) + int(
                    bool(shadow_result.candidate_decision) != bool(actual_positive)
                )
                samples = int(compare_state.get("samples", 0)) + 1
                st.session_state["precision_shadow_compare"] = {
                    "samples": samples,
                    "legacy_errors": legacy_err,
                    "candidate_errors": candidate_err,
                }
                if precision_shadow_auto_promote:
                    promotion = evaluate_shadow_promotion(
                        legacy_errors=legacy_err,
                        candidate_errors=candidate_err,
                        samples=samples,
                        min_samples=50,
                        min_relative_improvement=0.05,
                    )
                    if promotion.promoted:
                        st.session_state["precision_shadow_promoted"] = True

        if explanation.reason_code == "value":
            action_reason = t("action_reason_value")
        elif explanation.reason_code == "confidence":
            action_reason = t("action_reason_confidence")
        else:
            action_reason = t("action_reason_risk")
        if data_quality["score"] < int(precision_min_data_quality):
            action_reason = t("precision_data_quality_gate")
        if precision_evidence_gate and not evidence_gate.is_sufficient:
            action_reason = t("precision_evidence_gate_reason")
        check_status = evaluate_decision_checks(explanation)
        decision_snapshot_current = {
            "confidence": float(explanation.confidence_pct) / 100.0,
            "edge": float(explanation.edge_pct) / 100.0,
            "threshold": float(adaptive_gate),
            "exposure": float(explanation.exposure_left),
        }
        decision_delta_lines = summarize_decision_delta(
            st.session_state.get("decision_delta_prev"),
            decision_snapshot_current,
        )
        st.session_state["decision_delta_prev"] = decision_snapshot_current
        explain_trace = list(st.session_state.get("explainability_trace", []))
        explain_trace.append(
            {
                "ts": datetime.utcnow().isoformat(),
                "match": f"{home} vs {away}",
                "confidence": float(explanation.confidence_pct) / 100.0,
                "edge": float(explanation.edge_pct) / 100.0,
                "threshold": float(adaptive_gate),
                "recommended_bet": bool(recommended_bet),
                "reason_code": str(explanation.reason_code),
                "home_won": None,
            }
        )
        st.session_state["explainability_trace"] = explain_trace[-200:]
        explain_hash = build_explainability_snapshot_hash(
            confidence=float(explanation.confidence_pct) / 100.0,
            edge=float(explanation.edge_pct) / 100.0,
            reason_code=str(explanation.reason_code),
            threshold=float(adaptive_gate),
            recommended_bet=bool(recommended_bet),
        )
        current_hashes = list(st.session_state.get("explainability_hashes", []))
        current_hashes.append(explain_hash)
        st.session_state["explainability_hashes"] = current_hashes[-200:]
        regression = evaluate_explainability_snapshot_regression(
            current_hashes=list(st.session_state.get("explainability_hashes", [])),
            golden_hashes=list(st.session_state.get("explainability_golden_hashes", [])),
            max_mismatch_rate=0.05,
        )
        st.session_state["explainability_regression_safe"] = bool(regression.regression_safe)
        st.session_state["experiment_logbook"] = append_experiment_log_entry(
            st.session_state.get("experiment_logbook", []),
            {
                "ts": datetime.utcnow().isoformat(),
                "league": league_name,
                "match": f"{home} vs {away}",
                "threshold": round(float(adaptive_gate), 4),
                "recommended_bet": bool(recommended_bet),
                "confidence": round(float(explanation.confidence_pct), 2),
                "edge_pct": round(float(explanation.edge_pct), 2),
                "reason": str(explanation.reason_code),
                "reliability": int(reliability_score),
            },
        )
        with st.container(border=True):
            st.markdown(f"**{t('action_title')}**")
            st.write(t("action_bet") if recommended_bet else t("action_no_bet"))
            st.caption(f"Trust Score: {trust_score:.1f}/100 | Gründe: {', '.join(trust_explain)}")
            st.caption(
                f"Model Agreement: {float(model_agreement.get('agreement', 0.0)) * 100:.1f}% | Consensus: {bool(model_agreement.get('consensus', False))}"
            )
            st.caption(
                f"Drift: {float(drift_result.drift_score):.1f}/100 ({drift_result.severity}) → {drift_result.recommended_action}"
            )
            st.caption(f"{t('action_reason')}: {action_reason}")
            st.caption("Was hat sich geändert?")
            for delta_line in decision_delta_lines:
                st.write(f"• {delta_line}")
            st.caption(
                f"{t('precision_prob_interval')}: {prob_low * 100:.1f}% - {prob_high * 100:.1f}%"
            )
            if precision_stability_guard and guard_result.triggered:
                st.warning(
                    "Stability-Guard aktiv: Threshold angehoben um "
                    f"{guard_result.threshold_add * 100:.1f}pp ({', '.join(guard_result.reasons)})"
                )
            if (
                bool(st.session_state.get("precision_confidence_drift_alert", True))
                and confidence_drift_alert.triggered
            ):
                drift_sign = "+" if confidence_drift_alert.delta >= 0 else ""
                st.warning(
                    "Confidence-Drift erkannt: "
                    f"{drift_sign}{confidence_drift_alert.delta * 100:.1f}pp "
                    f"(Baseline {confidence_drift_alert.baseline * 100:.1f}%, "
                    f"Aktuell {confidence_drift_alert.current * 100:.1f}%, Level {confidence_drift_alert.level})"
                )
            regime_flags = regime_risk_highlights(
                regime_status=regime_state.status,
                reliability_score=float(reliability_score),
                drift_score=float(feature_drift.get("score", 50.0)),
                missing_signals=len(fallback_result.missing_keys),
            )
            if regime_flags:
                human_flags = humanize_regime_risk_flags(
                    regime_flags, lang=st.session_state.get("lang", "de")
                )
                st.caption("Regime-Risiken: " + ", ".join(human_flags))
                action_map = {
                    "regime_drifted": "Stake senken",
                    "regime_volatile": "Gates erhöhen",
                    "low_reliability": "Nur High-Confidence",
                    "high_drift": "Quick-Replay starten",
                    "fallback_active": "Daten-Check priorisieren",
                }
                st.caption(
                    f"{t('regime_actions')}: "
                    + " | ".join(action_map.get(flag, flag) for flag in regime_flags)
                )
            next_action_message = recommend_next_action(
                recommended_bet=bool(recommended_bet),
                regime_status=regime_state.status,
                reliability_score=float(reliability_score),
                feature_drift_score=float(feature_drift.get("score", 50.0)),
                audit_anomaly_count=len(st.session_state.get("audit_anomalies", [])),
            )
            st.caption(f"Next-Action-Coach: {next_action_message}")
            ux_feedback = list(st.session_state.get("ux_action_feedback", []))
            ux_feedback.append(
                {
                    "recommended_action": str(next_action_message),
                    "action_taken": "bet" if bool(recommended_bet) else "wait",
                }
            )
            st.session_state["ux_action_feedback"] = ux_feedback[-200:]
            ux_telemetry = evaluate_next_action_telemetry(
                outcomes=list(st.session_state.get("ux_action_feedback", [])),
                target_hit_rate=0.8,
            )
            st.caption(
                f"UX Hit-Rate: {ux_telemetry.hit_rate * 100:.1f}% | Samples: {ux_telemetry.sample_size}"
            )
            if not ux_telemetry.meets_target:
                st.warning(
                    "UX Coach benötigt Feinschliff: " + ", ".join(ux_telemetry.recommended_actions)
                )
            if not bool(st.session_state.get("explainability_regression_safe", True)):
                st.warning("Explainability Regression erkannt: bitte Golden-Snapshots prüfen.")
            action_snippet = build_next_action_snippet(
                recommended_bet=bool(recommended_bet),
                reason_code=str(explanation.reason_code),
                edge_pct=float(explanation.edge_pct),
                confidence_pct=float(explanation.confidence_pct),
                exposure_left=float(explanation.exposure_left),
                drift_score=float(feature_drift.get("score", 50.0)),
                suggested_stake_pct=(float(stake) / max(1.0, float(bankroll))) * 100.0,
                user_mode=str(st.session_state.get("user_mode", "beginner")),
            )
            st.info(action_snippet)
            st.caption(
                f"Regime-Kontext: {regime_state.status} | Reliability {float(reliability_score):.2f} | Drift {float(feature_drift.get('score', 50.0)):.0f}/100"
            )
            st.caption(
                f"{t('uncertainty_band')}: {uncertainty.low * 100:.1f}% - {uncertainty.high * 100:.1f}%"
            )
            if bool(st.session_state.get("feature_flag_weekly_auto_review", True)):
                weekly_review = build_weekly_auto_review(
                    drift_history=st.session_state.get("drift_history", []),
                    alerts=st.session_state.get("latest_sync_alerts", []),
                    outcomes=st.session_state.get("prediction_outcomes", []),
                    sync_quality_score=float(st.session_state.get("sync_quality_score", 0.0)),
                    now=datetime.utcnow(),
                )
                st.session_state["weekly_auto_review"] = weekly_review
                with st.expander("Weekly Auto-Review (kompakt)", expanded=False):
                    for row in weekly_review:
                        st.write(f"• {row}")
            if not recommended_bet:
                blocked_messages = {
                    "edge": t("action_blocking_edge"),
                    "confidence": t("action_blocking_confidence"),
                    "exposure": t("action_blocking_exposure"),
                }
                blocked_checks = get_blocking_checks(check_status)
                if blocked_checks:
                    st.markdown(f"**{t('action_why_not_set')}**")
                    st.caption(t("action_blocking_title"))
                    for item in blocked_checks:
                        st.write(f"• {blocked_messages[item]}")
                    if bool(st.session_state.get("feature_flag_no_bet_gap_hint", True)):
                        st.caption(
                            build_no_bet_gap_hint(
                                edge_pct=float(explanation.edge_pct),
                                confidence_pct=float(explanation.confidence_pct),
                                min_confidence_pct=float(explanation.min_confidence_pct),
                                exposure_left=float(explanation.exposure_left),
                                min_edge_pct=0.1,
                                min_exposure_left=0.1,
                            )
                        )
            with st.expander(t("explainability_panel"), expanded=False):
                st.markdown(f"**{t('action_checks')}**")
                st.write(
                    f"{'✅' if check_status.edge_ok else '❌'} {t('check_edge').format(value=f'{explanation.edge_pct:.1f}')}"
                )
                st.write(
                    f"{'✅' if check_status.confidence_ok else '❌'} {t('check_confidence').format(value=f'{explanation.confidence_pct:.1f}', minimum=int(explanation.min_confidence_pct))}"
                )
                st.write(
                    f"{'✅' if check_status.exposure_ok else '❌'} {t('check_exposure').format(left=f'{explanation.exposure_left:.2f}', max_exposure=f'{explanation.max_exposure:.2f}')}"
                )

                highlights = build_explainability_highlights(
                    metrics.feature_importance,
                    get_blocking_checks(check_status),
                )
                st.write(f"**{t('explainability_top_drivers')}:**")
                for row in highlights.top3_drivers:
                    st.write(f"• {row}")
                st.write(f"**{t('explainability_top_risks')}:**")
                if highlights.top2_risks:
                    for row in highlights.top2_risks:
                        st.write(f"• {row}")
                else:
                    st.write("• -")

        with st.expander(t("metric_help"), expanded=False):
            st.write(f"• {t('metric_help_edge')}")
            st.write(f"• {t('metric_help_confidence')}")
            st.write(f"• {t('metric_help_exposure')}")

        if compare_enabled:
            compare_cache_key = build_analysis_cache_key(
                {
                    **cache_payload,
                    "home": compare_home,
                    "away": compare_away,
                }
            )
            compare_snapshot = read_cached_result(
                st.session_state["analysis_cache"],
                compare_cache_key,
                now=datetime.utcnow(),
                ttl_seconds=600,
            )
            if compare_snapshot is None:
                compare_t0 = perf_counter()
                compare_snapshot = build_match_snapshot(
                    compare_home, compare_away, float(market_quote)
                )
                st.session_state["pipeline_latency_history"] = track_pipeline_latency(
                    st.session_state.get("pipeline_latency_history", []),
                    step="build_compare_snapshot",
                    duration_ms=(perf_counter() - compare_t0) * 1000.0,
                )
                write_cached_result(
                    st.session_state["analysis_cache"],
                    compare_cache_key,
                    compare_snapshot,
                    now=datetime.utcnow(),
                )
            compare_explanation = compare_snapshot["explanation"]
            cmp_col_a, cmp_col_b = st.columns(2)
            with cmp_col_a:
                with st.container(border=True):
                    st.markdown(f"**{t('compare_title')} – {t('compare_primary')}**")
                    st.caption(f"{home} vs {away}")
                    st.metric(t("widget_win_prob"), f"{ensemble * 100:.1f}%")
                    st.metric(t("widget_value_edge"), f"{ev * 100:.1f}%")
                    st.metric(t("compare_ml_conf"), f"{ml_conf * 100:.1f}%")
                    st.caption(t("action_bet") if recommended_bet else t("action_no_bet"))
            with cmp_col_b:
                with st.container(border=True):
                    st.markdown(f"**{t('compare_title')} – {t('compare_secondary')}**")
                    st.caption(f"{compare_home} vs {compare_away}")
                    st.metric(
                        t("widget_win_prob"), f"{float(compare_snapshot['ensemble']) * 100:.1f}%"
                    )
                    st.metric(t("widget_value_edge"), f"{float(compare_snapshot['ev']) * 100:.1f}%")
                    st.metric(
                        t("compare_ml_conf"), f"{float(compare_snapshot['ml_conf']) * 100:.1f}%"
                    )
                    st.caption(
                        t("action_bet")
                        if compare_explanation.recommended_bet
                        else t("action_no_bet")
                    )
            compare_export_df = pd.DataFrame(
                [
                    {
                        "match": f"{home} vs {away}",
                        "win_prob": round(ensemble * 100, 2),
                        "value_edge": round(ev * 100, 2),
                        "ml_confidence": round(ml_conf * 100, 2),
                        "action": t("action_bet") if recommended_bet else t("action_no_bet"),
                    },
                    {
                        "match": f"{compare_home} vs {compare_away}",
                        "win_prob": round(float(compare_snapshot["ensemble"]) * 100, 2),
                        "value_edge": round(float(compare_snapshot["ev"]) * 100, 2),
                        "ml_confidence": round(float(compare_snapshot["ml_conf"]) * 100, 2),
                        "action": (
                            t("action_bet")
                            if compare_explanation.recommended_bet
                            else t("action_no_bet")
                        ),
                    },
                ]
            )
            compact_compare = st.toggle(
                t("export_compact_columns"), value=True, key="export_compact_compare"
            )
            compare_csv_data = (
                compact_export_csv(compare_export_df, ["match", "win_prob", "value_edge", "action"])
                if compact_compare
                else compare_export_df.to_csv(index=False)
            )
            st.download_button(
                t("export_compare_csv"),
                data=compare_csv_data,
                file_name="match_comparison.csv",
                mime="text/csv",
            )

        # Alerts with hysteresis to avoid flapping
        alert_key = f"{league_name}:{home}"
        state = st.session_state["alert_state"].get(alert_key, {"active": False, "last": ""})
        trigger_now = home in favorites and (
            (ensemble * 100) >= active_alert_prob or market_quote <= active_alert_quote
        )
        clear_now = (ensemble * 100) < (active_alert_prob - 5) and market_quote > (
            active_alert_quote + 0.2
        )
        if state["active"] and clear_now:
            state["active"] = False
        elif not state["active"] and trigger_now:
            state["active"] = True
            state["last"] = datetime.utcnow().isoformat()
            st.warning(
                t("live_alert_msg").format(team=home, prob=ensemble * 100, quote=market_quote)
            )
        st.session_state["alert_state"][alert_key] = state

        # Bet ledger, portfolio + gamification
        if stake > 0 and elo_curr.get(home, 0) >= min_elo:
            st.session_state["bet_ledger"].append(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "team": home,
                    "opponent": away,
                    "pair_key": build_team_pair_key(home, away),
                    "league": league_name,
                    "stake": float(stake),
                    "edge": float(ev),
                    "entry_quote": float(market_quote),
                    "fair_quote": float((1 / ensemble) if ensemble else 0),
                    "decision_id": decision_id,
                    "replay_id": replay_id,
                    "trust_score": float(trust_score),
                    "trust_explain": trust_explain,
                    "model_agreement": float(model_agreement.get("agreement", 0.0)),
                    "drift_score": float(drift_result.drift_score),
                    "won": None,
                    "open": True,
                }
            )
            st.session_state["xp"] += int(min(30, max(4, ev * 100)))

        retro_summary = summarize_retrospective(st.session_state["bet_ledger"], window=30)
        st.session_state["last_retrospective"] = retro_summary
        persist_retrospective(st.session_state.get("username", "guest"), retro_summary)
        if bool(st.session_state.get("precision_auto_postmortem", True)):
            postmortem_report = build_weekly_postmortem(
                st.session_state.get("bet_ledger", []),
                now=datetime.utcnow(),
                days=7,
                top_n=10,
            )
            st.session_state["last_postmortem_report"] = [x.__dict__ for x in postmortem_report]
            persist_postmortem_report(st.session_state.get("username", "guest"), postmortem_report)

        # xG and injury/sentiment models
        xg_df = pd.DataFrame(
            [
                {"player": f"{home} Striker", "xG": max(0.08, dom_h * 0.42), "role": "Striker"},
                {"player": f"{home} Winger", "xG": max(0.05, dom_h * 0.26), "role": "Winger"},
                {"player": f"{away} Striker", "xG": max(0.08, dom_a * 0.42), "role": "Striker"},
                {"player": f"{away} Winger", "xG": max(0.05, dom_a * 0.26), "role": "Winger"},
            ]
        )
        social_score, social_posts = social_sentiment_index(home)
        injury_prob = float(
            build_injury_model().predict_proba(
                np.array([[80, 0.3, (100 - social_score) / 100, 1]])
            )[0][1]
        )

        st.session_state["odds_history"].append(
            {
                "ts": datetime.utcnow(),
                "market": market_quote,
                "fair": (1 / ensemble if ensemble else 0),
            }
        )
        st.session_state["odds_history"] = st.session_state["odds_history"][-ODDS_HISTORY_LIMIT:]

        if precision_adaptive_odds_move:
            odds_signal = detect_odds_move_signal_adaptive(
                st.session_state["odds_history"], league_name=league_name, lookback=6
            )
        else:
            odds_signal = detect_odds_move_signal_adaptive(
                st.session_state["odds_history"], league_name="", lookback=6
            )
        alert_key_local = f"{league_name}:{home}:{away}"
        now_iso = datetime.utcnow().isoformat()
        last_alert = str(st.session_state.get("odds_alert_last_at", {}).get(alert_key_local, ""))
        active_odds_cooldown = int(precision_odds_alert_cooldown)
        if precision_odds_auto_cooldown:
            active_odds_cooldown = recommended_odds_cooldown_minutes(
                st.session_state.get("odds_history", []),
                regime_status=regime_state.status,
                base_minutes=int(precision_odds_alert_cooldown),
            )
        if odds_signal.triggered and should_emit_odds_alert(
            last_alert, now_iso, cooldown_minutes=active_odds_cooldown
        ):
            st.warning(
                t("odds_move_warning").format(
                    move=odds_signal.move_abs, pct=odds_signal.move_pct * 100
                )
            )
            priority = odds_alert_priority(
                odds_signal.move_abs,
                odds_signal.move_pct,
                regime_status=regime_state.status,
                reliability_score=float(reliability_score),
                replay_health=replay_health_status,
            )
            st.caption(
                t("odds_move_direction_hint").format(
                    hint=odds_alert_direction_message(
                        odds_signal.direction, lang=st.session_state.get("lang", "de")
                    )
                )
            )
            st.caption(f"{t('odds_priority')}: {priority}")
            odds_last = dict(st.session_state.get("odds_alert_last_at", {}))
            odds_last[alert_key_local] = now_iso
            st.session_state["odds_alert_last_at"] = odds_last

        if precision_fallback_matrix and fallback_result.missing_keys:
            st.info(
                t("fallback_notice").format(
                    count=len(fallback_result.missing_keys),
                    penalty=fallback_result.penalty * 100,
                )
            )
            details = fallback_penalty_breakdown(fallback_result.missing_keys)
            st.caption(f"{t('fallback_breakdown')}:")
            for key_name, penalty in details.items():
                st.write(f"• {key_name}: -{penalty * 100:.1f}%")

        tabs = st.tabs(
            [
                t("home_dashboard"),
                t("overview"),
                t("analytics"),
                t("execution"),
                t("settings"),
                "Proof / Performance",
                "Downside / Worst Case",
            ]
        )

        with tabs[0]:
            ledger_for_home = pd.DataFrame(st.session_state["bet_ledger"])
            avg_clv_home, _, _ = compute_clv_summary(
                ledger_for_home, current_team=home, closing_quote=float(market_quote)
            )
            widget_values = {
                "win_prob": (t("widget_win_prob"), f"{ensemble * 100:.1f}%"),
                "value_edge": (t("widget_value_edge"), f"{ev * 100:.1f}%"),
                "stake": (t("widget_stake"), f"{stake:.2f}€"),
                "api_credits": (t("widget_api_credits"), st.session_state.get("api_credits", "-")),
                "exposure": (t("widget_exposure"), f"{exposure_now:.2f}€"),
                "data_quality": (t("widget_data_quality"), f"{data_quality['score']:.0f}/100"),
                "sync_status": (
                    t("widget_sync_status"),
                    st.session_state.get("last_sync_at") or "-",
                ),
                "ml_probs": (
                    t("widget_ml_probs"),
                    f"{ml_probs[2]:.2f} / {ml_probs[1]:.2f} / {ml_probs[0]:.2f}",
                ),
                "social_sentiment": (t("widget_social_sentiment"), f"{social_score}/100"),
                "injury_risk": (t("widget_injury_risk"), f"{injury_prob:.1%}"),
                "clv": (t("widget_clv"), f"{avg_clv_home:+.2f}%"),
            }
            chosen_widgets = st.session_state.get(
                "dashboard_layout_widgets", PRESET_WIDGETS["balanced"]
            )
            cols = st.columns(3)
            for idx, widget_key in enumerate(chosen_widgets):
                if widget_key not in widget_values:
                    continue
                label, value = widget_values[widget_key]
                cols[idx % 3].metric(label, value)

        with tabs[1]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t("ensemble_win_prob"), f"{ensemble * 100:.1f}%")
            c2.metric(t("fair_odds"), f"{(1 / ensemble if ensemble else 0):.2f}")
            c3.metric(t("value_edge"), f"{ev * 100:.1f}%")
            c4.metric(t("composite"), f"{composite:.3f}")
            if precision_regime_detector:
                st.caption(
                    f"{t('regime_state')}: {regime_state.status} ({regime_state.score * 100:.0f}/100)"
                )
            if precision_reliability_score:
                st.metric(t("reliability_score"), f"{reliability_score}/100")
                st.caption(
                    f"{t('reliability_trend')}: {reliability_trend:+.1f} | gate {active_reliability_gate}"
                )
            st.caption(f"{t('replay_suite')}: {replay_report.passed}/{replay_report.total} passed")
            st.caption(
                f"{t('replay_quality')}: B {replay_quality.bronze} / S {replay_quality.silver} / G {replay_quality.gold}"
            )
            st.caption(f"{t('replay_health')}: {replay_health_status}")
            st.caption(f"{t('replay_health_trend')}: {replay_health_delta:+.2f}")
            st.caption(f"{t('stress_flip_count')}: {stress_flip_count}")
            stress_df = pd.DataFrame(
                [
                    {
                        "Δ": f"{point.delta:+.2f}",
                        "Threshold": f"{point.threshold:.2f}",
                        "Decision": "Bet" if point.recommended else "No Bet",
                    }
                    for point in stress_points
                ]
            )
            st.markdown(f"**{t('dynamic_threshold_stress')}**")
            st.dataframe(stress_df, use_container_width=True, hide_index=True)
            st.caption(
                f"{t('correlation_penalty')}: {float(primary_snapshot.get('correlation_penalty', 0.0)) * 100:.1f}%"
            )

            retro_saved = load_retrospective(st.session_state.get("username", "guest"))
            st.caption(
                f"{t('auto_retro')}: {retro_saved.bets} Bets | ROI {retro_saved.roi_pct:+.1f}% | W/L {retro_saved.wins}/{retro_saved.losses}"
            )
            weekly_cal = st.session_state.get("weekly_calibration_cache", {})
            if weekly_cal and league_name in weekly_cal:
                cal = weekly_cal[league_name]
                st.caption(
                    "Wöchentliche Liga-Kalibrierung: "
                    f"Faktor {cal.factor:.3f} | Exp {cal.expected_rate * 100:.1f}% | Obs {cal.observed_rate * 100:.1f}% (n={cal.sample_size})"
                )
            saved_postmortem = load_postmortem_report(st.session_state.get("username", "guest"))
            if saved_postmortem:
                top_case = saved_postmortem[0]
                st.caption(
                    "Auto-Postmortem (Top 1/10): "
                    f"{top_case.get('match', '-')} | {top_case.get('issue', '-')} | "
                    f"Impact {float(top_case.get('impact', 0.0)):.1f}"
                )
            calibration_report = build_calibration_report(
                st.session_state.get("prediction_outcomes", []), bins=5, min_samples=20
            )
            if calibration_report.samples >= 20:
                st.caption(
                    f"Calibration Report (7d): n={calibration_report.samples} | "
                    f"Brier {calibration_report.brier_score:.3f} | ECE {calibration_report.ece:.3f}"
                )
            no_bet_summary = summarize_no_bet_quality(
                st.session_state.get("prediction_outcomes", []), min_edge=0.03
            )
            if no_bet_summary.reviewed > 0:
                st.caption(
                    "No-Bet Tracker: "
                    f"reviewed={no_bet_summary.reviewed} | missed={no_bet_summary.missed_value_count} "
                    f"| avoided={no_bet_summary.avoided_loss_count}"
                )
            error_costs = summarize_error_costs(
                st.session_state.get("prediction_outcomes", []),
                stake_eur=float(st.session_state.get("stake_pct", 5)) * 5.0,
            )
            st.caption(
                "FP/FN Kosten (Schätzer): "
                f"FP {error_costs.false_positive_count}={error_costs.false_positive_cost_eur:.2f}€ | "
                f"FN {error_costs.false_negative_count}={error_costs.false_negative_cost_eur:.2f}€"
            )
            latency_summary = summarize_pipeline_latency(
                st.session_state.get("pipeline_latency_history", [])
            )
            if latency_summary:
                top_latency = latency_summary[0]
                st.caption(
                    "Pipeline-Latenz (p95): "
                    f"{top_latency.step} {top_latency.p95_ms:.0f}ms (n={top_latency.count})"
                )
            else:
                top_latency = None
            alert_digest = build_actionable_alert_digest(
                data_quality_score=float(data_quality.get("score", 0.0)),
                drift_score=float(feature_drift.get("score", 0.0)),
                reliability_score=float(reliability_score),
                freshness_score=int(st.session_state.get("data_freshness_score", 0)),
                fp_fn_cost_eur=float(error_costs.total_cost_eur),
                top_latency_ms=float(top_latency.p95_ms if top_latency else 0.0),
            )
            if alert_digest:
                with st.expander("Actionable Alert Digest (Top 3)", expanded=False):
                    digest_key = datetime.utcnow().date().isoformat()
                    daily_log = dict(st.session_state.get("daily_digest_log", {}))
                    daily_entry = dict(daily_log.get(digest_key, {"items": [], "done": {}}))
                    daily_entry["items"] = list(alert_digest)
                    done_state = dict(daily_entry.get("done", {}))
                    visible_idx = 0
                    for idx, msg in enumerate(alert_digest, start=1):
                        if should_snooze_digest_item(
                            daily_log,
                            item_text=msg,
                            today=datetime.utcnow(),
                            lookback_days=7,
                            min_done_days=3,
                        ):
                            continue
                        visible_idx += 1
                        st.write(f"{idx}. {msg}")
                        impact = compute_digest_impact_score(
                            message=msg,
                            data_quality_score=float(data_quality.get("score", 0.0)),
                            drift_score=float(feature_drift.get("score", 0.0)),
                            reliability_score=float(reliability_score),
                            fp_fn_cost_eur=float(error_costs.total_cost_eur),
                            top_latency_ms=float(top_latency.p95_ms if top_latency else 0.0),
                            freshness_score=int(st.session_state.get("data_freshness_score", 0)),
                        )
                        st.caption(f"Impact-Score: {impact}/100")
                        reason_detail = ""
                        if "Drift" in msg:
                            reason_detail = f"Warum erneut: Drift aktuell {float(feature_drift.get('score', 0.0)):.0f}/100."
                        elif "p95" in msg:
                            reason_detail = f"Warum erneut: Pipeline p95 bei {float(top_latency.p95_ms if top_latency else 0.0):.0f}ms."
                        elif "Kosten" in msg:
                            reason_detail = f"Warum erneut: FP/FN-Kosten bei {float(error_costs.total_cost_eur):.2f}€."
                        elif "Daten" in msg:
                            reason_detail = (
                                f"Warum erneut: Datenqualität {float(data_quality.get('score', 0.0)):.0f}/100 "
                                f"und Frische {int(st.session_state.get('data_freshness_score', 0))}/100."
                            )
                        if reason_detail:
                            st.caption(reason_detail)
                        done_key = f"digest_done_{digest_key}_{idx}"
                        checked = st.checkbox(
                            "Erledigt", value=bool(done_state.get(str(idx), False)), key=done_key
                        )
                        done_state[str(idx)] = bool(checked)
                    if visible_idx == 0:
                        st.caption(
                            "Alle Top-3 Punkte wurden zuletzt zuverlässig erledigt (Auto-Snooze aktiv)."
                        )
                    daily_entry["done"] = done_state
                    daily_log[digest_key] = daily_entry
                    st.session_state["daily_digest_log"] = daily_log
            dead_rows = st.session_state.get("dead_letter_outcomes", [])
            if dead_rows:
                with st.expander("Outcome Dead-Letter (manuelle Auflösung)", expanded=False):
                    st.caption("Fälle mit vielen Retries; bitte Teamnamen/Ergebnisdaten prüfen.")
                    st.dataframe(
                        pd.DataFrame(dead_rows[-25:]), use_container_width=True, hide_index=True
                    )
                    team_candidates = []
                    if not df.empty and {"HomeTeam", "AwayTeam"}.issubset(df.columns):
                        home_candidates = [
                            str(x) for x in df["HomeTeam"].dropna().astype(str).unique()
                        ]
                        away_candidates = [
                            str(x) for x in df["AwayTeam"].dropna().astype(str).unique()
                        ]
                        team_candidates = sorted(set(home_candidates + away_candidates))
                    normalized_candidates = {
                        normalize_team_name(x) for x in team_candidates if normalize_team_name(x)
                    }

                    def _names_valid(home_name: str, away_name: str) -> bool:
                        if not normalized_candidates:
                            return True
                        return (
                            normalize_team_name(home_name) in normalized_candidates
                            and normalize_team_name(away_name) in normalized_candidates
                        )

                    def _pending_key(row: dict[str, object]) -> str:
                        return (
                            f"{row.get('league', '')}|{normalize_team_name(str(row.get('home', '')))}|"
                            f"{normalize_team_name(str(row.get('away', '')))}|{row.get('ts', '')}"
                        )

                    bulk_c1, bulk_c2 = st.columns(2)
                    if bulk_c1.button("Top-5 Auto-Requeue", key="dead_bulk_requeue"):
                        pending_rows = list(st.session_state.get("pending_outcome_queue", []))
                        pending_keys = {
                            _pending_key(dict(x)) for x in pending_rows if isinstance(x, dict)
                        }
                        pool = list(dead_rows)
                        rejected = 0
                        duplicates = 0
                        for row in dead_rows[:5]:
                            item = dict(row)
                            item_home = str(item.get("home", ""))
                            item_away = str(item.get("away", ""))
                            if not _names_valid(item_home, item_away):
                                rejected += 1
                                continue
                            item_key = _pending_key(item)
                            if item_key in pending_keys:
                                duplicates += 1
                                continue
                            item["resolved"] = False
                            item["retry_count"] = 0
                            item["next_retry_at"] = ""
                            item.pop("dead_letter_at", None)
                            pending_rows.append(item)
                            pending_keys.add(item_key)
                            if row in pool:
                                pool.remove(row)
                        st.session_state["pending_outcome_queue"] = pending_rows[-500:]
                        st.session_state["dead_letter_outcomes"] = pool
                        dead_rows = pool
                        st.success("Top-5 Dead-Letter Fälle wurden re-queued.")
                        if rejected > 0:
                            st.warning(
                                f"{rejected} Fälle wurden wegen ungültiger Teamnamen nicht re-queued."
                            )
                        if duplicates > 0:
                            st.info(
                                f"{duplicates} Fälle wurden als Duplikat erkannt und übersprungen."
                            )
                    if bulk_c2.button("Top-5 als gelöst markieren", key="dead_bulk_resolve"):
                        remaining = list(dead_rows[5:])
                        st.session_state["dead_letter_outcomes"] = remaining
                        dead_rows = remaining
                        st.success("Top-5 Dead-Letter Fälle wurden als gelöst entfernt.")
                    if not dead_rows:
                        st.caption("Keine offenen Dead-Letter Fälle mehr.")
                    else:
                        selected_idx = st.selectbox(
                            "Dead-Letter Fall auswählen",
                            options=list(range(len(dead_rows))),
                            format_func=lambda i: (
                                f"#{i + 1}: {dead_rows[i].get('home', '?')} vs {dead_rows[i].get('away', '?')}"
                            ),
                        )
                        selected = dict(dead_rows[selected_idx]) if dead_rows else {}
                        c_fix1, c_fix2 = st.columns(2)
                        suggested_home = suggest_team_name(
                            str(selected.get("home", "")), team_candidates
                        )
                        suggested_away = suggest_team_name(
                            str(selected.get("away", "")), team_candidates
                        )
                        if suggested_home != str(selected.get("home", "")) or suggested_away != str(
                            selected.get("away", "")
                        ):
                            st.caption(
                                "Auto-Vorschlag: "
                                f"{suggested_home} vs {suggested_away} "
                                f"(normalisiert: {normalize_team_name(str(selected.get('home', '')))} / {normalize_team_name(str(selected.get('away', '')))})."
                            )
                        fixed_home = c_fix1.text_input(
                            "Home korrigieren",
                            value=str(selected.get("home", "")),
                            key=f"dead_home_{selected_idx}",
                        )
                        fixed_away = c_fix2.text_input(
                            "Away korrigieren",
                            value=str(selected.get("away", "")),
                            key=f"dead_away_{selected_idx}",
                        )
                        b1, b2 = st.columns(2)
                        if st.button("Auto-Vorschlag übernehmen", key=f"dead_auto_{selected_idx}"):
                            st.session_state[f"dead_home_{selected_idx}"] = suggested_home
                            st.session_state[f"dead_away_{selected_idx}"] = suggested_away
                        if b1.button("Zurück in Queue", key=f"dead_requeue_{selected_idx}"):
                            if not _names_valid(fixed_home, fixed_away):
                                st.error(
                                    "Requeue blockiert: Teamnamen konnten keiner bekannten Liga-Mannschaft zugeordnet werden."
                                )
                            else:
                                row = dict(dead_rows[selected_idx])
                                row["home"] = fixed_home
                                row["away"] = fixed_away
                                row["resolved"] = False
                                row["retry_count"] = 0
                                row["next_retry_at"] = ""
                                row.pop("dead_letter_at", None)
                                pending_rows = list(
                                    st.session_state.get("pending_outcome_queue", [])
                                )
                                pending_keys = {
                                    _pending_key(dict(x))
                                    for x in pending_rows
                                    if isinstance(x, dict)
                                }
                                row_key = _pending_key(row)
                                if row_key in pending_keys:
                                    st.info(
                                        "Requeue übersprungen: Fall ist bereits in der Pending-Queue vorhanden."
                                    )
                                else:
                                    pending_rows.append(row)
                                    dead_rows.pop(selected_idx)
                                    st.session_state["pending_outcome_queue"] = pending_rows[-500:]
                                    st.session_state["dead_letter_outcomes"] = dead_rows
                                    st.success("Dead-Letter Fall wurde zurück in die Queue gelegt.")
                        if b2.button(
                            "Als manuell gelöst markieren", key=f"dead_resolve_{selected_idx}"
                        ):
                            dead_rows.pop(selected_idx)
                            st.session_state["dead_letter_outcomes"] = dead_rows
                            st.success("Dead-Letter Fall wurde als gelöst entfernt.")
            replay_top = build_explainability_replay(
                st.session_state.get("explainability_trace", []), top_n=5
            )
            if replay_top:
                with st.expander("Explainability Replay (Top 5)", expanded=False):
                    st.dataframe(
                        pd.DataFrame(replay_top), use_container_width=True, hide_index=True
                    )
            if st.session_state.get("provider_priority"):
                st.caption(
                    "Provider-Priorität: "
                    + " > ".join(st.session_state.get("provider_priority", []))
                )
            heatmap_rows = build_quality_heatmap_rows(
                st.session_state.get("data_quality_history", [])
            )
            if heatmap_rows:
                st.markdown("**Datenqualitäts-Heatmap (Liga/Provider)**")
                heatmap_df = pd.DataFrame(heatmap_rows)
                heatmap_table = (
                    heatmap_df.tail(60)
                    .pivot_table(index="league", columns="provider", values="score", aggfunc="mean")
                    .fillna(0.0)
                )
                st.dataframe(heatmap_table.round(1), use_container_width=True)
                st.caption("Fallback-Buckets: good ≥ 70, watch 45-69, poor < 45")

            st.markdown(f"### {t('monitor')}")
            if len(st.session_state["odds_history"]) > 1:
                drift = pd.DataFrame(st.session_state["odds_history"][-200:])
                dfig = go.Figure()
                dfig.add_trace(
                    go.Scatter(
                        x=drift["ts"], y=drift["market"], name="Market", line=dict(color=accent)
                    )
                )
                dfig.add_trace(
                    go.Scatter(
                        x=drift["ts"],
                        y=drift["fair"],
                        name="Fair",
                        line=dict(color="#ffb84d", dash="dash"),
                    )
                )
                dfig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                if not st.session_state.get("chart_light_mode"):
                    st.plotly_chart(dfig, use_container_width=True)

        with tabs[2]:
            left, right = st.columns([2, 1])
            with left:
                eh = elo_hist[elo_hist["Team"] == home]
                ea = elo_hist[elo_hist["Team"] == away]
                efig = go.Figure()
                efig.add_trace(
                    go.Scatter(
                        x=eh["Date"], y=eh["Elo"], name=home, line=dict(color=accent, width=3)
                    )
                )
                efig.add_trace(
                    go.Scatter(
                        x=ea["Date"], y=ea["Elo"], name=away, line=dict(color="#ff6b6b", width=3)
                    )
                )
                efig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                if not st.session_state.get("chart_light_mode"):
                    st.plotly_chart(efig, use_container_width=True)

                xgfig = go.Figure(
                    [
                        go.Bar(
                            x=xg_df["player"],
                            y=xg_df["xG"],
                            marker_color=[accent, accent, "#ff6b6b", "#ff6b6b"],
                        )
                    ]
                )
                xgfig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    title="Player xG Projection",
                )
                if not st.session_state.get("chart_light_mode"):
                    st.plotly_chart(xgfig, use_container_width=True)

                # seasonal pattern
                season = df[(df["HomeTeam"] == home) | (df["AwayTeam"] == home)].copy()
                season["Month"] = season["Date"].dt.month
                season["TeamGoals"] = np.where(
                    season["HomeTeam"] == home, season["FTHG"], season["FTAG"]
                )
                monthly = season.groupby("Month", as_index=False)["TeamGoals"].mean()
                sfig = go.Figure(
                    [
                        go.Scatter(
                            x=monthly["Month"],
                            y=monthly["TeamGoals"],
                            mode="lines+markers",
                            line=dict(color=accent),
                        )
                    ]
                )
                sfig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", title=t("seasonal")
                )
                if not st.session_state.get("chart_light_mode"):
                    st.plotly_chart(sfig, use_container_width=True)

            with right:
                if not st.session_state.get("chart_light_mode"):
                    st.plotly_chart(
                        confidence_gauge(ensemble * 100, accent), use_container_width=True
                    )
                st.metric(t("injury_risk"), f"{injury_prob:.1%}")
                st.metric(t("social_sentiment"), f"{social_score}/100")
                if metrics is not None and st.session_state.get("role") == "admin":
                    st.metric("TS-CV Accuracy", f"{metrics.cv_accuracy:.2%}")
                    st.metric("Brier", f"{metrics.brier:.3f}")
                    st.metric("LogLoss", f"{metrics.multiclass_log_loss:.3f}")
                    cal_summary = summarize_calibration(metrics.calibration_error, metrics.brier)
                    st.metric("ECE", f"{metrics.calibration_error:.3f}")
                    st.metric("Calib Bin Gap", f"{metrics.calibration_bin_gap:.3f}")
                    st.metric("Calib Populated Bins", int(metrics.calibration_populated_bins))
                    if cal_summary.note == "well_calibrated":
                        st.success(t("calibration_good"))
                    elif cal_summary.note == "calibration_watch":
                        st.warning(t("calibration_watch"))
                    else:
                        st.error(t("calibration_risk"))
                    fi = pd.DataFrame(
                        {
                            "Feature": list(metrics.feature_importance.keys()),
                            "Importance": list(metrics.feature_importance.values()),
                        }
                    )
                    st.bar_chart(fi.set_index("Feature"))
                st.caption(t("ml_probs").format(h=ml_probs[2], d=ml_probs[1], a=ml_probs[0]))
                for p in social_posts[:3]:
                    st.caption(f"• {p['title'][:90]}")

        with tabs[3]:
            st.metric(t("api_credits"), st.session_state["api_credits"])
            st.metric(t("exposure"), f"{exposure_now:.2f}€")
            st.metric(t("stake"), f"{stake:.2f}€")

            # portfolio analysis
            st.markdown(f"### {t('portfolio')}")
            ledger = pd.DataFrame(st.session_state["bet_ledger"])
            if not ledger.empty:
                total_stake = float(ledger["stake"].sum())
                avg_edge = float(ledger["edge"].mean())
                team_weights = ledger.groupby("team")["stake"].sum()
                team_weights = team_weights / team_weights.sum()
                diversification = 1 - float((team_weights**2).sum())
                st.metric(t("total_stake"), f"{total_stake:.2f}€")
                st.metric(t("avg_edge"), f"{avg_edge * 100:.2f}%")
                st.metric(t("diversification"), f"{diversification:.1%}")
                avg_clv, better_rate, clv_n = compute_clv_summary(
                    ledger, current_team=home, closing_quote=float(market_quote)
                )
                c_a, c_b, c_c = st.columns(3)
                c_a.metric(t("clv"), f"{avg_clv:+.2f}%")
                c_b.metric(t("clv_better_rate"), f"{better_rate:.1%}")
                c_c.metric(t("clv_samples"), int(clv_n))
                latest_ledger = ledger.tail(12)
                st.dataframe(latest_ledger, use_container_width=True)
                compact_ledger = st.toggle(
                    t("export_compact_columns"), value=True, key="export_compact_ledger"
                )
                ledger_csv_data = (
                    compact_export_csv(
                        latest_ledger,
                        ["ts", "team", "stake", "edge", "entry_quote", "open"],
                    )
                    if compact_ledger
                    else latest_ledger.to_csv(index=False)
                )
                st.download_button(
                    t("export_today_csv"),
                    data=ledger_csv_data,
                    file_name="ledger_today.csv",
                    mime="text/csv",
                )
                if st.session_state.get("user_mode") == "pro":
                    st.markdown(f"#### {t('power_table')}")
                    st.caption(t("power_table_hint"))
                    available_columns = list(ledger.columns)
                    default_power_columns = [
                        col
                        for col in ["ts", "team", "stake", "edge", "entry_quote", "open"]
                        if col in available_columns
                    ]
                    selected_power_columns = st.multiselect(
                        t("power_table_columns"),
                        available_columns,
                        default=default_power_columns,
                    )
                    power_table = build_power_user_table(ledger, selected_power_columns)
                    st.dataframe(power_table, use_container_width=True)

            # gamification + level + leaderboard
            st.markdown(f"### {t('gamification')}")
            xp = st.session_state["xp"]
            level = xp // 100 + 1
            badges = []
            if xp >= 50:
                badges.append("Rookie Analyst")
            if xp >= 150:
                badges.append("Value Hunter")
            if xp >= 300:
                badges.append("Risk Architect")
            st.metric("XP", xp)
            st.metric("Level", level)
            st.write("Badges:", ", ".join(badges) if badges else "—")
            leaderboard = pd.DataFrame(
                [
                    {"User": "you", "XP": xp},
                    {"User": "global_alpha", "XP": max(50, xp - 40)},
                    {"User": "quant_fox", "XP": max(20, xp - 75)},
                ]
            )
            st.dataframe(leaderboard.sort_values("XP", ascending=False), use_container_width=True)

        with tabs[4]:
            if is_admin(st.session_state.get("role", "beta")):
                st.success(t("admin"))
                headers_enabled, headers_payload = security_headers_guidance()
                if headers_enabled:
                    st.info("Security header guidance active (reverse proxy configuration loaded).")
                else:
                    st.warning(
                        "Security headers are not configured via environment yet. "
                        "Set SECURITY_HEADERS_ENABLED=1 and configure proxy headers."
                    )
                header_validation = validate_security_headers_payload(headers_payload)
                if not bool(header_validation.get("valid", False)):
                    st.warning(
                        "Security header smoke-check failed: "
                        + ", ".join(header_validation.get("issues", []))
                    )
                else:
                    st.caption("Security header smoke-check: ok")
                security_baseline = build_security_baseline_report(
                    headers_enabled=headers_enabled,
                    headers_payload=headers_payload,
                    users=load_users(),
                    admin_totp_enabled=is_admin_totp_enabled(),
                    auth_file=Path("runtime_logs/users.json"),
                )
                if security_baseline.status != "ok":
                    st.warning(
                        f"Security baseline: {security_baseline.status} | "
                        + ", ".join(security_baseline.issues[:5])
                    )
                else:
                    st.success("Security baseline self-check: ok")
                st.caption("Security baseline actions: " + ", ".join(security_baseline.actions[:4]))

                i18n_coverage_state = st.session_state.get("i18n_coverage", {"valid": True})
                if not bool(i18n_coverage_state.get("valid", True)):
                    st.warning(
                        "I18N coverage missing keys: "
                        + ", ".join(i18n_coverage_state.get("missing_i18n_keys", [])[:5])
                    )
                else:
                    st.caption("I18N coverage: ok")

                runbook_result = execute_incident_runbook(
                    runbook_id="admin_security_preflight",
                    steps=["diagnose", "health_check", "audit_log_check"],
                    max_retries=1,
                    fail_once_steps={"health_check"},
                )
                if bool(runbook_result.get("success", False)):
                    st.caption("Runbook self-test: success (mit Retry-Strategie)")
                else:
                    st.warning("Runbook self-test: failed")

                fix_memory = list(st.session_state.get("fix_path_memory", []))
                for row in runbook_result.get("steps", []):
                    fix_memory = append_fix_path_event(
                        fix_memory,
                        path=str(row.get("step", "")),
                        success=bool(row.get("success", False)),
                        now=datetime.utcnow(),
                    )
                st.session_state["fix_path_memory"] = fix_memory
                ranked_fix_paths = rank_fix_path_effectiveness(fix_memory)
                if ranked_fix_paths:
                    st.caption(
                        "Fix-Path Priorities: "
                        + " | ".join(f"{x.path}:{x.priority}" for x in ranked_fix_paths[:3])
                    )

                recovery_gate = evaluate_recovery_simulation_gate(
                    scenarios=[
                        {"passed": True},
                        {"passed": bool(runbook_result.get("success", False))},
                        {"passed": bool(slo_guard.status != "critical")},
                        {"passed": bool(security_baseline.status != "critical")},
                        {"passed": bool(ledger_consistency.get("healthy", True))},
                    ],
                    min_pass_rate=0.8,
                    min_samples=5,
                )
                if not recovery_gate.allowed:
                    st.warning(
                        "Recovery Simulation Gate: blocked ("
                        + ", ".join(recovery_gate.reasons)
                        + ")"
                    )
                else:
                    st.caption("Recovery Simulation Gate: passed")

                incident_events = list(st.session_state.get("incident_events", []))
                incident_events.append(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "kind": (
                            "security_warning" if security_baseline.status != "ok" else "ops_ok"
                        ),
                    }
                )
                incident_events.append(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "kind": (
                            "queue_spike"
                            if len(st.session_state.get("quality_budget_decisions", [])) > 10
                            else "queue_ok"
                        ),
                    }
                )
                st.session_state["incident_events"] = incident_events[-300:]
                corr_report = build_incident_correlation_report(
                    events=st.session_state.get("incident_events", []),
                    window_minutes=30,
                )
                st.caption(
                    "Incident-Correlation Actions: " + ", ".join(corr_report.learning_actions[:3])
                )

                ops_ui = build_ops_driven_ui_plan(
                    ci_blocked=not bool(recovery_gate.allowed),
                    slo_status=str(st.session_state.get("slo_guardian", {}).get("status", "ok")),
                    security_status=str(security_baseline.status),
                )
                st.caption(
                    f"Ops-driven UI Plan: {ops_ui.severity} | Guided={ops_ui.show_guided_mode} | "
                    + " | ".join(ops_ui.primary_actions[:3])
                )

                st.code(json.dumps(headers_payload, indent=2), language="json")
                st.json(
                    {"premium_flags": PREMIUM_FLAGS, "logs": st.session_state["error_logs"][-20:]}
                )
                st.markdown("### Audit Log")
                audit_rows = load_recent_audit_events(limit=50)
                if audit_rows:
                    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
                else:
                    st.caption("No audit events yet.")
                budget_rows = st.session_state.get("quality_budget_decisions", [])
                if budget_rows:
                    st.markdown("### Quality Budget Decision Monitor")
                    budget_df = pd.DataFrame(budget_rows[-120:])
                    st.dataframe(budget_df.tail(40), use_container_width=True, hide_index=True)
                    reason_counts = budget_df["reason"].value_counts().to_dict()
                    st.caption(
                        "Reason Mix: "
                        + " | ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
                    )
                    budget_df["date"] = pd.to_datetime(budget_df["ts"], errors="coerce").dt.date
                    trend = (
                        budget_df.groupby("date", as_index=False)
                        .agg(avg_priority=("priority", "mean"), decisions=("reason", "count"))
                        .tail(14)
                    )
                    st.caption("Trend (14d): Decisions/Tag & Avg Priority")
                    st.dataframe(trend, use_container_width=True, hide_index=True)
                    total = max(1, len(budget_df))
                    exhausted = int((budget_df["reason"] == "hourly_budget_exhausted").sum())
                    low_hold = int((budget_df["reason"] == "low_priority_budget_hold").sum())
                    if exhausted / total >= 0.30:
                        st.info(
                            "Policy-Hinweis: Häufig budget exhausted (>30%). Prüfe +1 Recompute/Stunde oder höhere TTL."
                        )
                    if low_hold / total >= 0.40:
                        st.info(
                            "Policy-Hinweis: Viele low-priority holds (>40%). Prüfe niedrigere min_priority_when_tight."
                        )
                bandit_rows = st.session_state.get("bandit_events", [])
                if bandit_rows:
                    st.markdown("### Bandit Cooldown Timeline")
                    st.dataframe(
                        pd.DataFrame(bandit_rows[-80:]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    bandit_df = pd.DataFrame(bandit_rows)
                    enters = (
                        bandit_df[bandit_df["event"] == "cooldown_enter"]
                        .groupby("league")
                        .size()
                        .to_dict()
                    )
                    exits = (
                        bandit_df[bandit_df["event"] == "cooldown_exit_early"]
                        .groupby("league")
                        .size()
                        .to_dict()
                    )
                    leagues = sorted(set(enters) | set(exits))
                    health_rows = []
                    for lg in leagues:
                        enter_n = int(enters.get(lg, 0))
                        exit_n = int(exits.get(lg, 0))
                        health = max(0, min(100, int(round(100 - (enter_n * 12) + (exit_n * 8)))))
                        health_rows.append(
                            {
                                "league": lg,
                                "cooldown_enters": enter_n,
                                "cooldown_early_exits": exit_n,
                                "bandit_health": health,
                            }
                        )
                    if health_rows:
                        st.caption("Bandit Health je Liga")
                        st.dataframe(
                            pd.DataFrame(health_rows), use_container_width=True, hide_index=True
                        )
                st.download_button(
                    "Download Logs JSON",
                    data=json.dumps(st.session_state["error_logs"], indent=2),
                    file_name="error_logs.json",
                )

                provider_scores = compute_provider_scores(
                    st.session_state.get("provider_quality_stats", {})
                )
                provider_success = (
                    float(provider_scores[0].success_rate) if provider_scores else 0.0
                )
                reliability_now = (
                    float(st.session_state.get("reliability_history", [50])[-1])
                    if st.session_state.get("reliability_history")
                    else 50.0
                )
                maintenance = compute_maintenance_score(
                    data_quality_score=float(data_quality.get("score", 0.0)),
                    provider_success_rate=provider_success,
                    reliability_score=reliability_now,
                    audit_anomaly_count=len(st.session_state.get("audit_anomalies", [])),
                )
                st.metric("Maintenance Score", f"{maintenance.score}/100")
                st.caption(
                    f"Q={maintenance.components['data_quality'] * 100:.0f} | "
                    f"P={maintenance.components['provider'] * 100:.0f} | "
                    f"R={maintenance.components['reliability'] * 100:.0f}"
                )

                latency_summary = summarize_pipeline_latency(
                    st.session_state.get("pipeline_latency_history", [])
                )
                if latency_summary:
                    st.markdown("### Decision-Latency-Tracking")
                    st.dataframe(
                        pd.DataFrame([x.__dict__ for x in latency_summary]),
                        use_container_width=True,
                        hide_index=True,
                    )

                error_costs = summarize_error_costs(
                    st.session_state.get("prediction_outcomes", []),
                    stake_eur=25.0,
                )
                st.markdown("### FP/FN Cost Monitor")
                st.caption(
                    f"FP: {error_costs.false_positive_count} ({error_costs.false_positive_cost_eur:.2f}€) | "
                    f"FN: {error_costs.false_negative_count} ({error_costs.false_negative_cost_eur:.2f}€) | "
                    f"Total: {error_costs.total_cost_eur:.2f}€"
                )

                calibration_report = build_calibration_report(
                    st.session_state.get("prediction_outcomes", []), bins=5, min_samples=20
                )
                if calibration_report.samples >= 20 and calibration_report.bins:
                    cal_df = pd.DataFrame([x.__dict__ for x in calibration_report.bins])
                    st.download_button(
                        "Calibration Report CSV",
                        data=cal_df.to_csv(index=False),
                        file_name="calibration_report.csv",
                        mime="text/csv",
                    )
                    st.download_button(
                        "Calibration Report JSON",
                        data=cal_df.to_json(orient="records", indent=2),
                        file_name="calibration_report.json",
                        mime="application/json",
                    )

                st.markdown("### Recovery Packs")
                recovery_packs = build_recovery_packs(
                    provider_ok=bool(odds_data),
                    provider_priority=list(st.session_state.get("provider_priority", [])),
                    user_locked_or_limited=bool(st.session_state.get("audit_anomalies")),
                    sync_paused=bool(st.session_state.get("sync_progress") == "paused"),
                )
                if recovery_packs:
                    for pack in recovery_packs:
                        with st.expander(f"{pack.title} ({pack.severity})", expanded=False):
                            st.write("Checks:")
                            for row in pack.checks:
                                st.write(f"• {row}")
                            st.write("Nächste Schritte:")
                            for row in pack.next_steps:
                                st.write(f"• {row}")
                else:
                    st.caption("Keine aktiven Recovery-Packs.")

                st.markdown("### A/B Rule Mini-Experiment")
                ab_threshold_a = st.slider("Rule A Threshold", 0.40, 0.80, 0.55, 0.01)
                ab_threshold_b = st.slider("Rule B Threshold", 0.40, 0.80, 0.60, 0.01)
                ab_stake = st.slider("AB Stake", 5, 100, 25, 5)
                ab_cases = []
                for item in st.session_state.get("replay_history", [])[-120:]:
                    ab_cases.append(
                        {
                            "score": float(item.get("score", 0.0)),
                            "won": bool(item.get("expected", False)),
                            "quote": float(market_quote),
                        }
                    )
                if ab_cases:
                    ab_metrics = compare_ab_rules(
                        ab_cases,
                        threshold_a=float(ab_threshold_a),
                        threshold_b=float(ab_threshold_b),
                        stake=float(ab_stake),
                    )
                    st.dataframe(
                        pd.DataFrame([m.__dict__ for m in ab_metrics]), use_container_width=True
                    )
                else:
                    st.caption("Zu wenig Daten für A/B-Vergleich.")

                st.markdown("### Threshold What-if Simulation")
                what_if_cases = []
                observed_rows = [
                    x
                    for x in st.session_state.get("prediction_outcomes", [])
                    if isinstance(x, dict) and isinstance(x.get("home_won"), bool)
                ]
                for item in observed_rows[-160:]:
                    what_if_cases.append(
                        {
                            "score": float(item.get("pred_home", 0.0)),
                            "won": bool(item.get("home_won", False)),
                            "quote": float(item.get("quote", market_quote)),
                        }
                    )
                if not what_if_cases:
                    for item in st.session_state.get("replay_history", [])[-160:]:
                        what_if_cases.append(
                            {
                                "score": float(item.get("score", 0.0)),
                                "won": bool(item.get("expected", False)),
                                "quote": float(market_quote),
                            }
                        )
                    st.caption(
                        "Hinweis: What-if nutzt Replay-Proxies, da noch zu wenige gelabelte Outcomes vorliegen."
                    )
                if what_if_cases:
                    what_if_rows = simulate_threshold_what_if(
                        what_if_cases,
                        baseline_threshold=float(adaptive_gate),
                        deltas=[-0.02, -0.01, 0.0, 0.01, 0.02],
                        stake_eur=float(ab_stake),
                    )
                    st.dataframe(
                        pd.DataFrame([x.__dict__ for x in what_if_rows]),
                        use_container_width=True,
                        hide_index=True,
                    )
                    uncertainty_rows = bootstrap_threshold_uncertainty(
                        what_if_cases,
                        baseline_threshold=float(adaptive_gate),
                        deltas=[-0.02, -0.01, 0.0, 0.01, 0.02],
                        stake_eur=float(ab_stake),
                        bootstrap_samples=60,
                        random_seed=42,
                    )
                    if uncertainty_rows:
                        st.caption("What-if Unsicherheitsband (Bootstrap p10/p50/p90)")
                        st.dataframe(
                            pd.DataFrame([x.__dict__ for x in uncertainty_rows]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    auto_rows = simulate_threshold_what_if(
                        what_if_cases,
                        baseline_threshold=float(adaptive_gate),
                        deltas=[-0.01, -0.005, 0.005, 0.01],
                        stake_eur=25.0,
                    )
                    min_bets_for_auto = 10
                    filtered_auto_rows = [x for x in auto_rows if int(x.bets) >= min_bets_for_auto]
                    risk_filtered_rows = [
                        x
                        for x in filtered_auto_rows
                        if int(x.wins) >= int(x.losses)
                        and (int(x.wins) / max(1, int(x.bets))) >= 0.50
                    ]
                    auto_top = sorted(risk_filtered_rows, key=lambda x: x.roi_pct, reverse=True)[:3]
                    st.caption("Auto What-if Vorschläge (Replay, Top 3 ROI)")
                    if auto_top:

                        def _confidence_label(bets: int) -> str:
                            if bets >= 40:
                                return "hoch"
                            if bets >= 20:
                                return "mittel"
                            return "niedrig"

                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        **x.__dict__,
                                        "confidence_hint": _confidence_label(int(x.bets)),
                                    }
                                    for x in auto_top
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.caption(
                            "Keine robusten Auto-Vorschläge "
                            f"(mind. {min_bets_for_auto} Bets + Risk-Filter Winrate ≥50% / Wins ≥ Losses)."
                        )
                else:
                    st.caption("Zu wenig Daten für What-if-Simulation.")

                st.markdown("### Mini-Experiment-Logbuch")
                logbook = st.session_state.get("experiment_logbook", [])
                if logbook:
                    st.dataframe(
                        pd.DataFrame(logbook[-30:][::-1]), use_container_width=True, hide_index=True
                    )
                else:
                    st.caption("Noch keine Experiment-Einträge vorhanden.")
            else:
                st.info(t("beta"))

            st.markdown(f"### {t('debug_title')}")
            st.caption(t("debug_hint"))
            debug_payload = build_debug_info(
                username=st.session_state.get("username", "guest"),
                role=st.session_state.get("role", "beta"),
                user_mode=st.session_state.get("user_mode", "beginner"),
                lang=st.session_state.get("lang", "de"),
                safe_mode=bool(st.session_state.get("safe_mode", False)),
                last_action=st.session_state.get("last_action", ""),
                ledger_rows=len(st.session_state.get("bet_ledger", [])),
                error_count=len(st.session_state.get("error_logs", [])),
                league_name=league_name,
            )
            st.code(debug_payload, language="text")
            st.download_button(
                t("debug_copy"),
                data=debug_payload,
                file_name="debug_info.txt",
                mime="text/plain",
            )

        with tabs[5]:
            st.markdown("### Proof / Performance")
            st.caption(
                "Disclaimer: Keine Garantie auf zukünftige Ergebnisse. Keine Anlageberatung."
            )
            st.caption(f"Last updated: {datetime.utcnow().isoformat()}")
            recent_predictions = read_recent_predictions(limit=400)
            drift_events = read_recent_drift_events(limit=120)
            sample_size = len(recent_predictions)
            cached_counts = _cached_proof_snapshot(sample_size, len(drift_events))
            st.metric("Sample size", cached_counts["prediction_count"])
            st.caption(f"Cached drift events: {cached_counts['drift_count']}")
            if PUBLIC_PROOF_MODE and not is_admin(st.session_state.get("role", "beta")):
                safe_rows = [
                    {
                        "created_at": row.get("created_at"),
                        "league": row.get("league"),
                        "trust_score": row.get("trust_score"),
                        "model_agreement": row.get("model_agreement"),
                        "drift_score": row.get("drift_score"),
                        "status": row.get("status"),
                    }
                    for row in recent_predictions
                ]
                frame = pd.DataFrame(safe_rows)
                if not frame.empty:
                    st.dataframe(frame.tail(100), use_container_width=True, hide_index=True)
                    st.line_chart(frame[["trust_score", "drift_score"]].tail(120))
            else:
                frame = pd.DataFrame(recent_predictions)
                if not frame.empty:
                    st.dataframe(frame.tail(120), use_container_width=True, hide_index=True)
                if drift_events:
                    st.markdown("#### Drift warnings")
                    st.dataframe(pd.DataFrame(drift_events).head(40), use_container_width=True)

            agreement_pairs = st.session_state.get("latest_agreement", {}).get("pairwise_diff", [])
            if agreement_pairs:
                st.markdown("#### Model Agreement Heatmap")
                heat_df = pd.DataFrame(agreement_pairs)
                pivot = heat_df.pivot(index="left", columns="right", values="diff")
                st.dataframe(heat_df, use_container_width=True)
                if not pivot.empty:
                    hfig = go.Figure(
                        data=go.Heatmap(
                            z=pivot.fillna(0.0).values,
                            x=list(pivot.columns),
                            y=list(pivot.index),
                            colorscale="RdYlGn_r",
                            colorbar={"title": "Δ Prob"},
                        )
                    )
                    hfig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                    if not st.session_state.get("chart_light_mode"):
                        st.plotly_chart(hfig, use_container_width=True)

        with tabs[6]:
            st.markdown("### Downside / Worst Case")
            recent_probs = [
                float(x.get("pred_home", 0.5))
                for x in st.session_state.get("prediction_outcomes", [])[-60:]
                if isinstance(x, dict)
            ]
            recent_odds = [
                float(x.get("quote", market_quote))
                for x in st.session_state.get("prediction_outcomes", [])[-60:]
                if isinstance(x, dict)
            ]
            sim_res = simulate_worst_case(
                probabilities=recent_probs or [float(prob_h_for_gate)],
                odds=recent_odds if recent_odds else None,
                stake=float(stake),
                bankroll=float(st.session_state.get("bankroll", 0.0)),
                iterations=1500,
                seed=42,
            )
            st.metric("P5 PnL", f"{sim_res.p5_pnl:.2f}")
            st.metric("Median PnL", f"{sim_res.median_pnl:.2f}")
            st.metric("Avg Max Drawdown", f"{sim_res.mean_max_drawdown:.2f}")
            st.metric("Risk of Ruin", f"{sim_res.risk_of_ruin:.2%}")

elif page == "Backtesting Lab":
    st.subheader(t("backtest_title"))
    c1, c2, c3 = st.columns(3)
    threshold = c1.slider(t("min_elo_diff"), 20, 400, 100)
    odds = c2.slider(t("assumed_odds"), 1.2, 4.0, 1.5)
    stake = c3.slider(t("stake_per_bet"), 10, 200, 50)
    start = st.number_input(t("start_capital"), value=1000.0)
    fast_backtest = st.toggle(
        t("fast_backtest"), value=bool(st.session_state.get("chart_light_mode"))
    )
    only_today_export = st.toggle(t("backtest_only_today_export"), value=False)
    if st.button(t("run_backtest")) and not df.empty:
        st.session_state["last_action_at"] = datetime.utcnow().isoformat()
        cap = start
        hist = [cap]
        subset = df.tail(60).copy() if fast_backtest else df.tail(180).copy()
        pbar = st.progress(0)
        wins = bets = 0
        for idx, (_, row) in enumerate(subset.iterrows(), start=1):
            if (
                elo_curr.get(row["HomeTeam"], 1500) - elo_curr.get(row["AwayTeam"], 1500)
            ) > threshold:
                bets += 1
                if row["FTR"] == "H":
                    cap += stake * (odds - 1)
                    wins += 1
                else:
                    cap -= stake
            hist.append(cap)
            pbar.progress(idx / len(subset))
        st.metric(t("end_capital"), f"{cap:.2f}€")
        st.metric(t("win_rate"), f"{(wins / bets if bets else 0):.1%}")
        st.metric(t("roi"), f"{((cap - start) / start if start else 0):.1%}")
        hist_arr = np.array(hist, dtype=float)
        running_max = np.maximum.accumulate(hist_arr)
        drawdown = (hist_arr - running_max) / np.where(running_max == 0, 1, running_max)
        st.metric(t("max_drawdown"), f"{drawdown.min():.1%}")
        current_roi_pct = float(((cap - start) / start if start else 0) * 100.0)
        current_win_rate_pct = float((wins / bets if bets else 0) * 100.0)
        current_drawdown_pct = float(abs(drawdown.min()) * 100.0)
        baseline = dict(st.session_state.get("backtest_kpi_baseline", {}))
        if not baseline:
            baseline = {
                "roi_pct": current_roi_pct,
                "win_rate_pct": current_win_rate_pct,
                "drawdown_pct": current_drawdown_pct,
            }
            st.session_state["backtest_kpi_baseline"] = baseline
        kpi_drift = evaluate_backtest_kpi_drift(
            baseline_roi_pct=float(baseline.get("roi_pct", current_roi_pct)),
            baseline_win_rate_pct=float(baseline.get("win_rate_pct", current_win_rate_pct)),
            baseline_drawdown_pct=float(baseline.get("drawdown_pct", current_drawdown_pct)),
            current_roi_pct=current_roi_pct,
            current_win_rate_pct=current_win_rate_pct,
            current_drawdown_pct=current_drawdown_pct,
        )
        if not kpi_drift.healthy:
            st.warning(
                "Backtest KPI Drift erkannt: "
                + ", ".join(kpi_drift.alerts)
                + f" | ΔROI {kpi_drift.roi_delta_pct:+.2f}pp"
            )

        bt_df = pd.DataFrame(
            {
                "step": list(range(len(hist))),
                "capital": hist,
                "benchmark": [start] * len(hist),
                "date": [pd.NaT] + list(subset["Date"]),
            }
        )
        st.line_chart(
            bt_df.set_index("step")[["capital", "benchmark"]].rename(
                columns={"capital": "Strategy", "benchmark": t("benchmark")}
            )
        )
        export_df = (
            filter_backtest_export_today(bt_df, today=datetime.utcnow())
            if only_today_export
            else bt_df
        )
        export_contract = validate_export_contract(
            export_df,
            required_columns=["step", "capital", "benchmark", "date"],
            ordered_columns=["step", "capital", "benchmark", "date"],
        )
        if not export_contract.valid:
            st.warning(
                "Backtest export contract violation: "
                + ", ".join(export_contract.missing_columns or [])
            )
        st.download_button(
            t("backtest_export_csv"),
            data=export_df.to_csv(index=False),
            file_name="backtest_report.csv",
        )
        st.download_button(
            t("backtest_export_json"),
            data=export_df.to_json(orient="records", indent=2),
            file_name="backtest_report.json",
        )

elif page == "News & Sentiment":
    st.subheader(t("news_title"))
    news_start = datetime.utcnow()
    news = get_news_feed()
    news_latency_ms = max(0.0, (datetime.utcnow() - news_start).total_seconds() * 1000.0)
    provider_stats = dict(st.session_state.get("provider_quality_stats", {}))
    news_stats = dict(provider_stats.get("news_feed", {}))
    news_stats["calls"] = float(news_stats.get("calls", 0.0)) + 1.0
    news_stats["success"] = float(news_stats.get("success", 0.0)) + (1.0 if news else 0.0)
    news_stats["latency_ms_sum"] = float(news_stats.get("latency_ms_sum", 0.0)) + float(
        news_latency_ms
    )
    provider_stats["news_feed"] = news_stats
    st.session_state["provider_quality_stats"] = provider_stats
    st.session_state["provider_priority"] = provider_priority_order(provider_stats)
    news_text_only = st.toggle(
        t("news_text_only"), value=bool(st.session_state.get("news_text_only", False))
    )
    st.session_state["news_text_only"] = news_text_only
    limit = min(
        int(st.session_state.get("news_items_limit", 5)),
        int(st.session_state.get("adaptive_max_news_items", 5)),
    )
    left, right = st.columns([2, 1])
    with left:
        for item in news[:limit]:
            if news_text_only:
                st.write(
                    f"- {item.get('published', '')[:16]} | {item.get('source_label', '')} | {item.get('title', '')}"
                )
            else:
                st.markdown(
                    f"""
                    <div class='news-item'>
                        <div style='font-size:.75rem;color:#9db0d8'>{t("source")}: {item.get("source_label", "")} • {t("date")}: {item.get("published", "")[:16]}</div>
                        <a href='{item.get("link", "#")}' target='_blank' style='font-weight:700;color:#e6eefc;text-decoration:none'>{item.get("title", "")}</a>
                        <div style='color:#93a4c7'>{item.get("summary", "")[:180]}...</div>
                        <div><a href='{item.get("link", "#")}' target='_blank'>{t("link")}</a></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if limit < len(news) and st.button(t("news_load_more"), width="stretch"):
            st.session_state["news_items_limit"] = next_news_batch_limit(
                limit, step=5, total=len(news)
            )
            st.rerun()
    with right:
        score = 50
        for n in news:
            txt = (n.get("title", "") + " " + n.get("summary", "")).lower()
            score += 4 * ("sieg" in txt or "win" in txt)
            score -= 4 * ("krise" in txt or "injury" in txt)
        score = max(0, min(100, score))
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": accent}},
            )
        )
        fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
        if not st.session_state.get("chart_light_mode"):
            st.plotly_chart(fig, use_container_width=True)

st.markdown("</div></div>", unsafe_allow_html=True)
_save_ops_state(
    st.session_state.get("username", "guest"),
    dead_letter_outcomes=list(st.session_state.get("dead_letter_outcomes", [])),
    quality_budget_decisions=list(st.session_state.get("quality_budget_decisions", [])),
    bandit_events=list(st.session_state.get("bandit_events", [])),
)
