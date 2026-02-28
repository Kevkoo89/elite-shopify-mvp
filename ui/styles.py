import streamlit as st


def render_global_styles(
    theme_key: str = "ultra", layout_key: str = "dashboard", login_layout: str = "center"
) -> None:
    st.markdown(
        """
        <style>
        #MainMenu,
        header,
        footer,
        [data-testid="stHeader"],
        [data-testid="stAppToolbar"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            display: none !important;
        }
        [data-testid="stAppViewContainer"], .stApp {
            margin-top: 0 !important;
            padding-top: 0 !important;
            border-top: 0 !important;
            box-shadow: none !important;
        }
        [data-testid="stAppViewContainer"] hr,
        .main hr {
            display: none !important;
            border: 0 !important;
            height: 0 !important;
            margin: 0 !important;
        }

        .ea-root {
            --bg: #0b1020;
            --bg2: #080c16;
            --panel: #111827;
            --panel2: #0f172a;
            --text: #e5ecff;
            --muted: #9fb0d0;
            --border: rgba(148, 163, 184, 0.18);
            --border2: rgba(148, 163, 184, 0.1);
            --shadow-sm: 0 4px 12px rgba(0, 0, 0, 0.22);
            --shadow-md: 0 14px 32px rgba(0, 0, 0, 0.32);
            --shadow-lg: 0 24px 56px rgba(0, 0, 0, 0.4);
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 22px;
            --space-1: 4px;
            --space-2: 8px;
            --space-3: 12px;
            --space-4: 16px;
            --space-5: 20px;
            --space-6: 24px;
            --space-7: 32px;
            --space-8: 40px;
            --accent: #4f7cff;
            --accent2: #26c6ff;
            --accent3: #7f56d9;
            --focus-ring: rgba(79, 124, 255, 0.35);
            --font-sans: "Inter", "Segoe UI", "Roboto", system-ui, sans-serif;
            --font-mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
            --fs-xs: 0.74rem;
            --fs-sm: 0.86rem;
            --fs-base: 0.96rem;
            --fs-lg: 1.08rem;
            --fs-xl: 1.3rem;
            --fs-2xl: 1.72rem;
            --lh-tight: 1.15;
            --lh-normal: 1.45;

            color: var(--text);
            font-family: var(--font-sans);
        }

        .ea-root,
        .ea-root .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .ea-root [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, var(--bg), var(--bg2));
        }

        .ea-root [data-testid="stSidebar"] {
            background: var(--panel2);
            border-right: 1px solid var(--border2);
        }

        .ea-root .block-container {
            max-width: 1320px;
            margin: 0 auto;
            padding-top: var(--space-5);
            padding-bottom: var(--space-6);
        }

        .ea-root .main-container {
            max-width: 1280px;
            margin: 0 auto;
            padding: var(--space-6);
        }

        .ea-root .ea-page {
            display: grid;
            gap: var(--space-6);
        }

        .ea-root .ea-header {
            background: var(--panel);
            border: 1px solid var(--border2);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
            padding: var(--space-5) var(--space-6);
            margin: var(--space-5) 0 var(--space-6);
        }

        .ea-root .ea-header__title,
        .ea-root h1,
        .ea-root h2,
        .ea-root h3 {
            letter-spacing: 0.01em;
            margin-top: var(--space-4);
            margin-bottom: var(--space-3);
        }

        .ea-root .ea-card,
        .ea-root [data-testid="stVerticalBlockBorderWrapper"],
        .ea-root [data-testid="metric-container"],
        .ea-root [data-testid="stMetric"] {
            background: var(--panel);
            color: var(--text);
            border: 1px solid var(--border2);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            padding: var(--space-5);
        }

        .ea-root .ea-card--soft {
            background: color-mix(in srgb, var(--panel) 82%, var(--bg2));
        }

        .ea-root .ea-card--flat {
            box-shadow: none;
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
        }

        .ea-root .ea-kpi,
        .ea-root .kpi-card,
        .ea-root [data-testid="metric-container"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 122px;
            text-align: center;
            gap: var(--space-2);
        }

        .ea-root .kpi-label,
        .ea-root [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: var(--fs-xs);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .ea-root .kpi-value,
        .ea-root [data-testid="stMetricValue"] {
            color: var(--text);
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1;
            letter-spacing: 0.01em;
        }

        .ea-root .stButton > button,
        .ea-root .ea-btn {
            min-height: 42px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
            background: color-mix(in srgb, var(--panel) 70%, #1f2937);
            color: var(--text);
            box-shadow: none;
            transition: all 0.2s ease;
        }

        .ea-root .stButton > button:hover {
            transform: translateY(-1px);
        }

        .ea-root .stButton > button:focus-visible,
        .ea-root input:focus,
        .ea-root textarea:focus {
            outline: none !important;
            box-shadow: 0 0 0 3px var(--focus-ring) !important;
        }

        .ea-root .stTextInput input,
        .ea-root .stTextArea textarea,
        .ea-root .stNumberInput input,
        .ea-root [data-testid="stSelectbox"] > div > div {
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border) !important;
            background: color-mix(in srgb, var(--panel) 90%, #000 10%) !important;
            color: var(--text) !important;
            min-height: 42px;
        }

        .ea-root .ea-sidebar-section {
            background: transparent;
            border: 0;
            border-bottom: 1px solid var(--border2);
            border-radius: 0;
            padding-bottom: var(--space-2);
        }

        .ea-root .ea-sidebar-label {
            color: var(--muted);
            font-size: var(--fs-xs);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .ea-root [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 4px;
        }

        .ea-root [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: var(--radius-md);
            padding: 8px 10px;
            border: 1px solid transparent;
            transition: all 0.2s ease;
            margin-bottom: 4px;
        }

        .ea-root [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: color-mix(in srgb, var(--panel) 76%, transparent);
        }

        .ea-root [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            border-color: var(--accent);
            background: color-mix(in srgb, var(--accent) 18%, transparent);
        }

        .ea-root .news-item {
            padding: var(--space-3) 0;
            border-bottom: 1px solid var(--border2);
        }

        .ea-root .login-page {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: var(--space-6);
        }

        .ea-root .ea-login {
            width: min(1060px, 100%);
        }

        .ea-root .ea-login-card,
        .ea-root .login-card {
            width: 100%;
            max-width: 480px;
            margin: 0 auto;
            padding: var(--space-7) var(--space-6);
            border-radius: var(--radius-xl);
            background: var(--panel);
            border: 1px solid var(--border2);
            box-shadow: var(--shadow-lg);
        }

        .ea-root .ea-login-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--space-3);
        }

        .ea-root .ea-login-title {
            margin: 0;
            font-size: var(--fs-2xl);
            line-height: var(--lh-tight);
        }

        .ea-root .ea-login-sub,
        .ea-root .ea-login-subline,
        .ea-root .ea-trust-line {
            color: var(--muted);
        }

        .ea-root .ea-feature-bullets {
            color: var(--muted);
            font-size: var(--fs-sm);
            line-height: 1.6;
        }

        .ea-root .ea-divider {
            height: 1px;
            border: 0;
            background: var(--border2);
            margin: var(--space-4) 0;
        }

        .ea-root[data-login-layout="split"] .ea-login-card {
            max-width: 520px;
        }

        .ea-root[data-login-layout="minimal"] .ea-login-card {
            max-width: 440px;
            padding-top: var(--space-5);
        }

        /* ===== ULTRA PREMIUM ===== */
        .ea-root[data-theme="ultra"] {
            --bg: #090f1e;
            --bg2: #060a14;
            --panel: rgba(16, 24, 39, 0.92);
            --panel2: rgba(11, 18, 32, 0.96);
            --accent: #4f7cff;
            --accent2: #22d3ee;
            --accent3: #8b5cf6;
            --focus-ring: rgba(79, 124, 255, 0.42);
            --radius-lg: 18px;
            --radius-xl: 24px;
        }

        .ea-root[data-theme="ultra"] [data-testid="stAppViewContainer"] {
            background: radial-gradient(980px 520px at 12% 4%, rgba(79, 124, 255, 0.2), transparent 52%),
                linear-gradient(180deg, var(--bg), var(--bg2));
        }

        .ea-root[data-theme="ultra"] [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 14, 28, 0.98), rgba(6, 10, 20, 0.96));
            box-shadow: inset -1px 0 0 rgba(148, 163, 184, 0.15), inset -26px 0 40px rgba(2, 6, 20, 0.36);
            width: 22rem !important;
            min-width: 22rem !important;
        }

        .ea-root[data-theme="ultra"] .stSelectbox [data-baseweb="select"] > div {
            border-radius: 14px !important;
            border-color: rgba(79, 124, 255, 0.32) !important;
        }

        .ea-root[data-theme="ultra"] .ea-card,
        .ea-root[data-theme="ultra"] [data-testid="metric-container"] {
            border-color: rgba(148, 163, 184, 0.08);
            box-shadow: 0 18px 42px rgba(2, 8, 20, 0.5);
            padding: var(--space-6);
        }

        .ea-root[data-theme="ultra"] .kpi-card,
        .ea-root[data-theme="ultra"] [data-testid="metric-container"] {
            min-height: 150px;
            border-radius: 20px;
        }

        .ea-root[data-theme="ultra"] .kpi-value,
        .ea-root[data-theme="ultra"] [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 760;
        }

        .ea-root[data-theme="ultra"] .stButton > button[kind="primary"] {
            border: 0 !important;
            border-radius: 16px !important;
            background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
            box-shadow: 0 12px 28px rgba(34, 211, 238, 0.24);
        }

        .ea-root[data-theme="ultra"] [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            border: 1px solid rgba(79, 124, 255, 0.45);
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(79, 124, 255, 0.24), rgba(34, 211, 238, 0.2));
            box-shadow: 0 0 0 1px rgba(79, 124, 255, 0.3), 0 0 18px rgba(34, 211, 238, 0.16);
        }

        .ea-root[data-theme="ultra"] .ea-header__title,
        .ea-root[data-theme="ultra"] h2 {
            font-size: 1.22rem;
            font-weight: 650;
            margin-top: var(--space-6);
            margin-bottom: var(--space-4);
        }

        /* ===== BLOOMBERG / INSTITUTIONAL ===== */
        .ea-root[data-theme="bloomberg"] {
            --bg: #0b0f14;
            --panel: #0f1621;
            --panel2: #0c121b;
            --text: #e6edf7;
            --muted: #9db0c7;
            --border: rgba(230, 237, 247, 0.12);
            --border2: rgba(230, 237, 247, 0.18);
            --radius: 6px;
            --radius2: 8px;
            --pad: 10px;
            --pad2: 12px;
            --gap: 10px;
            --shadow: none;
            --focus: rgba(255, 184, 0, 0.35);
            --accent: #ffb800;
            --accent2: #4dd2ff;
            --font: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }

        .ea-root[data-theme="bloomberg"] .stApp {
            background: var(--bg) !important;
            color: var(--text) !important;
        }

        .ea-root[data-theme="bloomberg"] * {
            font-family: var(--font) !important;
            letter-spacing: 0.2px;
        }

        .ea-root[data-theme="bloomberg"] section.main,
        .ea-root[data-theme="bloomberg"] [data-testid="stVerticalBlock"],
        .ea-root[data-theme="bloomberg"] [data-testid="stHorizontalBlock"] {
            background: transparent;
        }

        .ea-root[data-theme="bloomberg"] .main-container {
            padding: var(--pad2);
            gap: var(--gap);
        }

        .ea-root[data-theme="bloomberg"] .ea-card,
        .ea-root[data-theme="bloomberg"] [data-testid="metric-container"],
        .ea-root[data-theme="bloomberg"] [data-testid="stVerticalBlockBorderWrapper"],
        .ea-root[data-theme="bloomberg"] [data-testid="stMetric"] {
            background: var(--panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            box-shadow: none !important;
            padding: var(--pad2) !important;
        }

        .ea-root[data-theme="bloomberg"] [data-testid="stSidebar"] {
            background: var(--panel2) !important;
            border-right: 1px solid var(--border2) !important;
            width: 290px !important;
            min-width: 290px !important;
        }

        .ea-root[data-theme="bloomberg"] [data-testid="stSidebar"] .block-container {
            padding-top: 10px !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
        }

        .ea-root[data-theme="bloomberg"] [data-testid="stSidebar"] details {
            background: transparent;
            border: 1px solid var(--border);
            border-radius: var(--radius);
        }

        .ea-root[data-theme="bloomberg"] [data-testid="stSidebar"] summary {
            text-transform: uppercase;
            font-size: 12px;
            color: var(--muted);
            letter-spacing: 1px;
        }

        .ea-root[data-theme="bloomberg"] .stButton button {
            background: transparent !important;
            border: 1px solid var(--border2) !important;
            border-radius: var(--radius) !important;
            color: var(--text) !important;
            padding: 8px 12px !important;
            box-shadow: none !important;
        }

        .ea-root[data-theme="bloomberg"] .stButton button:hover {
            border-color: var(--accent) !important;
            color: #ffffff !important;
        }

        .ea-root[data-theme="bloomberg"] .stButton button:focus-visible {
            outline: 2px solid var(--focus) !important;
            outline-offset: 1px;
        }

        .ea-root[data-theme="bloomberg"] [data-baseweb="select"] > div {
            background: transparent !important;
            border: 1px solid var(--border2) !important;
            border-radius: var(--radius) !important;
            box-shadow: none !important;
            color: var(--text) !important;
            min-height: 38px !important;
        }

        .ea-root[data-theme="bloomberg"] [data-baseweb="select"] > div:focus-within {
            outline: 2px solid var(--focus) !important;
            outline-offset: 1px;
        }

        .ea-root[data-theme="bloomberg"] [data-baseweb="select"] [class*="placeholder"] {
            color: var(--muted) !important;
        }

        .ea-root[data-theme="bloomberg"] h1,
        .ea-root[data-theme="bloomberg"] h2,
        .ea-root[data-theme="bloomberg"] h3,
        .ea-root[data-theme="bloomberg"] h4,
        .ea-root[data-theme="bloomberg"] p,
        .ea-root[data-theme="bloomberg"] [data-testid="stMetric"] {
            text-align: left !important;
        }

        .ea-root[data-theme="bloomberg"] [data-testid="stMetricLabel"] {
            text-transform: uppercase;
            font-size: 12px !important;
            color: var(--muted) !important;
            letter-spacing: 1px;
        }

        .ea-root[data-theme="bloomberg"] [data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 640;
            text-align: left !important;
        }
        /* ===== NEON TECH / CYBER ===== */
        .ea-root[data-theme="neon"] {
            --bg: #04050f;
            --bg2: #080b18;
            --panel: rgba(12, 15, 30, 0.94);
            --panel2: #05070f;
            --text: #dff0ff;
            --muted: #8ea5bf;
            --border: rgba(91, 116, 145, 0.26);
            --border2: rgba(91, 116, 145, 0.16);
            --accent: #00e5ff;
            --accent2: #ff38d1;
            --focus-ring: rgba(0, 229, 255, 0.42);
            --radius-md: 10px;
            --radius-lg: 12px;
        }

        .ea-root[data-theme="neon"] [data-testid="stAppViewContainer"] {
            background: radial-gradient(900px 460px at 80% -20%, rgba(0, 229, 255, 0.07), transparent 55%),
                linear-gradient(180deg, var(--bg), var(--bg2));
        }

        .ea-root[data-theme="neon"] [data-testid="stSidebar"] {
            background: #04060d;
            border-right: 1px solid rgba(0, 229, 255, 0.2);
            width: 20rem !important;
            min-width: 20rem !important;
        }

        .ea-root[data-theme="neon"] .stSelectbox [data-baseweb="select"] > div {
            border-radius: 10px !important;
            border-color: rgba(91, 116, 145, 0.35) !important;
        }

        .ea-root[data-theme="neon"] .stSelectbox [data-baseweb="select"] > div:focus-within {
            box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.28) !important;
        }

        .ea-root[data-theme="neon"] .ea-card,
        .ea-root[data-theme="neon"] [data-testid="metric-container"] {
            background: rgba(10, 13, 28, 0.9);
            border: 1px solid rgba(91, 116, 145, 0.14);
            box-shadow: none;
        }

        .ea-root[data-theme="neon"] .kpi-value,
        .ea-root[data-theme="neon"] [data-testid="stMetricValue"] {
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.18);
        }

        .ea-root[data-theme="neon"] .stButton > button {
            transition: all 0.14s ease;
            background: #0f1528;
            border: 1px solid rgba(91, 116, 145, 0.22);
        }

        .ea-root[data-theme="neon"] .stButton > button[kind="primary"] {
            background: #11182e !important;
            border: 1px solid rgba(0, 229, 255, 0.34) !important;
        }

        .ea-root[data-theme="neon"] .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 0 0 1px rgba(0, 229, 255, 0.32), 0 0 15px rgba(255, 56, 209, 0.18);
        }

        .ea-root[data-theme="neon"] [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 0;
            border-bottom: 1px solid rgba(91, 116, 145, 0.2);
            margin-bottom: 2px;
        }

        .ea-root[data-theme="neon"] [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            border: 0;
            border-bottom: 2px solid #00e5ff;
            background: rgba(0, 229, 255, 0.06);
            box-shadow: inset 0 -1px 0 rgba(255, 56, 209, 0.35);
        }

        .ea-root[data-theme="neon"] .ea-card:hover {
            box-shadow: 0 0 0 1px rgba(0, 229, 255, 0.24);
        }

        /* ===== CLEAN ENTERPRISE ===== */
        .ea-root[data-theme="clean"] {
            --bg: #f3f5f8;
            --bg2: #edf1f6;
            --panel: #ffffff;
            --panel2: #eef2f6;
            --text: #111827;
            --muted: #6b7280;
            --border: #d8dee9;
            --border2: #e2e8f1;
            --accent: #335eea;
            --accent2: #3b82f6;
            --focus-ring: rgba(51, 94, 234, 0.24);
            --shadow-sm: 0 2px 8px rgba(17, 24, 39, 0.05);
            --shadow-md: 0 8px 20px rgba(17, 24, 39, 0.06);
            --shadow-lg: 0 16px 34px rgba(17, 24, 39, 0.08);
            --radius-lg: 14px;
        }

        .ea-root[data-theme="clean"] [data-testid="stSidebar"] {
            background: #f1f3f5;
            border-right: 1px solid #d9dfe8;
            box-shadow: none;
            width: 20rem !important;
            min-width: 20rem !important;
        }

        .ea-root[data-theme="clean"] .stSelectbox [data-baseweb="select"] > div {
            border-radius: 10px !important;
            min-height: 42px !important;
            background: #ffffff !important;
            border: 1px solid #d3dbea !important;
        }

        .ea-root[data-theme="clean"] .stSelectbox [data-baseweb="select"] > div:focus-within {
            box-shadow: 0 0 0 3px rgba(51, 94, 234, 0.14) !important;
        }

        .ea-root[data-theme="clean"] .main-container {
            padding: 26px 28px;
        }

        .ea-root[data-theme="clean"] .ea-card,
        .ea-root[data-theme="clean"] [data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid var(--border2);
            box-shadow: var(--shadow-sm);
        }

        .ea-root[data-theme="clean"] .kpi-card,
        .ea-root[data-theme="clean"] [data-testid="metric-container"] {
            min-height: 106px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(17, 24, 39, 0.04);
        }

        .ea-root[data-theme="clean"] .stButton > button {
            border-radius: 10px !important;
            background: #ffffff;
            border: 1px solid #cfd8e7;
            color: #111827;
        }

        .ea-root[data-theme="clean"] .stButton > button[kind="primary"] {
            background: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid #2948b8 !important;
            box-shadow: none !important;
        }

        .ea-root[data-theme="clean"] [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 999px;
            border: 1px solid transparent;
            background: transparent;
            padding: 8px 12px;
        }

        .ea-root[data-theme="clean"] [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: rgba(51, 94, 234, 0.1);
            border: 1px solid rgba(51, 94, 234, 0.22);
            box-shadow: none;
        }

        @media (max-width: 960px) {
            .ea-root .main-container {
                padding: var(--space-4);
            }
            .ea-root .ea-login-card {
                padding: var(--space-5);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
