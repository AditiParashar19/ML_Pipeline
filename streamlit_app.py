import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State Init ─────────────────────────────────────────────────────────
defaults = {
    'step': 0,
    'problem_type': None,
    'df': None,
    'df_clean': None,
    'target': None,
    'features': None,
    'selected_features': None,
    'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
    'model': None,
    'model_name': None,
    'model_params': {},
    'k_folds': 5,
    'cv_results': None,
    'test_results': None,
    'outlier_mask': None,
    'outlier_method': None,
    'outlier_count': 0,
    'theme': 'dark',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Read theme ONCE at top so entire page uses same value ─────────────────────
is_dark = st.session_state.theme == 'dark'

# ── Color helpers ─────────────────────────────────────────────────────────────
def acc():         return "#00d4ff" if is_dark else "#0284c7"
def acc3():        return "#10b981" if is_dark else "#059669"
def muted():       return "#64748b"
def text_col():    return "#e2e8f0" if is_dark else "#1e293b"
def text_strong(): return "#ffffff" if is_dark else "#0f172a"

# ── Theme CSS ─────────────────────────────────────────────────────────────────
DARK_VARS = """
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1e3a5f;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --accent3: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --text-strong: #ffffff;
    --muted: #64748b;
    --card-bg: rgba(17,24,39,0.8);
    --sidebar-bg: #111827;
    --input-bg: #1a2235;
    --shadow: rgba(0,0,0,0.4);
    --plot-bg: rgba(17,24,39,0.6);
    --hgA: #1a2235;
    --hgB: #111827;
    --dot-text: #0a0e1a;
    --info-bg: rgba(0,212,255,0.08);
    --info-border: rgba(0,212,255,0.3);
    --chip-bg: rgba(0,212,255,0.12);
    --chip-border: rgba(0,212,255,0.3);
    --df-bg: #1a2235;
    --df-text: #e2e8f0;
    --df-header-bg: #0a0e1a;
    --df-header-text: #00d4ff;
    --df-row-hover: rgba(0,212,255,0.06);
    --df-border: #1e3a5f;
    --expander-text: #ffffff;
"""
LIGHT_VARS = """
    --bg: #f0f4f8;
    --surface: #ffffff;
    --surface2: #e8eef5;
    --border: #c3d4e8;
    --accent: #0284c7;
    --accent2: #7c3aed;
    --accent3: #059669;
    --warn: #d97706;
    --danger: #dc2626;
    --text: #1e293b;
    --text-strong: #0f172a;
    --muted: #64748b;
    --card-bg: rgba(255,255,255,0.9);
    --sidebar-bg: #ffffff;
    --input-bg: #f8fafc;
    --shadow: rgba(0,0,0,0.08);
    --plot-bg: rgba(248,250,252,0.9);
    --hgA: #e8eef5;
    --hgB: #ffffff;
    --dot-text: #ffffff;
    --info-bg: rgba(2,132,199,0.08);
    --info-border: rgba(2,132,199,0.35);
    --chip-bg: rgba(2,132,199,0.12);
    --chip-border: rgba(2,132,199,0.4);
    --df-bg: #ffffff;
    --df-text: #1e293b;
    --df-header-bg: #e8eef5;
    --df-header-text: #0284c7;
    --df-row-hover: rgba(2,132,199,0.05);
    --df-border: #c3d4e8;
    --expander-text: #1e293b;
"""

theme_vars = DARK_VARS if is_dark else LIGHT_VARS

# ── Theme icon: only moon or sun, no text ─────────────────────────────────────
toggle_icon = "☀️" if is_dark else ""
toggle_tooltip = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"

# Expander header text color — explicit per theme
expander_text_color = "#ffffff" if is_dark else "#1e293b"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;800&display=swap');

:root {{ {theme_vars} }}

html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}}
.stApp {{ background: var(--bg) !important; }}

[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--text) !important; }}

h1, h2, h3, h4, h5, h6 {{
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-strong) !important;
}}

label, .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
.stMarkdown strong, .stMarkdown span,
[data-testid="stWidgetLabel"] > div, [data-testid="stWidgetLabel"] p,
.stRadio label div p, .stCheckbox label div p,
.stSelectbox label, .stMultiSelect label,
.stSlider label, .stNumberInput label,
.stTextInput label, .stFileUploader label,
div[data-testid="stForm"] label, .element-container p, p {{
    color: var(--text) !important;
}}

/* ── Theme toggle button — icon only, circular ── */
div[data-testid="stToggle"] {{
    display: flex !important;
    align-items: center !important;
}}
div[data-testid="stToggle"] label {{ display: none !important; }}
div[data-testid="stToggle"] p    {{ display: none !important; }}

.theme-toggle-wrapper {{ display: flex; justify-content: flex-end; margin-bottom: 12px; }}
.theme-icon-btn {{
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 50%; width: 42px; height: 42px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; cursor: pointer; transition: all 0.2s;
    box-shadow: 0 2px 8px var(--shadow);
}}
.theme-icon-btn:hover {{
    border-color: var(--accent);
    box-shadow: 0 0 12px rgba(0,180,255,0.3);
    transform: scale(1.08);
}}

/* ── Buttons ── */
.stButton > button {{
    background: var(--accent) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    transition: opacity 0.2s !important;
}}
.stButton > button:hover {{ opacity: 0.85 !important; }}

/* ── Step Header ── */
.step-header {{
    background: linear-gradient(135deg, var(--hgA), var(--hgB));
    border: 1px solid var(--border); border-left: 4px solid var(--accent);
    border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: 0 2px 12px var(--shadow);
}}
.step-number {{
    background: var(--accent); color: var(--dot-text);
    width: 40px; height: 40px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace; font-weight: 700; font-size: 16px; flex-shrink: 0;
}}
.step-title   {{ font-size: 22px; font-weight: 800; margin: 0; color: var(--text-strong); }}
.step-subtitle {{ color: var(--muted); font-size: 13px; margin: 2px 0 0; }}

/* ── Metric Cards ── */
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 16px; margin: 20px 0; }}
.metric-card {{
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; text-align: center;
    box-shadow: 0 2px 8px var(--shadow);
}}
.metric-value {{ font-size: 32px; font-weight: 800; color: var(--accent); font-family: 'Space Mono', monospace; }}
.metric-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}

/* ── Boxes ── */
.info-box {{
    background: var(--info-bg); border: 1px solid var(--info-border);
    border-radius: 10px; padding: 14px 18px; margin: 12px 0; font-size: 14px; color: var(--text);
}}
.warn-box {{
    background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.3);
    border-radius: 10px; padding: 14px 18px; margin: 12px 0; font-size: 14px; color: var(--warn);
}}
.success-box {{
    background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px; padding: 14px 18px; margin: 12px 0; font-size: 14px; color: var(--accent3);
}}
.danger-box {{
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px; padding: 14px 18px; margin: 12px 0; font-size: 14px; color: var(--danger);
}}

.section-divider {{ border: none; border-top: 1px solid var(--border); margin: 28px 0; }}

.chip {{
    display: inline-block; background: var(--chip-bg); border: 1px solid var(--chip-border);
    color: var(--accent); border-radius: 20px; padding: 3px 12px; font-size: 12px; margin: 2px;
}}

/* ── Horizontal steps bar ── */
.hsteps {{
    display: flex; align-items: center; margin: 0 0 32px;
    background: var(--surface); border-radius: 14px; padding: 14px 20px;
    border: 1px solid var(--border); overflow-x: auto; gap: 0;
    box-shadow: 0 2px 8px var(--shadow);
}}
.hstep {{ display: flex; align-items: center; gap: 8px; white-space: nowrap; flex-shrink: 0; }}
.hstep-dot {{
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; font-family: 'Space Mono', monospace;
    border: 2px solid var(--border); color: var(--muted);
    background: var(--surface2); flex-shrink: 0;
}}
.hstep-dot.done   {{ background: var(--accent3); border-color: var(--accent3); color: white; }}
.hstep-dot.active {{ background: var(--accent); border-color: var(--accent); color: var(--dot-text); box-shadow: 0 0 16px rgba(0,180,255,0.4); }}
.hstep-label        {{ font-size: 12px; color: var(--muted); font-weight: 500; }}
.hstep-label.active {{ color: var(--accent); font-weight: 700; }}
.hstep-label.done   {{ color: var(--accent3); }}
.hstep-connector      {{ width: 40px; height: 2px; background: var(--border); margin: 0 4px; flex-shrink: 0; }}
.hstep-connector.done {{ background: var(--accent3); }}

/* ── Sidebar logo ── */
.logo-area  {{ padding: 20px 0 24px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }}
.logo-title {{ font-size: 20px; font-weight: 800; color: var(--accent); letter-spacing: -0.5px; }}
.logo-sub   {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; }}

/* ── Inputs ── */
.stSelectbox > div > div, .stMultiSelect > div > div,
.stNumberInput > div > div > input, .stTextInput > div > div > input {{
    background: var(--input-bg) !important; border-color: var(--border) !important; color: var(--text) !important;
}}

/* ── Expanders — header text explicitly white in dark, dark in light ── */
.streamlit-expanderHeader {{
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: {expander_text_color} !important;
}}
/* Target every text node inside the expander header */
/* ── Remove White Hover Effect on Expander Headers ── */
.streamlit-expanderHeader:hover {{
    background: var(--surface2) !important; /* Forces background to stay consistent */
    border-color: var(--border) !important;
}}
[data-testid="stToast"] {{
    background-color: var(--surface) !important; /* Uses your dark/light surface color */
    border: 1px solid var(--border) !important;
}}

/* This targets the actual text inside the toast */
[data-testid="stToast"] .stMarkdown p {{
    color: var(--text-strong) !important; /* Changes to white in dark, dark-blue in light */
    font-weight: 500 !important;
}}

/* Target the icon inside the toast */
[data-testid="stToast"] [data-testid="stIcon"] {{
    color: var(--accent) !important;
}}

/* Forces text color to stay readable even if Streamlit tries to invert it */
.streamlit-expanderHeader:hover p, 
.streamlit-expanderHeader:hover span,
.streamlit-expanderHeader:hover svg {{
    color: var(--text-strong) !important;
    fill: var(--text-strong) !important;
}}
.streamlit-expanderHeader p,
.streamlit-expanderHeader span,
.streamlit-expanderHeader div,
.streamlit-expanderHeader label,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] > details > summary,
[data-testid="stExpander"] > details > summary * {{
    color: {expander_text_color} !important;
    font-weight: 600 !important;
}}
/* Arrow/chevron icon inside expander header */
.streamlit-expanderHeader svg {{
    fill: {expander_text_color} !important;
    stroke: {expander_text_color} !important;
}}
[data-testid="stExpander"] summary svg {{
    fill: {expander_text_color} !important;
    stroke: {expander_text_color} !important;
}}
.streamlit-expanderContent {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}}

/* ── DataFrames — full light/dark theming ── */
.stDataFrame {{ border: 1px solid var(--df-border) !important; border-radius: 10px !important; overflow: hidden !important; }}
.stDataFrame [data-testid="stDataFrameResizable"] {{ background: var(--df-bg) !important; }}
.stDataFrame iframe {{ background: var(--df-bg) !important; }}
.stDataFrame [class*="header"], .stDataFrame th, .stDataFrame [role="columnheader"] {{
    background: var(--df-header-bg) !important; color: var(--df-header-text) !important;
    font-weight: 700 !important; border-bottom: 1px solid var(--df-border) !important;
}}
.stDataFrame td, .stDataFrame [role="gridcell"], .stDataFrame [class*="cell"] {{
    background: var(--df-bg) !important; color: var(--df-text) !important; border-color: var(--df-border) !important;
}}
.stDataFrame tr:hover td, .stDataFrame [role="row"]:hover [role="gridcell"] {{ background: var(--df-row-hover) !important; }}
.stDataFrame ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
.stDataFrame ::-webkit-scrollbar-track {{ background: var(--df-bg); }}
.stDataFrame ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

/* ── Radio ── */
.stRadio > div {{ gap: 12px !important; }}
.stRadio label {{
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 20px !important; cursor: pointer; transition: all 0.2s;
}}
.stRadio label:hover {{ border-color: var(--accent); }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: var(--surface) !important; border-radius: 10px; gap: 20px; padding: 4px;
    border: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{ background: transparent !important; color: var(--muted) !important; border-radius: 8px; }}
.stTabs [aria-selected="true"] {{ background: #7DBCFF !important; color: white !important; }}

.js-plotly-plot {{ border-radius: 12px; overflow: hidden; }}

/* ── JSON display ── */
.stJson {{
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text) !important;
}}
</style>
""", unsafe_allow_html=True)

# ── STEP DEFINITIONS ──────────────────────────────────────────────────────────
STEPS = [
    ("Problem Type"), ("Input Data"), ("EDA"), ("Engineering"),
    ("Features"), ("Data Split"), ("Model Select"),
    ("Training"), ("Metrics"), ("Tune"),("Prediction"),
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <div class="logo-title">⚡ ML Pipeline Studio</div>
        <div class="logo-sub">Pipeline Builder</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme toggle: icon-only button (sun / moon) ────────────────────────
    col_icon, col_lbl = st.columns([1, 3])
    with col_icon:
        btn_label = "☀️" if is_dark else "🌙"
        if st.button(btn_label, key="theme_btn", help=toggle_tooltip, use_container_width=False):
            st.session_state.theme = 'light' if is_dark else 'dark'
            st.rerun()
    with col_lbl:
        mode_text = "Light Mode" if is_dark else "Dark Mode"
        st.markdown(f"<span style='color:var(--muted);font-size:13px;line-height:38px'>{mode_text}</span>",
                    unsafe_allow_html=True)

    st.markdown("<hr style='border-color:var(--border);margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("**Pipeline Steps**")

    for i, (label) in enumerate(STEPS):
        is_active = st.session_state.step == i
        is_done   = st.session_state.step > i
        if is_dark:
            a_c, d_c, m_c = "#00d4ff", "#10b981", "#64748b"
            a_bg, d_bg    = "rgba(0,212,255,0.1)", "rgba(16,185,129,0.07)"
            a_br, d_br    = "rgba(0,212,255,0.4)", "rgba(16,185,129,0.3)"
            dt_c          = "#0a0e1a"
        else:
            a_c, d_c, m_c = "#0284c7", "#059669", "#94a3b8"
            a_bg, d_bg    = "rgba(2,132,199,0.1)", "rgba(5,150,105,0.07)"
            a_br, d_br    = "rgba(2,132,199,0.4)", "rgba(5,150,105,0.3)"
            dt_c          = "#ffffff"
        color  = d_c if is_done else (a_c if is_active else m_c)
        bg     = a_bg if is_active else (d_bg if is_done else "transparent")
        border = a_br if is_active else (d_br if is_done else "transparent")
        badge  = "✓" if is_done else str(i+1)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:9px 14px;
             border-radius:8px;margin:3px 0;background:{bg};border:1px solid {border};">
            <span style="width:24px;height:24px;border-radius:50%;background:{color};
                  color:{dt_c};font-weight:700;font-size:11px;display:flex;
                  align-items:center;justify-content:center;flex-shrink:0">{badge}</span>
            <span style="font-size:13px;font-weight:{'700' if is_active else '400'};
                  color:{color}"> {label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:var(--border);margin:20px 0'>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        st.markdown("**Quick Jump**")
        jump_labels = [f"{i+1}. {lb}" for i, (lb) in enumerate(STEPS)]
        jump = st.selectbox("Go to step", jump_labels, index=st.session_state.step, label_visibility="collapsed")
        if st.button("Jump", use_container_width=True):
            st.session_state.step = int(jump.split(".")[0]) - 1
            st.rerun()
        st.markdown("<hr style='border-color:var(--border);margin:20px 0'>", unsafe_allow_html=True)

    if st.session_state.problem_type:
        st.markdown(f"<span class='chip'>{st.session_state.problem_type}</span>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        df_ = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
        st.markdown(f"<span class='chip'>{df_.shape[0]} rows × {df_.shape[1]} cols</span>", unsafe_allow_html=True)
    if st.session_state.target:
        st.markdown(f"<span class='chip'>Target: {st.session_state.target}</span>", unsafe_allow_html=True)


# ── Utilities ─────────────────────────────────────────────────────────────────
def render_hsteps():
    parts = []
    for i, (label) in enumerate(STEPS):
        if i > 0:
            cls = "done" if st.session_state.step > i-1 else ""
            parts.append(f'<div class="hstep-connector {cls}"></div>')
        dot_cls  = "done" if st.session_state.step > i else ("active" if st.session_state.step == i else "")
        lbl_cls  = dot_cls
        dot_text = "✓" if st.session_state.step > i else str(i+1)
        parts.append(f"""<div class="hstep">
            <div class="hstep-dot {dot_cls}">{dot_text}</div>
            <div class="hstep-label {lbl_cls}">{label}</div>
        </div>""")
    st.markdown(f'<div class="hsteps">{"".join(parts)}</div>', unsafe_allow_html=True)

def step_header(num, title, subtitle=""):
    st.markdown(f"""<div class="step-header">
        <div class="step-number">{num}</div>
        <div><div class="step-title">{title}</div>
        <div class="step-subtitle">{subtitle}</div></div>
    </div>""", unsafe_allow_html=True)

def nav_buttons(back=True, next_label="Continue →", next_disabled=False):
    c1, c2, c3 = st.columns([1, 6, 1])
    with c1:
        if back and st.session_state.step > 0:
            if st.button("← Back", use_container_width=True):
                st.session_state.step -= 1; st.rerun()
    with c3:
        if st.button(next_label, use_container_width=True, disabled=next_disabled):
            st.session_state.step += 1; st.rerun()

def plotly_theme():
    if is_dark:
        return dict(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.6)',
            font=dict(color='#e2e8f0', family='Outfit'),
            xaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f', color='#e2e8f0'),
            yaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f', color='#e2e8f0'),
            colorway=['#00d4ff','#7c3aed','#10b981','#2abd53','#ef4444','#ec4899'],
        )
    else:
        return dict(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,252,0.9)',
            font=dict(color='#1e293b', family='Outfit'),
            xaxis=dict(gridcolor='#c3d4e8', zerolinecolor='#c3d4e8', color='#1e293b'),
            yaxis=dict(gridcolor='#c3d4e8', zerolinecolor='#c3d4e8', color='#1e293b'),
            colorway=['#0284c7','#7c3aed','#059669','#16a34a','#dc2626','#db2777'],
        )

BAR_SCALE  = ['#1e3a5f','#00d4ff'] if is_dark else ['#bfdbfe','#0284c7']
HIST_COLOR = '#00d4ff' if is_dark else '#0284c7'

# ── Helper: render a styled DataFrame table ───────────────────────────────────
def styled_df(df_to_show, max_rows=20):
    preview = df_to_show.head(max_rows)
    header_cells = "".join(
        f"<th style='background:var(--df-header-bg);color:var(--df-header-text);"
        f"padding:8px 12px;text-align:left;font-size:12px;font-weight:700;"
        f"border-bottom:2px solid var(--df-border);white-space:nowrap'>{c}</th>"
        for c in preview.columns
    )
    rows_html = ""
    for i, (_, row) in enumerate(preview.iterrows()):
        bg = "var(--df-bg)" if i % 2 == 0 else "var(--surface2)"
        cells = "".join(
            f"<td style='padding:7px 12px;font-size:13px;color:var(--df-text);"
            f"border-bottom:1px solid var(--df-border);white-space:nowrap'>{v}</td>"
            for v in row.values
        )
        rows_html += f"<tr style='background:{bg}'>{cells}</tr>"
    table_html = f"""
    <div style="overflow-x:auto;border-radius:10px;border:1px solid var(--df-border);
                box-shadow:0 2px 8px var(--shadow);margin:8px 0">
      <table style="border-collapse:collapse;width:100%;background:var(--df-bg)">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    st.markdown(table_html, unsafe_allow_html=True)
    if len(df_to_show) > max_rows:
        st.markdown(
            f"<div style='color:var(--muted);font-size:12px;margin-top:4px'>"
            f"Showing {max_rows} of {len(df_to_show):,} rows</div>",
            unsafe_allow_html=True
        )

def detect_problem_type(df, target):
    y = df[target]

    n_unique = y.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(y)

    if not is_numeric:
        return "Classification"

    if n_unique <= 15:
        return "Classification"

    return "Regression"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Problem Type
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    render_hsteps()
    step_header("1", "Define Your Problem", "Tell the pipeline what you're trying to solve")

    st.markdown("""<div class="info-box">
        Select the <strong>type of machine learning problem</strong> you want to solve.
        This determines the models, metrics, and evaluation strategies available throughout the pipeline.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(2,132,199,0.08),rgba(2,132,199,0.02));
             border:1px solid rgba(2,132,199,0.3);border-radius:16px;padding:28px;text-align:center;margin:10px 0">
            <div style="font-size:56px"></div>
            <div style="font-size:20px;font-weight:800;color:{acc()};margin:10px 0">Classification</div>
            <div style="color:{muted()};font-size:14px">Predict discrete categories or classes.<br>
            e.g. spam detection, disease diagnosis, churn prediction</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select Classification", key="cls_btn", use_container_width=True):
            st.session_state.problem_type = "Classification"
            st.session_state.step = 1
            st.rerun()
    with col2:
        st.markdown(f"""<div style="background:linear-gradient(135deg,rgba(124,58,237,0.08),rgba(124,58,237,0.02));
             border:1px solid rgba(124,58,237,0.3);border-radius:16px;padding:28px;text-align:center;margin:10px 0">
            <div style="font-size:56px"></div>
            <div style="font-size:20px;font-weight:800;color:#7c3aed;margin:10px 0">Regression</div>
            <div style="color:{muted()};font-size:14px">Predict continuous numeric values.<br>
            e.g. house prices, stock forecasting, demand prediction</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Select Regression", key="reg_btn", use_container_width=True):
            st.session_state.problem_type = "Regression"
            st.session_state.step = 1
            st.rerun()

    

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Input Data
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    render_hsteps()
    step_header("2", "Input Data", "Upload your dataset and configure target feature")
 
    tab1, tab2 = st.tabs(["   Upload CSV   ", "   Use Sample Dataset    "])
    with tab1:
        uploaded = st.file_uploader("Drop your CSV file here", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.markdown(f'<div class="success-box">✅ Loaded <strong>{df.shape[0]:,} rows × {df.shape[1]} columns</strong></div>', unsafe_allow_html=True)
            st.toast("Dataset Loaded Successfully!", icon="🎉")
    with tab2:
        sample_ds = st.selectbox("Choose a sample dataset:", [
            "Iris (Classification)", "Breast Cancer (Classification)",
            "Boston Housing (Regression)", "Diabetes (Regression)"
        ])
        if st.button("Load Sample"):
            from sklearn import datasets
            if "Iris" in sample_ds:     d = datasets.load_iris(as_frame=True)
            elif "Breast" in sample_ds: d = datasets.load_breast_cancer(as_frame=True)
            elif "Boston" in sample_ds:
                try:
                    from sklearn.datasets import fetch_california_housing; d = fetch_california_housing(as_frame=True)
                except: d = datasets.load_diabetes(as_frame=True)
            else: d = datasets.load_diabetes(as_frame=True)
            df = d.frame; st.session_state.df = df
            st.markdown(f'<div class="success-box">✅ Loaded <strong>{df.shape[0]:,} rows × {df.shape[1]} columns</strong></div>', unsafe_allow_html=True)
            

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        num_numeric = len(df.select_dtypes(include=np.number).columns)
        num_cat     = len(df.select_dtypes(exclude=np.number).columns)
        num_missing = df.isnull().sum().sum()
        st.markdown(f"""<div class="metric-grid">
            <div class="metric-card"><div class="metric-value">{df.shape[0]:,}</div><div class="metric-label">Rows</div></div>
            <div class="metric-card"><div class="metric-value">{df.shape[1]}</div><div class="metric-label">Columns</div></div>
            <div class="metric-card"><div class="metric-value">{num_numeric}</div><div class="metric-label">Numeric</div></div>
            <div class="metric-card"><div class="metric-value">{num_cat}</div><div class="metric-label">Categorical</div></div>
            <div class="metric-card"><div class="metric-value">{num_missing:,}</div><div class="metric-label">Missing</div></div>
        </div>""", unsafe_allow_html=True)

        with st.expander("Preview Data", expanded=False):
            styled_df(df, max_rows=20)
        with st.expander("Data Types & Info", expanded=False):
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str).values,
                'Non-Null': df.notnull().sum().values,
                'Null %': (df.isnull().mean()*100).round(2).values
            })
            styled_df(dtype_df, max_rows=50)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### Target Feature Selection")
        target = st.selectbox("Select your target variable:", df.columns.tolist(), index=len(df.columns)-1)
        st.session_state.target = target
        
        # 🚨 SMART WARNING SYSTEM
        if target:
            detected_type = detect_problem_type(df, target)
            user_type = st.session_state.problem_type
        
            if user_type and detected_type != user_type:
                st.warning(
                    f"⚠️ Possible mismatch detected!\n\n"
                    f"Selected: **{user_type}**\n"
                    f"Detected: **{detected_type}**\n\n"
                    f"This may lead to poor performance."
                )
        
                # 🔁 Auto-fix button
                if st.button(f"Switch to {detected_type}", key="fix_problem_type"):
                    st.session_state.problem_type = detected_type
                    st.rerun()
        feat_cols = [c for c in df.columns if c != target]
        sel_feats = st.multiselect("Select feature columns:", feat_cols, default=feat_cols)
        st.session_state.features = sel_feats

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("###  PCA Data Shape Visualization")
        numeric_feats = df[sel_feats].select_dtypes(include=np.number).dropna()
        if len(numeric_feats.columns) >= 2:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_feats)
            n_comp = min(3, len(numeric_feats.columns))
            pca    = PCA(n_components=n_comp)
            comps  = pca.fit_transform(scaled)
            explained = pca.explained_variance_ratio_ * 100
            pca_df = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(n_comp)])
            pca_df['Target'] = df[target].astype(str).values[:len(pca_df)]
            if n_comp >= 3:
                fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Target', title="PCA — 3D Data Shape",
                                    labels={'PC1':f'PC1 ({explained[0]:.1f}%)','PC2':f'PC2 ({explained[1]:.1f}%)','PC3':f'PC3 ({explained[2]:.1f}%)'})
            else:
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target', title="PCA — 2D Data Shape",
                                 labels={'PC1':f'PC1 ({explained[0]:.1f}%)','PC2':f'PC2 ({explained[1]:.1f}%)'})
            fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)
            ev_fig = px.bar(x=[f'PC{i+1}' for i in range(n_comp)], y=explained,
                            labels={'x':'Component','y':'Explained Variance (%)'},
                            title="PCA Explained Variance", color=explained, color_continuous_scale=BAR_SCALE)
            ev_fig.update_layout(**plotly_theme()); st.plotly_chart(ev_fig, use_container_width=True)
        next_disabled = (st.session_state.target is None) or (detected_type != user_type)
        nav_buttons(back=False, next_disabled=next_disabled)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    render_hsteps()
    step_header("3", "Exploratory Data Analysis", "Understand distributions, correlations, and patterns")

    df       = st.session_state.df
    target   = st.session_state.target
    feats    = st.session_state.features or [c for c in df.columns if c != target]
    numeric_df = df[feats].select_dtypes(include=np.number)

    tab1, tab2, tab3, tab4 = st.tabs(["  Distributions  ", "  Correlations  ", "  Box Plots  ", "  Target Analysis  "])

    with tab1:
        cols_to_plot = numeric_df.columns[:12]
        n = len(cols_to_plot); rows = (n + 2) // 3
        fig = make_subplots(rows=rows, cols=3, subplot_titles=list(cols_to_plot))
        for i, col in enumerate(cols_to_plot):
            r, c = divmod(i, 3)
            fig.add_trace(go.Histogram(x=numeric_df[col].dropna(), name=col, marker_color=HIST_COLOR, opacity=0.75), row=r+1, col=c+1)
        fig.update_layout(**plotly_theme(), height=max(300, rows*260), showlegend=False, title="Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr = numeric_df.corr()
        fig  = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         title="Pearson Correlation Matrix", aspect='auto', text_auto='.2f')
        fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)
        if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            tgt_corr = numeric_df.join(df[target]).corr()[target].drop(target).sort_values(key=abs, ascending=False)
            fig2 = px.bar(x=tgt_corr.index, y=tgt_corr.values, title=f"Feature Correlation with Target '{target}'",
                          color=tgt_corr.values, color_continuous_scale='RdBu_r', labels={'x':'Feature','y':'Correlation'})
            fig2.update_layout(**plotly_theme()); st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig = go.Figure()
        for col in numeric_df.columns[:10]:
            fig.add_trace(go.Box(y=numeric_df[col].dropna(), name=col, marker_color='#7c3aed', line_color=HIST_COLOR))
        fig.update_layout(**plotly_theme(), title="Feature Box Plots", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if pd.api.types.is_numeric_dtype(df[target]):
            fig = px.histogram(df, x=target, nbins=40, color_discrete_sequence=[HIST_COLOR], title=f"Distribution of {target}")
            fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)
            desc_df = df[target].describe().reset_index().rename(columns={'index':'Statistic', target:'Value'})
            styled_df(desc_df)
        else:
            vc = df[target].value_counts().reset_index(); vc.columns = ['Class', 'Count']
            fig = px.bar(vc, x='Class', y='Count', color='Count', color_continuous_scale='Blues', title=f"Class Distribution of {target}")
            fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Data Engineering
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    render_hsteps()
    step_header("4", "Data Engineering & Cleaning", "Handle missing values and detect/remove outliers")

    df_work      = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df.copy()
    target       = st.session_state.target
    feats        = st.session_state.features or [c for c in df_work.columns if c != target]
    numeric_feats = df_work[feats].select_dtypes(include=np.number).columns.tolist()

    # ── Expander labels styled inline so colour is 100% reliable ─────────
    exp_label_style = f"color:{expander_text_color};font-weight:600"

    with st.expander(" Missing Value Imputation", expanded=True):
        missing = df_work[numeric_feats].isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.markdown('<div class="success-box">✅ No missing values in numeric features!</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warn-box">⚠️ Found missing values in {len(missing)} feature(s)</div>', unsafe_allow_html=True)
            miss_df = missing.reset_index().rename(columns={'index':'Feature', 0:'Missing Count'})
            styled_df(miss_df)
        imp_method = st.selectbox("Imputation Method:", ["Mean", "Median", "Mode", "Zero"])
        if st.button("Apply Imputation"):
            for col in numeric_feats:
                if df_work[col].isnull().any():
                    v = {'Mean': df_work[col].mean(), 'Median': df_work[col].median(),
                         'Mode': df_work[col].mode()[0], 'Zero': 0}[imp_method]
                    df_work[col].fillna(v, inplace=True)
            st.session_state.df_clean = df_work
            st.markdown('<div class="success-box">✅ Imputation applied!</div>', unsafe_allow_html=True)
            st.toast("Missing Values handled!")

    with st.expander(" Outlier Detection", expanded=True):
        method = st.selectbox("Detection Method:", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        st.session_state.outlier_method = method
        if st.button(" Detect Outliers"):
            X    = df_work[numeric_feats].dropna()
            mask = pd.Series([False]*len(df_work), index=df_work.index)
            if method == "IQR":
                for col in numeric_feats:
                    Q1, Q3 = df_work[col].quantile(0.25), df_work[col].quantile(0.75)
                    IQR    = Q3 - Q1
                    mask   = mask | ((df_work[col] < Q1-1.5*IQR) | (df_work[col] > Q3+1.5*IQR))
            else:
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler(); Xs = sc.fit_transform(X)
                if method == "Isolation Forest":
                    from sklearn.ensemble import IsolationForest
                    preds = IsolationForest(contamination=0.05, random_state=42).fit_predict(Xs)
                    mask.iloc[X.index] = (preds == -1)
                elif method == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(Xs)
                    mask.iloc[X.index] = (labels == -1)
                else:
                    from sklearn.cluster import OPTICS
                    labels = OPTICS(min_samples=5).fit_predict(Xs)
                    mask.iloc[X.index] = (labels == -1)

            n_out = mask.sum()
            st.session_state.outlier_mask  = mask
            st.session_state.outlier_count = int(n_out)
            st.session_state.df_clean      = df_work
            pct = 100 * n_out / len(df_work)
            if n_out == 0:
                st.markdown('<div class="success-box">✅ No outliers detected!</div>', unsafe_allow_html=True)
            else:
                st.toast("Outlier Detected!", icon="⚠️")
                st.markdown(f'<div class="warn-box">⚠️ Detected <strong>{n_out}</strong> outliers ({pct:.1f}% of data)</div>', unsafe_allow_html=True)
                from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
                Xn = df_work[numeric_feats].fillna(df_work[numeric_feats].mean())
                sc = StandardScaler(); Xs = sc.fit_transform(Xn)
                pc = PCA(n_components=2).fit_transform(Xs)
                vis_df = pd.DataFrame({'PC1':pc[:,0],'PC2':pc[:,1],
                                       'Outlier':mask.astype(str).map({'True':'Outlier','False':'Normal'})})
                fig = px.scatter(vis_df, x='PC1', y='PC2', color='Outlier',
                                 color_discrete_map={'Outlier':'#ef4444','Normal':HIST_COLOR},
                                 title=f"Outlier Visualization ({method})")
                fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)
                if st.button(" Remove Outliers", type="primary"):
                    st.toast("Outlier Removed!")
                    df_work = df_work[~mask].reset_index(drop=True)
                    st.session_state.df_clean  = df_work
                    st.session_state.outlier_mask = None
                    st.markdown(f'<div class="success-box">✅ Outliers Removed {n_out} rows. New shape: {df_work.shape}</div>', unsafe_allow_html=True)
                    

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Feature Selection
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    render_hsteps()
    step_header("5", "Feature Selection", "Identify the most informative features for your model")

    df    = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    target = st.session_state.target
    feats  = st.session_state.features or [c for c in df.columns if c != target]
    numeric_feats = df[feats].select_dtypes(include=np.number).columns.tolist()
    Xdf = df[numeric_feats].fillna(df[numeric_feats].mean())
    ydf = df[target]

    method = st.selectbox("Feature Selection Method:", ["Variance Threshold", "Correlation with Target", "Information Gain (MI)"])

    if method == "Variance Threshold":
        threshold = st.slider("Variance Threshold:", 0.0, 1.0, 0.01, 0.01)
        from sklearn.feature_selection import VarianceThreshold
        variances = Xdf.var()
        if (variances < threshold).all():
            st.markdown(f'<div class="danger-box">❌ No features meet threshold <b>{threshold}</b>. Lower it.</div>', unsafe_allow_html=True)
            selected = numeric_feats; results = variances.sort_values(ascending=False)
        else:
            sel = VarianceThreshold(threshold=threshold); sel.fit(Xdf)
            variances = pd.Series(sel.variances_, index=numeric_feats)
            results   = variances.sort_values(ascending=False)
            selected  = variances[variances >= threshold].index.tolist()
        fig = px.bar(x=results.index, y=results.values, color=results.values, color_continuous_scale=BAR_SCALE,
                     title="Feature Variances", labels={'x':'Feature','y':'Variance'})
        fig.add_hline(y=threshold, line_dash='dash', line_color='#ef4444', annotation_text=f"Threshold={threshold}")
        fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)

    elif method == "Correlation with Target":
        threshold = st.slider("Min |Correlation|:", 0.0, 1.0, 0.1, 0.05)
        if pd.api.types.is_numeric_dtype(ydf):
            corrs    = Xdf.corrwith(ydf).abs().sort_values(ascending=False)
            selected = corrs[corrs >= threshold].index.tolist()
            fig = px.bar(x=corrs.index, y=corrs.values, color=corrs.values, color_continuous_scale=BAR_SCALE,
                         title=f"|Correlation| with {target}", labels={'x':'Feature','y':'|Correlation|'})
            fig.add_hline(y=threshold, line_dash='dash', line_color='#ef4444')
            fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="warn-box">Target must be numeric. Using MI instead.</div>', unsafe_allow_html=True)
            method = "Information Gain (MI)"

    if method == "Information Gain (MI)":
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        threshold = st.slider("Min Mutual Information:", 0.0, 1.0, 0.05, 0.01)
        mi = (mutual_info_classif if st.session_state.problem_type == "Classification" else mutual_info_regression)(
            Xdf, ydf if st.session_state.problem_type != "Classification" else ydf.astype(str), random_state=42)
        mi_s     = pd.Series(mi, index=numeric_feats).sort_values(ascending=False)
        selected = mi_s[mi_s >= threshold].index.tolist()
        fig = px.bar(x=mi_s.index, y=mi_s.values, color=mi_s.values, color_continuous_scale='Viridis',
                     title="Mutual Information with Target", labels={'x':'Feature','y':'Mutual Information'})
        fig.add_hline(y=threshold, line_dash='dash', line_color='#ef4444')
        fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'<div class="info-box">✅ Selected <strong>{len(selected)}</strong> features out of {len(numeric_feats)} using {method}</div>', unsafe_allow_html=True)
    final_sel = st.multiselect("Confirm / adjust selected features:", numeric_feats, default=selected)
    st.session_state.selected_features = final_sel
    nav_buttons(next_disabled=len(final_sel)==0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Data Split
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    render_hsteps()
    step_header("6", "Train / Test Split", "Partition data for unbiased model evaluation")

    df        = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    target    = st.session_state.target
    sel_feats = st.session_state.selected_features or st.session_state.features

    col1, col2 = st.columns(2)
    with col1: test_size    = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05)
    with col2: random_state = st.number_input("Random Seed:", value=42, step=1)
    stratify_opt = False
    if st.session_state.problem_type == "Classification":
        stratify_opt = st.checkbox("Stratified Split (recommended for classification)", value=True)

    train_pct, test_pct = int((1-test_size)*100), int(test_size*100)
    fig = go.Figure(go.Bar(x=[train_pct, test_pct], y=['Split'], orientation='h',
                           marker_color=[acc(), '#7c3aed'],
                           text=[f'Train {train_pct}%', f'Test {test_pct}%'],
                           textposition='inside', textfont=dict(color='white', size=13)))
    theme = plotly_theme(); theme['xaxis'].update({'range':[0,100],'showticklabels':False})
    fig.update_layout(**theme, height=100, showlegend=False, title="Train/Test Split Ratio", margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    if st.button(" Apply Split", type="primary"):
        from sklearn.model_selection import train_test_split
        Xdf = df[sel_feats].fillna(df[sel_feats].mean()); ydf = df[target]
        stratify = ydf if (stratify_opt and st.session_state.problem_type == "Classification") else None
        X_train, X_test, y_train, y_test = train_test_split(
            Xdf, ydf, test_size=test_size, random_state=int(random_state), stratify=stratify)
        st.session_state.X_train, st.session_state.X_test = X_train, X_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        st.markdown(f"""<div class="metric-grid">
            <div class="metric-card"><div class="metric-value">{len(X_train):,}</div><div class="metric-label">Train Samples</div></div>
            <div class="metric-card"><div class="metric-value">{len(X_test):,}</div><div class="metric-label">Test Samples</div></div>
            <div class="metric-card"><div class="metric-value">{len(sel_feats)}</div><div class="metric-label">Features</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="success-box">✅ Data split complete!</div>', unsafe_allow_html=True)
        st.toast("Data split complete!")

    nav_buttons(next_disabled=st.session_state.X_train is None)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Model Selection
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 6:
    render_hsteps()
    step_header("7", "Model Selection", "Choose and configure your machine learning algorithm")

    pt = st.session_state.problem_type
    models_avail = (["Logistic Regression", "SVM (Classifier)", "Random Forest Classifier", "K-Nearest Neighbors"]
                    if pt == "Classification" else
                    ["Linear Regression", "SVM (Regressor)", "Random Forest Regressor", "Ridge Regression"])
    model_choice = st.radio("Select Models: ",models_avail, horizontal=False)
    st.session_state.model_name = model_choice

    params = {}
    st.markdown("####  Model Hyperparameters")
    if "SVM" in model_choice:
        c1, c2 = st.columns(2)
        with c1: kernel = st.selectbox("Kernel:", ["rbf","linear","poly","sigmoid"]); params['kernel'] = kernel
        with c2: C = st.number_input("C:", 0.01, 100.0, 1.0); params['C'] = C
        if kernel == "poly": params['degree'] = st.slider("Degree:", 2, 8, 3)
    elif "Random Forest" in model_choice:
        c1, c2 = st.columns(2)
        with c1: params['n_estimators'] = st.slider("Trees:", 10, 500, 100, 10)
        with c2: params['max_depth']    = st.selectbox("Max Depth:", [None,5,10,15,20,30])
    elif "Logistic" in model_choice:
        params['solver']   = st.selectbox("Solver:", ["lbfgs","liblinear","saga"])
        params['max_iter'] = 1000
    elif "Ridge" in model_choice:
        params['alpha'] = st.number_input("Alpha:", 0.01, 100.0, 1.0)
    elif "K-Nearest" in model_choice:
        params['n_neighbors'] = st.slider("k:", 1, 30, 5)
    st.session_state.model_params = params

    model_info = {
        "Linear Regression":        ("Fits a linear relationship. Fast, interpretable, assumes linearity."),
        "Logistic Regression":      ("Predicts class probabilities via sigmoid. Great classification baseline."),
        "SVM (Classifier)":         ("Optimal hyperplane separator. Powerful with kernel trick."),
        "SVM (Regressor)":          ("SVR within ε-margin. Robust to outliers."),
        "Random Forest Classifier": ("Ensemble of trees. Handles non-linearity, reduces overfitting."),
        "Random Forest Regressor":  ("Ensemble for regression. Robust and powerful."),
        "K-Nearest Neighbors":      ("k nearest points classification. Non-parametric, instance-based."),
        "Ridge Regression":         ("Linear + L2 regularization. Handles multicollinearity."),
    }
    if model_choice in model_info:
        desc = model_info[model_choice]
        card_bg = "rgba(0,212,255,0.06)" if is_dark else "rgba(2,132,199,0.06)"
        card_br = "rgba(0,212,255,0.2)"  if is_dark else "rgba(2,132,199,0.2)"
        st.markdown(f"""<div style="background:{card_bg};border:1px solid {card_br};border-radius:12px;padding:20px;margin:16px 0">
            <div style="font-size:28px"></div>
            <div style="font-weight:700;margin:8px 0;color:{text_strong()}">{model_choice}</div>
            <div style="color:{muted()};font-size:14px">{desc}</div>
        </div>""", unsafe_allow_html=True)
    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Training + KFold
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 7:
    render_hsteps()
    step_header("8", "Model Training & K-Fold Validation", "Train with cross-validation for robust evaluation")

    k = st.slider("K (number of folds):", 3, 20, 5)
    st.session_state.k_folds = k
    st.markdown("""<div class="info-box">
        📖 K-Fold cross-validation splits training data into K folds, training K times (each time using a different fold as validation).
    </div>""", unsafe_allow_html=True)

    fold_inactive = '#1e3a5f' if is_dark else '#dbeafe'
    fig = go.Figure()
    for fold in range(k):
        for j in range(k):
            fig.add_shape(type="rect", x0=j, x1=j+1, y0=fold-0.4, y1=fold+0.4,
                          line=dict(color='#0a0e1a' if is_dark else '#f0f4f8', width=1),
                          fillcolor='#7c3aed' if j == fold else fold_inactive)
        fig.add_annotation(x=fold+0.5, y=fold, text="Val", showarrow=False, font=dict(color='white', size=10))
    theme = plotly_theme()
    theme['xaxis'].update({'title':"Fold Index",'tickmode':'array','tickvals':list(range(k)),'ticktext':[f"F{i+1}" for i in range(k)]})
    theme['yaxis'].update({'title':"Iteration",'tickmode':'array','tickvals':list(range(k)),'ticktext':[f"Iter {i+1}" for i in range(k)]})
    fig.update_layout(**theme, height=max(200, k*30+80), title=f"{k}-Fold CV Schema")
    st.plotly_chart(fig, use_container_width=True)

    if st.button(" Train Model", type="primary", use_container_width=True):
        from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        import sklearn.linear_model as lm, sklearn.svm as svm_mod
        import sklearn.ensemble as ens, sklearn.neighbors as knn_mod

        pt = st.session_state.problem_type; mn = st.session_state.model_name; mp = st.session_state.model_params
        X_train = st.session_state.X_train; y_train = st.session_state.y_train
        X_test  = st.session_state.X_test;  y_test  = st.session_state.y_test

        if mn == "Linear Regression": model = lm.LinearRegression()
        elif mn == "Ridge Regression": model = lm.Ridge(**mp)
        elif mn == "Logistic Regression": model = lm.LogisticRegression(**mp)
        elif mn == "SVM (Classifier)": model = svm_mod.SVC(**mp, probability=True)
        elif mn == "SVM (Regressor)": model = svm_mod.SVR(**mp)
        elif mn == "Random Forest Classifier": model = ens.RandomForestClassifier(**mp, random_state=42)
        elif mn == "Random Forest Regressor": model = ens.RandomForestRegressor(**mp, random_state=42)
        elif mn == "K-Nearest Neighbors": model = knn_mod.KNeighborsClassifier(**mp)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)

        # KFold CV
        scoring = 'accuracy' if pt == 'Classification' else 'neg_mean_squared_error'
        cv = StratifiedKFold(n_splits=k) if pt == 'Classification' else KFold(n_splits=k, shuffle=True, random_state=42)


        with st.spinner("Training..."):
            cv_scores = cross_val_score(model, X_tr_s, y_train, cv=cv, scoring=scoring)
            model.fit(X_tr_s, y_train)

        y_pred = model.predict(X_te_s)
        st.session_state.model = model; st.session_state.scaler = sc
        st.session_state.y_pred = y_pred; st.session_state.cv_scores = cv_scores

        cv_display = cv_scores*100 if pt == "Classification" else np.sqrt(-cv_scores)
        unit = "Accuracy (%)" if pt == "Classification" else "RMSE"
        st.markdown(f"""<div class="metric-grid">
            <div class="metric-card"><div class="metric-value">{cv_display.mean():.2f}</div><div class="metric-label">Mean CV {unit}</div></div>
            <div class="metric-card"><div class="metric-value">{cv_display.std():.2f}</div><div class="metric-label">CV Std Dev</div></div>
            <div class="metric-card"><div class="metric-value">{cv_display.max():.2f}</div><div class="metric-label">Best Fold</div></div>
            <div class="metric-card"><div class="metric-value">{cv_display.min():.2f}</div><div class="metric-label">Worst Fold</div></div>
        </div>""", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(k)], y=cv_display, marker_color=[HIST_COLOR]*k, name='CV Score'))
        fig.add_hline(y=cv_display.mean(), line_dash='dash', line_color=acc3(), annotation_text=f"Mean={cv_display.mean():.2f}")
        fig.update_layout(**plotly_theme(), title=f"K-Fold CV Scores ({k} folds)", yaxis_title=unit, xaxis_title="Fold")
        st.plotly_chart(fig, use_container_width=True)
        st.balloons()
        st.toast("Model training complete!")
        st.markdown('<div class="success-box">✅ Training complete!</div>', unsafe_allow_html=True)

    nav_buttons(next_disabled=st.session_state.model is None)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Performance Metrics
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 8:
    render_hsteps()
    step_header("9", "Performance Metrics", "Evaluate model accuracy, check for over/underfitting")

    pt      = st.session_state.problem_type; model  = st.session_state.model
    y_test  = st.session_state.y_test;       y_pred = getattr(st.session_state, 'y_pred', None)
    y_train = st.session_state.y_train;      X_train = st.session_state.X_train; X_test = st.session_state.X_test
    cv_scores = getattr(st.session_state, 'cv_scores', None); sc = getattr(st.session_state, 'scaler', None)

    if model is None or y_pred is None:
        st.markdown('<div class="warn-box">⚠️ Complete model training (Step 7) first.</div>', unsafe_allow_html=True)
    else:
        X_tr_s    = sc.transform(X_train) if sc else X_train
        X_te_s    = sc.transform(X_test)  if sc else X_test
        train_pred = model.predict(X_tr_s)

        if pt == "Classification":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            train_acc = accuracy_score(y_train, train_pred); test_acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
            f1   = f1_score(y_test, y_pred,        average='weighted', zero_division=0)
            st.markdown(f"""<div class="metric-grid">
                <div class="metric-card"><div class="metric-value">{test_acc*100:.1f}%</div><div class="metric-label">Test Accuracy</div></div>
                <div class="metric-card"><div class="metric-value">{train_acc*100:.1f}%</div><div class="metric-label">Train Accuracy</div></div>
                <div class="metric-card"><div class="metric-value">{prec*100:.1f}%</div><div class="metric-label">Precision</div></div>
                <div class="metric-card"><div class="metric-value">{rec*100:.1f}%</div><div class="metric-label">Recall</div></div>
                <div class="metric-card"><div class="metric-value">{f1:.3f}</div><div class="metric-label">F1 Score</div></div>
            </div>""", unsafe_allow_html=True)
            gap = train_acc - test_acc; cv_mean = cv_scores.mean() if cv_scores is not None else test_acc
            if train_acc < 0.7:
                st.markdown('<div class="warn-box">⚠️ <strong>Possible Underfitting</strong>: Low train accuracy.</div>', unsafe_allow_html=True)
            elif gap > 0.15:
                st.markdown('<div class="danger-box">🔴 <strong>Overfitting</strong>: Large train/test gap.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ <strong>Good Fit</strong>: Train and test accuracy are close.</div>', unsafe_allow_html=True)

            fig = go.Figure(go.Bar(x=['Train','CV Mean','Test'], y=[train_acc*100,cv_mean*100,test_acc*100],
                                   marker_color=[acc(),'#7c3aed',acc3()],
                                   text=[f'{v*100:.1f}%' for v in [train_acc,cv_mean,test_acc]],
                                   textposition='outside', textfont=dict(color=text_col())))
            fig.update_layout(**plotly_theme(), title="Train / CV / Test Accuracy", yaxis_title="Accuracy (%)", yaxis_range=[0,115])
            st.plotly_chart(fig, use_container_width=True)

            cm = confusion_matrix(y_test, y_pred); labels = sorted(y_test.unique().tolist())
            fig2 = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale='Blues',
                             title="Confusion Matrix", labels=dict(x='Predicted', y='Actual'))
            fig2.update_layout(**plotly_theme()); st.plotly_chart(fig2, use_container_width=True)

        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            train_r2 = r2_score(y_train, train_pred); test_r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred)); mae = mean_absolute_error(y_test, y_pred)
            st.markdown(f"""<div class="metric-grid">
                <div class="metric-card"><div class="metric-value">{test_r2:.3f}</div><div class="metric-label">Test R²</div></div>
                <div class="metric-card"><div class="metric-value">{train_r2:.3f}</div><div class="metric-label">Train R²</div></div>
                <div class="metric-card"><div class="metric-value">{rmse:.3f}</div><div class="metric-label">RMSE</div></div>
                <div class="metric-card"><div class="metric-value">{mae:.3f}</div><div class="metric-label">MAE</div></div>
            </div>""", unsafe_allow_html=True)
            gap = train_r2 - test_r2
            if test_r2 < 0.4:
                st.markdown('<div class="warn-box">⚠️ <strong>Possible Underfitting</strong>: Low R².</div>', unsafe_allow_html=True)
            elif gap > 0.2:
                st.markdown('<div class="danger-box">🔴 <strong>Overfitting</strong>: Large R² gap.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ <strong>Good Fit</strong>: Model generalizes well.</div>', unsafe_allow_html=True)

            fig = px.scatter(x=y_test.values, y=y_pred, opacity=0.7, labels={'x':'Actual','y':'Predicted'},
                             title="Actual vs Predicted", color_discrete_sequence=[HIST_COLOR])
            mn_v, mx_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig.add_shape(type='line', x0=mn_v, x1=mx_v, y0=mn_v, y1=mx_v, line=dict(color=acc3(), dash='dash', width=2))
            fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)

            residuals = y_test.values - y_pred
            fig2 = px.scatter(x=y_pred, y=residuals, opacity=0.7, labels={'x':'Predicted','y':'Residual'},
                              title="Residual Plot", color_discrete_sequence=['#7c3aed'])
            fig2.add_hline(y=0, line_color='#ef4444', line_dash='dash')
            fig2.update_layout(**plotly_theme()); st.plotly_chart(fig2, use_container_width=True)

    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 9:
    render_hsteps()
    step_header("10", "Hyperparameter Tuning", "Optimize model performance with automated search")

    mn      = st.session_state.model_name; pt = st.session_state.problem_type
    X_train = st.session_state.X_train;   y_train = st.session_state.y_train
    X_test  = st.session_state.X_test;    y_test  = st.session_state.y_test
    sc      = getattr(st.session_state, 'scaler', None)

    if X_train is None:
        st.markdown('<div class="warn-box">⚠️ Complete previous steps first.</div>', unsafe_allow_html=True)
    else:
        X_tr_s = sc.transform(X_train) if sc else X_train
        X_te_s = sc.transform(X_test)  if sc else X_test

        search_type = st.radio("Search Method:", ["Grid Search", "Random Search"], horizontal=True)
        n_iter = st.slider("Iterations (Random Search):", 5, 100, 20) if search_type == "Random Search" else None
        cv_k   = st.slider("CV Folds for tuning:", 3, 10, 5)
        st.markdown("#### 🔧 Parameter Grid")

        import sklearn.linear_model as lm, sklearn.svm as svm_mod
        import sklearn.ensemble as ens, sklearn.neighbors as knn_mod

        param_map = {
            "Linear Regression":        (lm.LinearRegression(),                    {'fit_intercept': [True, False]}),
            "Ridge Regression":         (lm.Ridge(),                               {'alpha': [0.01,0.1,1.0,10.0,100.0]}),
            "Logistic Regression":      (lm.LogisticRegression(max_iter=1000),     {'C': [0.01,0.1,1,10,100],'solver':['lbfgs','liblinear']}),
            "SVM (Classifier)":         (svm_mod.SVC(probability=True),            {'C':[0.1,1,10,100],'kernel':['rbf','linear','poly'],'gamma':['scale','auto']}),
            "SVM (Regressor)":          (svm_mod.SVR(),                            {'C':[0.1,1,10,100],'kernel':['rbf','linear'],'epsilon':[0.01,0.1,0.5]}),
            "Random Forest Classifier": (ens.RandomForestClassifier(random_state=42), {'n_estimators':[50,100,200],'max_depth':[None,5,10,20],'min_samples_split':[2,5,10]}),
            "Random Forest Regressor":  (ens.RandomForestRegressor(random_state=42),  {'n_estimators':[50,100,200],'max_depth':[None,5,10,20],'min_samples_split':[2,5,10]}),
            "K-Nearest Neighbors":      (knn_mod.KNeighborsClassifier(),           {'n_neighbors':[3,5,7,9,11,15],'weights':['uniform','distance'],'metric':['euclidean','manhattan']}),
        }
        model_base, param_grids = param_map[mn]
        st.json(param_grids)
        scoring = 'accuracy' if pt == 'Classification' else 'neg_mean_squared_error'

        if st.button(" Run Hyperparameter Search", type="primary"):
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
            with st.spinner("Searching..."):
                searcher = (GridSearchCV(model_base, param_grids, cv=cv_k, scoring=scoring, n_jobs=-1)
                            if search_type == "Grid Search" else
                            RandomizedSearchCV(model_base, param_grids, n_iter=n_iter or 20, cv=cv_k,
                                               scoring=scoring, n_jobs=-1, random_state=42))
                searcher.fit(X_tr_s, y_train)

            best_model  = searcher.best_estimator_; best_params = searcher.best_params_
            results_df  = pd.DataFrame(searcher.cv_results_).sort_values('rank_test_score')
            y_pred_old  = st.session_state.get('y_pred')
            y_pred_new  = best_model.predict(X_te_s)

            if pt == "Classification":
                from sklearn.metrics import accuracy_score
                old_score   = accuracy_score(y_test, y_pred_old) if y_pred_old is not None else 0
                new_score   = accuracy_score(y_test, y_pred_new)
                metric_name = "Accuracy"; improvement = (new_score-old_score)*100
                old_display = f"{old_score*100:.2f}%"; new_display = f"{new_score*100:.2f}%"
            else:
                from sklearn.metrics import r2_score
                old_score   = r2_score(y_test, y_pred_old) if y_pred_old is not None else 0
                new_score   = r2_score(y_test, y_pred_new)
                metric_name = "R²"; improvement = new_score-old_score
                old_display = f"{old_score:.4f}"; new_display = f"{new_score:.4f}"

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Previous {metric_name}", old_display)
            c2.metric(f"Tuned {metric_name}", new_display,
                      delta=f"+{improvement:.2f}{'%' if pt=='Classification' else ''}" if improvement > 0
                      else f"{improvement:.2f}")
            c3.metric("Best Params Found", str(len(best_params)))

            st.markdown("#### 🏆 Best Hyperparameters")
            for k_p, v_p in best_params.items():
                st.markdown(f"<span class='chip'>{k_p} = {v_p}</span>", unsafe_allow_html=True)

            st.markdown("#### Top Results")
            disp_cols = [c for c in ['rank_test_score','mean_test_score','std_test_score','params'] if c in results_df.columns]
            styled_df(results_df[disp_cols].head(10))

            if 'mean_test_score' in results_df.columns:
                fig = px.histogram(results_df, x='mean_test_score', nbins=20,
                                   color_discrete_sequence=['#7c3aed'], title="CV Score Distribution")
                fig.add_vline(x=searcher.best_score_, line_color=acc3(), line_dash='dash', annotation_text="Best")
                fig.update_layout(**plotly_theme()); st.plotly_chart(fig, use_container_width=True)

            st.session_state.model = best_model; st.session_state.y_pred = y_pred_new
            st.markdown('<div class="success-box">✅ Model updated! Go back to Step 8 to see updated metrics.</div>', unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("###  Pipeline Complete!")
    st.toast("Pipeline Completed Successfully!")
    st.snow()
    finish_bg = f"linear-gradient(135deg,rgba(16,185,129,0.1),rgba({'0,212,255' if is_dark else '2,132,199'},0.05))"
    st.markdown(f"""<div style="background:{finish_bg};border:1px solid rgba(16,185,129,0.3);
         border-radius:16px;padding:28px;text-align:center">
        <div style="font-size:48px">🏆</div>
        <div style="font-size:22px;font-weight:800;color:{acc3()};margin:12px 0">Pipeline Successfully Completed</div>
        <div style="color:{muted()}">Problem: {st.session_state.problem_type} · Model: {st.session_state.model_name}</div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Make Predictions", use_container_width=True):
            st.session_state.step=10;
            st.rerun()
    with col2:
        if st.button(" Back to Metrics", use_container_width=True):
            st.session_state.step = 8; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 10:
    render_hsteps()
    step_header("11", "Prediction Panel", "Use trained model for real-time predictions")

    model = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    features = st.session_state.get("selected_features")

    if model is None:
        st.markdown('<div class="warn-box">⚠️ Train a model first (Step 8)</div>', unsafe_allow_html=True)
    else:
        st.markdown("### Enter Feature Values")

        input_data = {}
        cols = st.columns(3)

        for i, col in enumerate(features):
            with cols[i % 3]:
                input_data[col] = st.number_input(f"{col}", value=0.0)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Predict", use_container_width=True):
            input_df = pd.DataFrame([input_data])

            try:
                if scaler is not None:
                    input_scaled = scaler.transform(input_df)
                else:
                    input_scaled = input_df

                pred = model.predict(input_scaled)[0]

                st.markdown(f"""
                <div class="success-box" style="text-align:center;font-size:20px">
                    Prediction Result: <strong>{pred}</strong>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button(" Restart Pipeline", use_container_width=True):
            theme_backup = st.session_state.theme

            # Clear everything
            for key in list(st.session_state.keys()):
                del st.session_state[key]

            # Restore theme only
            st.session_state.theme = theme_backup

            st.rerun()

    with col2:
        if st.button("📊 Back to Metrics", use_container_width=True):
            st.session_state.step = 8
            st.rerun()

    
