"""Visual system: compact operations dashboard with readable evidence cards."""
from html import escape
import streamlit as st

CSS="""
<style>
:root {--ink:#e7eff9;--muted:#91a5bd;--cyan:#59e1c5;--line:#22324a;--panel:#111f32}
.stApp {background:radial-gradient(ellipse at 95% 0%,#14332f55,transparent 40%),#0a1322;color:var(--ink)}
.block-container {max-width:1560px;padding:2rem 2.8rem 3rem}
header[data-testid="stHeader"] {background:transparent}
[data-testid="stSidebar"] {background:#0d1829;border-right:1px solid #1e2d43}
[data-testid="stSidebar"]>div:first-child {padding-top:1.3rem}
h1,h2,h3 {font-family:'Segoe UI',sans-serif;letter-spacing:-.035em}
h1 {font-size:2.7rem!important;font-weight:750!important;line-height:1.13!important}
h2 {font-size:1.35rem!important;font-weight:650!important}
h3 {font-size:1.06rem!important;font-weight:650!important}
p,li {line-height:1.6}
[data-testid="stCaptionContainer"] {color:#91a5bd}
[data-testid="stMetric"] {background:linear-gradient(150deg,#14243a,#101d30);border:1px solid #25364d;border-radius:15px;padding:18px 20px;min-height:116px}
[data-testid="stMetricLabel"] {color:#9bafc7;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em}
[data-testid="stMetricValue"] {font-size:2rem;font-weight:650;letter-spacing:-.045em}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px}
button[kind="primary"] {background:#63e4c5!important;color:#08221d!important;font-weight:700!important;border:0!important}
button {border-radius:9px!important}
.vp-logo {display:flex;gap:12px;align-items:center;margin:0 0 1.8rem;padding:0 4px}
.vp-mark {width:42px;height:48px;background:#61e5c6;color:#09291f;display:grid;place-items:center;font-size:24px;font-weight:900;clip-path:polygon(0 0,100% 0,100% 78%,50% 100%,0 78%)}
.vp-wordmark {font-size:17px;line-height:1.15;font-weight:800;letter-spacing:.10em}
.vp-wordmark small {font-size:9px;color:#90a5bd;letter-spacing:.18em;font-weight:500}
.vp-eyebrow {font-size:11px;font-weight:650;letter-spacing:.18em;color:#63e4c5;text-transform:uppercase;margin:8px 0 12px}
.vp-subtitle {max-width:760px;color:#9aadc4;font-size:15px;margin-top:0;margin-bottom:22px;line-height:1.7}
.vp-topline {display:flex;align-items:center;justify-content:space-between;color:#91a5bd;font-size:11px;letter-spacing:.08em;margin:0 0 1.5rem;border-bottom:1px solid #213149;padding-bottom:14px}
.vp-pill {display:inline-flex;align-items:center;gap:6px;border:1px solid #2c4b48;color:#86e4ce;background:#14373166;border-radius:30px;padding:5px 11px;font-size:11px;letter-spacing:.02em;white-space:nowrap}
.vp-pill.amber {background:#45331855;border-color:#72552b;color:#ffd58b}
.vp-brief {border:1px solid #335957;border-radius:17px;background:linear-gradient(115deg,#153732aa,#112c3455);padding:21px 24px;margin:14px 0 20px}
.vp-brief h3 {margin:0 0 7px;color:#e3fff6;font-size:1.2rem!important}
.vp-brief p {color:#a6c5c2;margin:0;font-size:13px}
.vp-signal {border-left:3px solid #60dfc4;background:#12293a;border-radius:0 9px 9px 0;padding:12px 16px;margin-bottom:9px;color:#d6e6f2;font-size:13px}
.vp-note {border:1px solid #3c3d38;background:#2c2b2144;padding:14px 17px;border-radius:11px;color:#c9bda2;font-size:12px;line-height:1.7;margin:12px 0}
.vp-foot {font-size:10px;line-height:1.8;color:#7e93ac;border-top:1px solid #223149;margin-top:24px;padding-top:16px}
.vp-kicker {color:#90a7bf;font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin:20px 0 8px}
.vp-detail {font-size:12px;color:#b3c4d9;line-height:1.9}
[data-testid="stSidebar"] label {font-size:13px}
[data-testid="stDataFrame"] {border:1px solid #26364d;border-radius:12px;overflow:hidden}
[data-baseweb="tab-list"] {gap:12px;background:transparent;border-bottom:1px solid #25364d}
[data-baseweb="tab"] {font-size:13px;padding:10px 2px}
@media(max-width:800px){.block-container{padding:1.3rem 1rem 2rem}h1{font-size:2rem!important}.vp-topline{letter-spacing:0}.vp-brief{padding:16px}}
</style>
"""


def install():st.markdown(CSS,unsafe_allow_html=True)


def heading(number,title,description):
    st.markdown(f'<div class="vp-eyebrow">{escape(number)} / INTELIGENCIA DE INSPECCIÓN</div>',unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="vp-subtitle">{escape(description)}</div>',unsafe_allow_html=True)


def brief(title,description):
    st.markdown(f'<div class="vp-brief"><h3>{escape(title)}</h3><p>{escape(description)}</p></div>',unsafe_allow_html=True)


def signal(text):st.markdown(f'<div class="vp-signal">{escape(str(text))}</div>',unsafe_allow_html=True)


def note(text):st.markdown(f'<div class="vp-note">{escape(str(text))}</div>',unsafe_allow_html=True)


def metric(label,value,help=None):st.metric(label,value,help=help)
