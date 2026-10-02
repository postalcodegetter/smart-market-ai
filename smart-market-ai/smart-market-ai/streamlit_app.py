"""
SMART MARKET AI - Professional Trading Terminal (Streamlit front-end)

Educational/informational only. Not financial advice.

This file is the ONLY UI file in the app. It calls into smart_market_ai_ml.py
(unchanged) for every piece of data and never re-implements backend logic -
it only formats, charts, and lays out what the backend already returns.
"""

from pathlib import Path
import io
import json
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import smart_market_ai_ml as ai

# =============================================================================
# CONSTANTS
# =============================================================================

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "market_model.joblib"
PAPER_TRADES_FILE = ROOT / "paper_trades.csv"
PATTERN_MEMORY_FILE = ROOT / "pattern_memory.json"
DB_FILE = ROOT / "smart_market_ai.db"

st.set_page_config(
    page_title="Smart Market AI | Terminal",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

ai.load_env_file()

# =============================================================================
# THEME
# =============================================================================

THEMES = {
    "Midnight": {
        "bg": "#0a0e17",
        "bg2": "#0d1220",
        "panel": "#111827",
        "panel2": "#161e2e",
        "border": "rgba(148, 163, 184, 0.12)",
        "text": "#e5e9f0",
        "muted": "#8b94a7",
        "accent": "#3b82f6",
        "accent2": "#22d3ee",
        "green": "#22c55e",
        "red": "#ef4444",
        "amber": "#f59e0b",
        "purple": "#a78bfa",
    },
    "Slate Green": {
        "bg": "#0b0f0e",
        "bg2": "#0e1413",
        "panel": "#121a18",
        "panel2": "#182521",
        "border": "rgba(148, 163, 184, 0.12)",
        "text": "#e6f0ec",
        "muted": "#8aa39a",
        "accent": "#10b981",
        "accent2": "#34d399",
        "green": "#22c55e",
        "red": "#f87171",
        "amber": "#fbbf24",
        "purple": "#c4b5fd",
    },
}

PLOTLY_FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"


def inject_css(t):
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{background: transparent;}}

        .stApp {{
            background:
                radial-gradient(circle at 15% 0%, {t['bg2']} 0%, {t['bg']} 45%),
                {t['bg']};
        }}

        .block-container {{
            padding-top: 1.1rem;
            padding-bottom: 3rem;
            max-width: 1600px;
        }}

        /* ---------- Scrollbar ---------- */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: {t['bg']}; }}
        ::-webkit-scrollbar-thumb {{ background: {t['border']}; border-radius: 6px; }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: {t['bg2']};
            border-right: 1px solid {t['border']};
        }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

        /* ---------- Headings ---------- */
        h1, h2, h3, h4 {{ color: {t['text']} !important; font-weight: 700 !important; letter-spacing: -0.01em; }}
        p, span, label, .stMarkdown {{ color: {t['text']}; }}

        .smai-section-title {{
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {t['muted']};
            margin: 1.6rem 0 0.7rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .smai-section-title::after {{
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, {t['border']}, transparent);
        }}

        /* ---------- Header banner ---------- */
        .smai-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.9rem;
            background: linear-gradient(135deg, {t['panel']} 0%, {t['panel2']} 100%);
            border: 1px solid {t['border']};
            border-radius: 16px;
            padding: 1.1rem 1.5rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }}
        .smai-brand {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }}
        .smai-brand-mark {{
            width: 42px; height: 42px;
            border-radius: 11px;
            background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
            display: flex; align-items: center; justify-content: center;
            font-size: 1.3rem;
            box-shadow: 0 4px 14px {t['accent']}55;
        }}
        .smai-brand-title {{ font-size: 1.25rem; font-weight: 800; color: {t['text']}; line-height: 1.1; }}
        .smai-brand-sub {{ font-size: 0.72rem; color: {t['muted']}; letter-spacing: 0.04em; }}

        .smai-chip-row {{ display: flex; gap: 0.55rem; flex-wrap: wrap; }}
        .smai-chip {{
            display: flex; align-items: center; gap: 0.4rem;
            background: {t['bg']};
            border: 1px solid {t['border']};
            border-radius: 999px;
            padding: 0.42rem 0.85rem;
            font-size: 0.76rem;
            color: {t['muted']};
            font-weight: 500;
        }}
        .smai-chip b {{ color: {t['text']}; font-weight: 700; }}
        .smai-dot {{
            width: 8px; height: 8px; border-radius: 50%;
            background: {t['green']};
            box-shadow: 0 0 0 3px {t['green']}33;
            animation: smai-pulse 2s infinite;
        }}
        .smai-dot.off {{ background: {t['muted']}; box-shadow: none; animation: none; }}
        @keyframes smai-pulse {{
            0% {{ box-shadow: 0 0 0 0 {t['green']}55; }}
            70% {{ box-shadow: 0 0 0 7px {t['green']}00; }}
            100% {{ box-shadow: 0 0 0 0 {t['green']}00; }}
        }}

        .smai-badge {{
            padding: 0.42rem 0.85rem; border-radius: 999px; font-size: 0.76rem; font-weight: 700;
            border: 1px solid transparent;
        }}
        .smai-badge.bull {{ background: {t['green']}22; color: {t['green']}; border-color: {t['green']}44; }}
        .smai-badge.bear {{ background: {t['red']}22; color: {t['red']}; border-color: {t['red']}44; }}
        .smai-badge.neutral {{ background: {t['amber']}22; color: {t['amber']}; border-color: {t['amber']}44; }}
        .smai-badge.vol {{ background: {t['purple']}22; color: {t['purple']}; border-color: {t['purple']}44; }}

        /* ---------- KPI cards ---------- */
        .smai-kpi {{
            background: linear-gradient(160deg, {t['panel']}, {t['panel2']});
            border: 1px solid {t['border']};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            height: 100%;
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }}
        .smai-kpi:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 26px rgba(0,0,0,0.30);
            border-color: {t['accent']}55;
        }}
        .smai-kpi-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.55rem; }}
        .smai-kpi-icon {{ font-size: 1.05rem; opacity: 0.9; }}
        .smai-kpi-label {{ font-size: 0.72rem; color: {t['muted']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
        .smai-kpi-value {{ font-size: 1.65rem; font-weight: 800; color: {t['text']}; font-family: 'JetBrains Mono', monospace; line-height: 1.1; }}
        .smai-kpi-delta {{ font-size: 0.76rem; font-weight: 700; margin-top: 0.35rem; }}
        .smai-kpi-delta.up {{ color: {t['green']}; }}
        .smai-kpi-delta.down {{ color: {t['red']}; }}
        .smai-kpi-delta.flat {{ color: {t['muted']}; }}

        /* ---------- Panels ---------- */
        .smai-panel {{
            background: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: 14px;
            padding: 1rem 1.1rem 0.6rem 1.1rem;
            margin-bottom: 0.9rem;
        }}
        .smai-panel-title {{
            font-size: 0.88rem; font-weight: 700; color: {t['text']};
            margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem;
        }}

        /* ---------- Signal cards ---------- */
        .smai-signal-card {{
            background: linear-gradient(160deg, {t['panel']}, {t['panel2']});
            border: 1px solid {t['border']};
            border-left: 3px solid {t['muted']};
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.6rem;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }}
        .smai-signal-card:hover {{ transform: translateX(3px); box-shadow: 0 6px 18px rgba(0,0,0,0.25); }}
        .smai-signal-card.up {{ border-left-color: {t['green']}; }}
        .smai-signal-card.down {{ border-left-color: {t['red']}; }}
        .smai-signal-card.hold {{ border-left-color: {t['amber']}; }}
        .smai-sig-top {{ display: flex; justify-content: space-between; align-items: center; }}
        .smai-sig-ticker {{ font-weight: 800; font-size: 1.0rem; color: {t['text']}; font-family: 'JetBrains Mono', monospace; }}
        .smai-sig-action {{ font-size: 0.68rem; font-weight: 800; padding: 0.16rem 0.55rem; border-radius: 999px; letter-spacing: 0.04em; }}
        .smai-sig-action.up {{ background: {t['green']}22; color: {t['green']}; }}
        .smai-sig-action.down {{ background: {t['red']}22; color: {t['red']}; }}
        .smai-sig-action.hold {{ background: {t['amber']}22; color: {t['amber']}; }}
        .smai-sig-meta {{ font-size: 0.72rem; color: {t['muted']}; margin-top: 0.25rem; display: flex; gap: 0.7rem; flex-wrap: wrap; }}
        .smai-sig-bar-track {{ background: {t['bg']}; border-radius: 999px; height: 6px; margin-top: 0.5rem; overflow: hidden; }}
        .smai-sig-bar-fill {{ height: 100%; border-radius: 999px; }}
        .smai-quality {{
            font-size: 0.65rem; font-weight: 800; padding: 0.08rem 0.4rem; border-radius: 5px;
            background: {t['bg']}; color: {t['muted']}; border: 1px solid {t['border']};
        }}

        /* ---------- Misc ---------- */
        .smai-note {{ font-size: 0.76rem; color: {t['muted']}; }}
        .smai-empty {{
            border: 1px dashed {t['border']}; border-radius: 12px; padding: 1.4rem;
            text-align: center; color: {t['muted']}; font-size: 0.85rem;
        }}

        div[data-testid="stMetric"] {{
            background: {t['panel']};
            border: 1px solid {t['border']};
            padding: 0.8rem 1rem;
            border-radius: 12px;
        }}
        div[data-testid="stMetric"] label {{ color: {t['muted']} !important; }}

        div.stButton > button {{
            background: linear-gradient(135deg, {t['accent']}, {t['accent2']});
            color: white; border: none; border-radius: 10px; font-weight: 700;
            padding: 0.5rem 1rem; transition: filter 0.15s ease;
        }}
        div.stButton > button:hover {{ filter: brightness(1.12); }}

        div[data-baseweb="tab-list"] {{ gap: 4px; }}
        button[data-baseweb="tab"] {{
            background: {t['panel']}; border-radius: 10px 10px 0 0 !important; color: {t['muted']};
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color: {t['text']}; background: {t['panel2']}; }}

        [data-testid="stExpander"] {{
            background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 12px;
        }}

        .stDataFrame {{ border-radius: 12px; overflow: hidden; border: 1px solid {t['border']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SMALL HELPERS (unchanged from original logic)
# =============================================================================


def parse_tickers(text):
    return [t.strip().upper() for t in text.replace("\n", ",").split(",") if t.strip()]


def download_df(df, filename, label="Download CSV", key=None):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    st.download_button(label, buffer.getvalue(), filename, "text/csv", key=key, use_container_width=True)


def zscore_columns(data):
    normalized = data.copy()
    for col in normalized.columns:
        series = pd.to_numeric(normalized[col], errors="coerce")
        std = series.std()
        if std and std > 0:
            normalized[col] = (series - series.mean()) / std
        else:
            normalized[col] = 0
    return normalized.clip(-2.5, 2.5)


def load_model_summary():
    payload = ai.load_trained_payload()
    return payload if isinstance(payload, dict) else {}


def records_to_frame(records):
    rows = []
    for record in records:
        rows.append(
            {
                "ticker": record["ticker"],
                "price": record["price"],
                "model_signal": record.get("model_signal", record["signal"]),
                "signal": record["signal"],
                "raw_prob_pct": record.get("raw_prob_up", record["prob_up"]) * 100,
                "prob_up_pct": record["prob_up"] * 100,
                "setup_candidate": record.get("setup_candidate", True),
                "active_regime": record.get("active_regime", ""),
                "model_route": record.get("model_route", ""),
                "quality": record["quality"],
                "size": record["position_size"],
                "rsi": record["rsi"],
                "whale_score": record["whale_score"],
                "risk_on_score": record["risk_on_score"],
                "trend_score": record.get("trend_score", 0.5),
                "volatility_score": record.get("volatility_score", 0.5),
                "market_breadth_score": record.get("market_breadth_score", 0.5),
                "credit_stress_score": record.get("credit_stress_score", 0.5),
                "intel_score": record.get("intelligence_score", 0),
                "sma20_dist": record.get("sma20_dist", 0.0),
                "event": record.get("event_note", ""),
                "provider": record.get("intelligence_provider", ""),
                "explanation": ai.explain_record(record),
                "mi_score": ai.intelligence_report_score(record),
                "bias": ai.intelligence_bias_label(record),
                "bearish_score": ai.bearish_pressure_score(record),
                "bearish_label": ai.bearish_pressure_label(record),
                "sector": ai.sector_for_ticker(record["ticker"]),
            }
        )
    return pd.DataFrame(rows)


def run_live_scan(tickers):
    with st.spinner("Scanning tickers..."):
        records, payload = ai.live_prediction_records(tickers)
    if not records:
        st.warning("No live records returned. Check that a trained model exists and data providers are working.")
        return pd.DataFrame(), payload
    return records_to_frame(records), payload


@st.cache_data(ttl=300, show_spinner=False)
def cached_fetch(ticker, period="1y", interval="1d"):
    df = ai.fetch(ticker, period=period, interval=interval)
    return df


def snapshot_kpis(scan, trades_summary):
    if scan.empty:
        conf = bullish = bearish = avg_prob = 0
        regime = "n/a"
    else:
        conf = float((scan["prob_up_pct"] - 50).abs().mean() * 2)
        bullish = int((scan["signal"] == "UP").sum())
        bearish = int((scan["signal"] == "DOWN").sum())
        avg_prob = float(scan["prob_up_pct"].mean())
        regime = scan["active_regime"].replace("", np.nan).dropna()
        regime = regime.mode().iloc[0] if not regime.empty else "n/a"
    return {
        "confidence": conf,
        "bullish": bullish,
        "bearish": bearish,
        "avg_prob": avg_prob,
        "regime": regime,
        "win_rate": trades_summary.get("win_rate"),
        "portfolio_return": trades_summary.get("total_return"),
    }


def daily_paper_returns(closed):
    d = closed.copy()
    if d.empty or "actual_return" not in d.columns:
        return pd.Series(dtype=float)
    date_col = "exit_date" if "exit_date" in d.columns else d.columns[0]
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return pd.Series(dtype=float)
    d["actual_return"] = pd.to_numeric(d["actual_return"], errors="coerce")
    return d.groupby(date_col)["actual_return"].sum().sort_index()


def paper_trades_summary():
    if not PAPER_TRADES_FILE.exists():
        return {}, pd.DataFrame()
    try:
        trades = pd.read_csv(PAPER_TRADES_FILE)
    except pd.errors.EmptyDataError:
        return {}, pd.DataFrame()
    if "actual_return" not in trades.columns:
        return {}, trades
    trades["actual_return"] = pd.to_numeric(trades["actual_return"], errors="coerce")
    closed = trades.dropna(subset=["actual_return"]).copy()
    if closed.empty:
        return {}, trades
    if "correct" in closed.columns:
        win_rate = closed["correct"].astype(str).str.lower().eq("true").mean() * 100
    else:
        win_rate = (closed["actual_return"] > 0).mean() * 100
    daily = daily_paper_returns(closed)
    equity = (1 + daily).cumprod()
    total_return = (equity.iloc[-1] - 1) * 100 if len(equity) else 0.0
    summary = {
        "win_rate": win_rate,
        "total_return": total_return,
        "avg_return": closed["actual_return"].mean() * 100,
        "closed": len(closed),
        "max_drawdown": ai.max_drawdown_from_returns(daily) * 100,
        "profit_factor": ai.profit_factor_from_returns(daily),
    }
    return summary, trades


# =============================================================================
# UI ATOMS (cards / badges)
# =============================================================================


def fmt_delta(curr, prev, suffix="", places=1):
    if prev is None or curr is None:
        return "", "flat"
    diff = curr - prev
    if abs(diff) < 10 ** (-places) / 2:
        return f"\u2013 flat vs last scan", "flat"
    arrow = "\u25b2" if diff > 0 else "\u25bc"
    direction = "up" if diff > 0 else "down"
    return f"{arrow} {abs(diff):.{places}f}{suffix} vs last scan", direction


def kpi_card(icon, label, value, delta_text=None, delta_dir="flat"):
    delta_html = f'<div class="smai-kpi-delta {delta_dir}">{delta_text}</div>' if delta_text else ""
    st.markdown(
        f"""
        <div class="smai-kpi">
            <div class="smai-kpi-top">
                <div class="smai-kpi-label">{label}</div>
                <div class="smai-kpi-icon">{icon}</div>
            </div>
            <div class="smai-kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_card(row, t):
    signal = row["signal"]
    cls = "up" if signal == "UP" else "down" if signal == "DOWN" else "hold"
    color = t["green"] if cls == "up" else t["red"] if cls == "down" else t["amber"]
    prob = row["prob_up_pct"]
    whale = row["whale_score"]
    st.markdown(
        f"""
        <div class="smai-signal-card {cls}">
            <div class="smai-sig-top">
                <span class="smai-sig-ticker">{row['ticker']}</span>
                <span class="smai-sig-action {cls}">{signal}</span>
            </div>
            <div class="smai-sig-meta">
                <span>{prob:.0f}% prob</span>
                <span class="smai-quality">Quality {row['quality']}</span>
                <span>Whale {whale:+.2f}</span>
                <span>RSI {row['rsi']:.0f}</span>
            </div>
            <div class="smai-sig-bar-track">
                <div class="smai-sig-bar-fill" style="width:{max(0,min(100,prob))}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_chip(ok, label_true, label_false):
    return f'<span class="smai-chip"><span class="smai-dot{" " if ok else " off"}"></span>{label_true if ok else label_false}</span>'


def regime_badge_class(regime):
    regime = str(regime).lower()
    if regime in ("bear",):
        return "bear"
    if regime in ("trend",):
        return "bull"
    if regime in ("high_volatility",):
        return "vol"
    return "neutral"


def rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(fig, t, height=380, legend=True):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOTLY_FONT, color=t["text"], size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0) if legend else dict(),
        hoverlabel=dict(bgcolor=t["panel2"], font_color=t["text"], bordercolor=t["border"]),
    )
    fig.update_xaxes(gridcolor=t["border"], zerolinecolor=t["border"])
    fig.update_yaxes(gridcolor=t["border"], zerolinecolor=t["border"])
    return fig


# =============================================================================
# CHART BUILDERS (Plotly)
# =============================================================================


def chart_candlestick(df, ticker, t):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=t["green"],
                decreasing_line_color=t["red"],
                name=ticker,
            )
        ]
    )
    if "Volume" in df.columns:
        vol_colors = [t["green"] if c >= o else t["red"] for o, c in zip(df["Open"], df["Close"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, opacity=0.25, yaxis="y2", name="Volume")
        )
    fig.update_layout(
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(overlaying="y", side="left", showgrid=False, range=[0, df["Volume"].max() * 4] if "Volume" in df.columns else None),
        xaxis_rangeslider_visible=False,
    )
    return base_layout(fig, t, height=430, legend=False)


def chart_factor_heatmap(scan, t):
    heat_cols = {
        "prob_up_pct": "Prob edge",
        "whale_score": "Whale",
        "risk_on_score": "Risk",
        "trend_score": "Trend",
        "volatility_score": "Vol",
        "market_breadth_score": "Breadth",
        "credit_stress_score": "Credit",
        "rsi": "RSI tilt",
        "intel_score": "Intel",
        "mi_score": "Score",
    }
    raw = scan.set_index("ticker")[list(heat_cols.keys())].copy()
    raw["prob_up_pct"] = raw["prob_up_pct"] - 50
    raw["rsi"] = raw["rsi"] - 50
    raw = raw.rename(columns=heat_cols)
    z = zscore_columns(raw)
    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=z.columns,
            y=z.index,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="z"),
            hovertemplate="%{y} · %{x}<br>z=%{z:.2f}<extra></extra>",
        )
    )
    return base_layout(fig, t, height=max(360, 26 * len(z) + 90), legend=False)


def chart_sector_treemap(scan, t):
    grp = scan.copy()
    grp["abs_score"] = grp["mi_score"].abs().clip(lower=1)
    fig = px.treemap(
        grp,
        path=[px.Constant("Universe"), "sector", "ticker"],
        values="abs_score",
        color="mi_score",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        hover_data={"signal": True, "prob_up_pct": ":.1f"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color=t["bg"])), textfont=dict(color="white"))
    return base_layout(fig, t, height=420, legend=False)


def chart_bias_sunburst(scan, t):
    color_map = {
        "Bullish": t["green"], "Caution Bullish": "#8bc34a", "Neutral": t["amber"],
        "Conflicted": "#ff9800", "Caution Bearish": "#ff7043", "Bearish": t["red"],
    }
    fig = px.sunburst(
        scan,
        path=["bias", "signal", "ticker"],
        color="bias",
        color_discrete_map=color_map,
    )
    return base_layout(fig, t, height=420, legend=False)


def chart_correlation_matrix(price_frames, t):
    returns = pd.DataFrame({tk: df["Close"].pct_change() for tk, df in price_frames.items()}).dropna(how="all")
    corr = returns.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="\u03c1"),
            hovertemplate="%{y} vs %{x}<br>\u03c1=%{z:.2f}<extra></extra>",
        )
    )
    return base_layout(fig, t, height=max(360, 26 * len(corr) + 90), legend=False)


def chart_probability_distribution(scan, t):
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scan["prob_up_pct"],
            nbinsx=16,
            marker_color=t["accent"],
            opacity=0.85,
            name="Prob up %",
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color=t["muted"])
    return base_layout(fig, t, height=320, legend=False)


def chart_whale_activity(scan, t):
    colors = np.where(scan["whale_score"] >= 0, t["green"], t["red"])
    ordered = scan.sort_values("whale_score")
    fig = go.Figure(
        go.Bar(
            x=ordered["whale_score"],
            y=ordered["ticker"],
            orientation="h",
            marker_color=np.where(ordered["whale_score"] >= 0, t["green"], t["red"]),
        )
    )
    fig.add_vline(x=0, line_color=t["muted"])
    return base_layout(fig, t, height=max(320, 22 * len(ordered) + 60), legend=False)


def chart_gauge(value, title, t, min_v=0, max_v=100, good_high=True):
    color = t["green"] if (value >= 60) == good_high else t["amber"] if 40 <= value < 60 else t["red"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 13, "color": t["muted"]}},
            number={"suffix": "%", "font": {"color": t["text"], "size": 30}},
            gauge={
                "axis": {"range": [min_v, max_v], "tickcolor": t["muted"]},
                "bar": {"color": color},
                "bgcolor": t["bg"],
                "borderwidth": 0,
                "steps": [
                    {"range": [min_v, max_v * 0.4], "color": rgba(t["red"], 0.15)},
                    {"range": [max_v * 0.4, max_v * 0.6], "color": rgba(t["amber"], 0.15)},
                    {"range": [max_v * 0.6, max_v], "color": rgba(t["green"], 0.15)},
                ],
            },
        )
    )
    return base_layout(fig, t, height=230, legend=False)


def chart_equity_curve(equity, t):
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(equity))),
            y=equity.values,
            mode="lines",
            line=dict(color=t["accent2"], width=2),
            fill="tozeroy",
            fillcolor=rgba(t["accent"], 0.13),
            name="Equity",
        )
    )
    return base_layout(fig, t, height=300, legend=False)


def chart_drawdown(returns, t):
    values = pd.Series(returns).fillna(0)
    equity = (1 + values).cumprod()
    drawdown = (equity / equity.cummax() - 1) * 100
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(drawdown))),
            y=drawdown.values,
            mode="lines",
            line=dict(color=t["red"], width=1.5),
            fill="tozeroy",
            fillcolor=rgba(t["red"], 0.13),
        )
    )
    return base_layout(fig, t, height=300, legend=False)


def chart_monthly_returns(closed, date_col, t):
    d = closed.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d["year"] = d[date_col].dt.year
    d["month"] = d[date_col].dt.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = d.pivot_table(index="year", columns="month", values="actual_return", aggfunc="mean") * 100
    pivot = pivot.reindex(columns=month_order)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            texttemplate="%{z:.1f}",
            colorbar=dict(title="%"),
        )
    )
    return base_layout(fig, t, height=max(220, 60 * len(pivot) + 90), legend=False)


def chart_return_box(closed, t):
    fig = px.box(closed, x="signal", y="actual_return", color="signal",
                 color_discrete_map={"UP": t["green"], "DOWN": t["red"], "HOLD": t["amber"]}, points="all")
    fig.update_yaxes(tickformat=".1%")
    return base_layout(fig, t, height=340, legend=False)


def chart_factor_radar(row, t):
    categories = ["Trend", "Volatility", "Breadth", "Credit", "Risk-On", "Whale (norm)"]
    values = [
        row["trend_score"], row["volatility_score"], row["market_breadth_score"],
        row["credit_stress_score"], row["risk_on_score"], (row["whale_score"] + 1) / 2,
    ]
    values = [float(np.clip(v, 0, 1)) for v in values]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill="toself",
            line=dict(color=t["accent2"]),
            fillcolor=rgba(t["accent"], 0.2),
            name=row["ticker"],
        )
    )
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor=t["border"]),
                                  angularaxis=dict(gridcolor=t["border"])))
    return base_layout(fig, t, height=320, legend=False)


def waterfall_components(row):
    """Mirrors ai.intelligence_report_score exactly, for transparency only."""
    direction_bias = (row["prob_up_pct"] / 100 - 0.5) * 100
    whale = row["whale_score"] * 18
    regime = (row["risk_on_score"] - 0.5) * 16
    intel = row["intel_score"] * 8
    rsi_v = row["rsi"]
    timing = 0.0
    if 40 <= rsi_v <= 65:
        timing += 4
    elif rsi_v > 75:
        timing -= 5
    elif rsi_v < 30:
        timing -= 2
    if abs(row["sma20_dist"]) > 0.10:
        timing -= 4

    signal = row["signal"]
    whale_raw = row["whale_score"]
    intel_v = row["intel_score"]
    if signal == "UP":
        conflicted = (whale_raw <= -0.20 and rsi_v < 40) or intel_v <= -1
    elif signal == "DOWN":
        conflicted = (whale_raw >= 0.20 and rsi_v > 60) or intel_v >= 1
    else:
        conflicted = (whale_raw <= -0.20 and rsi_v < 40) or (whale_raw >= 0.20 and rsi_v > 70)
    conflict_penalty = -10 if conflicted else 0

    return {
        "Direction bias": direction_bias,
        "Whale": whale,
        "Regime": regime,
        "Intel": intel,
        "Timing": timing,
        "Conflict penalty": conflict_penalty,
    }


def chart_score_waterfall(row, t):
    parts = waterfall_components(row)
    labels = list(parts.keys())
    values = list(parts.values())
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(values) + ["total"],
            x=labels + ["Total score"],
            y=values + [sum(values)],
            connector={"line": {"color": t["border"]}},
            increasing={"marker": {"color": t["green"]}},
            decreasing={"marker": {"color": t["red"]}},
            totals={"marker": {"color": t["accent"]}},
        )
    )
    return base_layout(fig, t, height=340, legend=False)


def chart_tag_performance(tag_perf, t):
    ordered = tag_perf.sort_values("avg_return")
    fig = go.Figure(
        go.Bar(
            x=ordered["avg_return"],
            y=ordered.index,
            orientation="h",
            marker_color=np.where(ordered["avg_return"] >= 0, t["green"], t["red"]),
        )
    )
    fig.add_vline(x=0, line_color=t["muted"])
    return base_layout(fig, t, height=max(300, 26 * len(ordered) + 60), legend=False)


# =============================================================================
# TABLE STYLING
# =============================================================================


def style_scan_table(df, t):
    def color_signal(val):
        if val == "UP":
            return f"color: {t['green']}; font-weight: 700;"
        if val == "DOWN":
            return f"color: {t['red']}; font-weight: 700;"
        return f"color: {t['amber']}; font-weight: 700;"

    styler = df.style
    if "signal" in df.columns:
        styler = styler.applymap(color_signal, subset=["signal"])
    if "quality" in df.columns:
        styler = styler.applymap(
            lambda v: f"font-weight:700; color:{t['green'] if v == 'A' else t['amber'] if v == 'B' else t['muted']}",
            subset=["quality"],
        )
    if "prob_up_pct" in df.columns:
        styler = styler.background_gradient(subset=["prob_up_pct"], cmap="RdYlGn", vmin=0, vmax=100)
    return styler


# =============================================================================
# SIDEBAR
# =============================================================================


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
                <span style="font-size:1.4rem;">\U0001F4E1</span>
                <span style="font-weight:800;font-size:1.05rem;">Control Center</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        theme_name = st.selectbox("Theme", list(THEMES.keys()), index=0)

        st.markdown('<div class="smai-section-title">Scanner</div>', unsafe_allow_html=True)
        tickers_text = st.text_area(
            "Universe (comma or newline separated)",
            ",".join(ai.DEFAULT_TICKERS),
            height=110,
            key="sidebar_tickers",
        )
        top_only = st.checkbox("Top setups only", value=False)
        include_corr = st.checkbox("Include correlation matrix (fetches price history)", value=True)
        run_scan = st.button("\u26a1 Run Scan", type="primary", use_container_width=True)

        st.markdown('<div class="smai-section-title">Quick Actions</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Clear cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared")
        if c2.button("Reset scan", use_container_width=True):
            for k in ("scan", "payload", "prev_kpis", "corr_frames"):
                st.session_state.pop(k, None)
            st.success("Scan reset")

        st.markdown('<div class="smai-section-title">System Status</div>', unsafe_allow_html=True)
        model_summary = load_model_summary()
        st.markdown(
            status_chip(MODEL_FILE.exists(), "Model loaded", "Model missing")
            + status_chip(bool(ai.finnhub_api_key()), "Finnhub API connected", "Finnhub API missing")
            + status_chip(DB_FILE.exists(), "Database online", "Database missing")
            + status_chip(ai.ENABLE_FINBERT, "FinBERT enabled", "FinBERT disabled"),
            unsafe_allow_html=True,
        )
        if model_summary:
            st.caption(
                f"Model: **{model_summary.get('general_model_name', 'n/a')}** "
                f"· Status: **{model_summary.get('model_status', 'n/a')}** "
                f"· Trained: {str(model_summary.get('trained_at', 'n/a'))[:19]}"
            )

        with st.expander("\U0001F9EA Diagnostics", expanded=False):
            st.write("Model file:", "Found" if MODEL_FILE.exists() else "Missing")
            st.write("Paper trades:", "Found" if PAPER_TRADES_FILE.exists() else "Missing")
            st.write("Pattern memory:", "Found" if PATTERN_MEMORY_FILE.exists() else "Missing")
            st.write("SQLite DB:", "Found" if DB_FILE.exists() else "Missing")
            st.write("Feature count:", len(ai.FEATURE_COLUMNS))
            payload = st.session_state.get("payload")
            if payload:
                st.json({k: str(v)[:300] for k, v in payload.items() if k != "model"})
            with st.expander("Feature columns"):
                st.write(ai.FEATURE_COLUMNS)

        st.caption("Educational/informational only. Not financial advice.")

    return theme_name, tickers_text, top_only, run_scan, include_corr


# =============================================================================
# HEADER + KPI ROW
# =============================================================================


def render_header(scan, model_summary, t):
    now = dt.datetime.now()
    regime = "n/a"
    if not scan.empty:
        modes = scan["active_regime"].replace("", np.nan).dropna()
        regime = modes.mode().iloc[0] if not modes.empty else "n/a"
    regime_display = regime.replace("_", " ").title() if regime != "n/a" else "Unknown"
    model_name = model_summary.get("general_model_name", "n/a") if model_summary else "No model"
    model_status = model_summary.get("model_status", "") if model_summary else ""

    st.markdown(
        f"""
        <div class="smai-header">
            <div class="smai-brand">
                <div class="smai-brand-mark">\U0001F9E0</div>
                <div>
                    <div class="smai-brand-title">Smart Market AI</div>
                    <div class="smai-brand-sub">INSTITUTIONAL SIGNAL TERMINAL</div>
                </div>
            </div>
            <div class="smai-chip-row">
                <span class="smai-chip"><span class="smai-dot"></span><b>Live</b></span>
                <span class="smai-badge {regime_badge_class(regime)}">\U0001F4CA {regime_display}</span>
                <span class="smai-chip">\U0001F9E9 Model: <b>{model_name}</b>{(' · ' + model_status) if model_status else ''}</span>
                <span class="smai-chip">\U0001F4C5 {now.strftime('%a %d %b %Y &middot; %H:%M')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(scan, trades_summary, t):
    kpis = snapshot_kpis(scan, trades_summary)
    prev = st.session_state.get("prev_kpis", {})

    cols = st.columns(7)
    with cols[0]:
        d, dirn = fmt_delta(kpis["confidence"], prev.get("confidence"), "pt")
        kpi_card("\U0001F3AF", "AI Confidence", f"{kpis['confidence']:.0f}%", d, dirn)
    with cols[1]:
        d, dirn = fmt_delta(kpis["bullish"], prev.get("bullish"), "", 0)
        kpi_card("\U0001F4C8", "Bullish Signals", f"{kpis['bullish']}", d, dirn)
    with cols[2]:
        d, dirn = fmt_delta(kpis["bearish"], prev.get("bearish"), "", 0)
        kpi_card("\U0001F4C9", "Bearish Signals", f"{kpis['bearish']}", d, dirn)
    with cols[3]:
        wr = kpis["win_rate"]
        kpi_card("\U0001F3C6", "Win Rate", f"{wr:.1f}%" if wr is not None else "n/a")
    with cols[4]:
        pr = kpis["portfolio_return"]
        kpi_card("\U0001F4B0", "Portfolio Return", f"{pr:+.1f}%" if pr is not None else "n/a")
    with cols[5]:
        kpi_card("\U0001F30A", "Market Regime", str(kpis["regime"]).replace("_", " ").title())
    with cols[6]:
        d, dirn = fmt_delta(kpis["avg_prob"], prev.get("avg_prob"), "pt")
        kpi_card("\u26a1", "Avg Probability", f"{kpis['avg_prob']:.1f}%", d, dirn)

    st.session_state["prev_kpis"] = kpis


# =============================================================================
# LEFT / RIGHT COLUMNS
# =============================================================================


def render_left_column(scan, t):
    st.markdown('<div class="smai-panel-title">\U0001F4E1 Live Scanner</div>', unsafe_allow_html=True)
    if scan.empty:
        st.markdown('<div class="smai-empty">Run a scan from the sidebar to populate the terminal.</div>', unsafe_allow_html=True)
        return

    search = st.text_input("Search ticker", "", key="scanner_search", placeholder="e.g. NVDA")
    filter_signal = st.multiselect("Filter by action", ["UP", "DOWN", "HOLD"], default=["UP", "DOWN", "HOLD"])

    filtered = scan[scan["signal"].isin(filter_signal)]
    if search:
        filtered = filtered[filtered["ticker"].str.contains(search.upper())]
    filtered = filtered.sort_values(["signal", "prob_up_pct"], ascending=[True, False])

    st.caption(f"{len(filtered)} of {len(scan)} tickers shown")
    for _, row in filtered.head(25).iterrows():
        signal_card(row, t)

    st.markdown('<div class="smai-section-title">Watchlist (Bearish Pressure)</div>', unsafe_allow_html=True)
    watch = scan.sort_values("bearish_score", ascending=False)
    watch = watch[watch["bearish_score"] >= 8].head(8)
    if watch.empty:
        st.markdown('<div class="smai-empty">No meaningful bearish pressure right now.</div>', unsafe_allow_html=True)
    else:
        st.dataframe(
            watch[["ticker", "bearish_label", "bearish_score", "prob_up_pct", "whale_score"]],
            use_container_width=True, hide_index=True,
            column_config={
                "bearish_score": st.column_config.ProgressColumn("Pressure", min_value=0, max_value=max(40, watch["bearish_score"].max()), format="%.0f"),
                "prob_up_pct": st.column_config.NumberColumn("Prob Up", format="%.1f%%"),
                "whale_score": st.column_config.NumberColumn("Whale", format="%.2f"),
            },
        )

    download_df(scan, "live_scan.csv", key="dl_scan_left")


def render_right_column(scan, t):
    st.markdown('<div class="smai-panel-title">\U0001F4C8 Interactive Chart & Latest Prediction</div>', unsafe_allow_html=True)
    if scan.empty:
        st.markdown('<div class="smai-empty">Run a scan to see the candlestick chart and top prediction.</div>', unsafe_allow_html=True)
        return

    ranked = scan.sort_values("mi_score", ascending=False)
    default_idx = 0
    ticker_options = ranked["ticker"].tolist()
    selected = st.selectbox("Select ticker", ticker_options, index=default_idx, key="chart_ticker")
    row = scan[scan["ticker"] == selected].iloc[0]

    try:
        with st.spinner(f"Loading price history for {selected}..."):
            price_df = cached_fetch(selected, period="1y")
        if price_df is not None and not price_df.empty:
            st.plotly_chart(chart_candlestick(price_df, selected, t), use_container_width=True, config={"displaylogo": False}, key=f"chart_candle_{selected}")
        else:
            st.markdown('<div class="smai-empty">No price history available.</div>', unsafe_allow_html=True)
    except Exception as exc:
        st.warning(f"Could not load chart for {selected}: {exc}")

    cls = "up" if row["signal"] == "UP" else "down" if row["signal"] == "DOWN" else "hold"
    color = t["green"] if cls == "up" else t["red"] if cls == "down" else t["amber"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Signal", row["signal"])
    m2.metric("Prob Up", f"{row['prob_up_pct']:.1f}%")
    m3.metric("Price", f"${row['price']:.2f}")
    m4.metric("Quality", row["quality"])
    st.markdown(f"<div class='smai-note'>{row['explanation']}</div>", unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.caption("Factor radar")
        st.plotly_chart(chart_factor_radar(row, t), use_container_width=True, config={"displaylogo": False}, key=f"chart_radar_{row['ticker']}")
    with rc2:
        st.caption("Market intelligence score breakdown")
        st.plotly_chart(chart_score_waterfall(row, t), use_container_width=True, config={"displaylogo": False}, key=f"chart_waterfall_{row['ticker']}")

    st.markdown('<div class="smai-section-title">Current Market Snapshot</div>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Avg Risk-On", f"{scan['risk_on_score'].mean():.2f}")
    b2.metric("Avg Breadth", f"{scan['market_breadth_score'].mean():.2f}")
    b3.metric("Avg Credit Stress", f"{scan['credit_stress_score'].mean():.2f}")
    b4.metric("Avg Volatility", f"{scan['volatility_score'].mean():.2f}")


# =============================================================================
# ROW SECTIONS
# =============================================================================


def render_second_row(scan, include_corr, t):
    st.markdown('<div class="smai-section-title">\U0001F4CA Cross-Sectional Analysis</div>', unsafe_allow_html=True)
    if scan.empty:
        st.markdown('<div class="smai-empty">Run a scan to populate heatmaps, sector map, and correlations.</div>', unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Factor Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_factor_heatmap(scan, t), use_container_width=True, config={"displaylogo": False}, key="chart_factor_heatmap")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Sector Treemap</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_sector_treemap(scan, t), use_container_width=True, config={"displaylogo": False}, key="chart_sector_treemap")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Bias Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_bias_sunburst(scan, t), use_container_width=True, config={"displaylogo": False}, key="chart_bias_sunburst")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="smai-panel"><div class="smai-panel-title">Correlation Matrix</div>', unsafe_allow_html=True)
    if not include_corr:
        st.caption("Correlation matrix disabled in sidebar (enable to fetch price history).")
    else:
        top_tickers = scan.sort_values("mi_score", key=lambda s: s.abs(), ascending=False)["ticker"].head(15).tolist()
        price_frames = {}
        with st.spinner("Fetching price history for correlation matrix..."):
            for tk in top_tickers:
                try:
                    df = cached_fetch(tk, period="6mo")
                    if df is not None and not df.empty:
                        price_frames[tk] = df
                except Exception:
                    continue
        if len(price_frames) >= 2:
            st.plotly_chart(chart_correlation_matrix(price_frames, t), use_container_width=True, config={"displaylogo": False}, key="chart_correlation_matrix")
        else:
            st.caption("Not enough price history retrieved to build a correlation matrix.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_third_row(scan, t):
    st.markdown('<div class="smai-section-title">\U0001F52E Probability, Whale Flow & Breadth</div>', unsafe_allow_html=True)
    if scan.empty:
        st.markdown('<div class="smai-empty">Run a scan to see probability distribution and whale activity.</div>', unsafe_allow_html=True)
        return

    c1, c2, c3 = st.columns([1, 1.3, 1])
    with c1:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Probability Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_probability_distribution(scan, t), use_container_width=True, config={"displaylogo": False}, key="chart_prob_dist_third_row")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Whale Activity by Ticker</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_whale_activity(scan, t), use_container_width=True, config={"displaylogo": False}, key="chart_whale_activity")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Market Breadth</div>', unsafe_allow_html=True)
        breadth_pct = float(scan["market_breadth_score"].mean() * 100)
        st.plotly_chart(chart_gauge(breadth_pct, "Breadth score", t), use_container_width=True, config={"displaylogo": False}, key="chart_gauge_breadth")
        risk_pct = float(scan["risk_on_score"].mean() * 100)
        st.plotly_chart(chart_gauge(risk_pct, "Risk-on score", t), use_container_width=True, config={"displaylogo": False}, key="chart_gauge_risk_on")
        st.markdown("</div>", unsafe_allow_html=True)


def render_fourth_row(trades_summary, trades_df, t):
    st.markdown('<div class="smai-section-title">\U0001F4B0 Paper Trade Performance</div>', unsafe_allow_html=True)
    if not PAPER_TRADES_FILE.exists() or trades_df.empty:
        st.markdown('<div class="smai-empty">No paper_trades.csv found yet.</div>', unsafe_allow_html=True)
        return

    if not trades_summary:
        st.markdown('<div class="smai-empty">No closed trades yet - equity, drawdown, and monthly return charts will appear once trades close.</div>', unsafe_allow_html=True)
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        download_df(trades_df, "paper_trades.csv", key="dl_trades_4th")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Closed", trades_summary["closed"])
    m2.metric("Win Rate", f"{trades_summary['win_rate']:.1f}%")
    m3.metric("Avg Return", f"{trades_summary['avg_return']:+.2f}%")
    m4.metric("Max Drawdown", f"{trades_summary['max_drawdown']:.1f}%")
    pf = trades_summary["profit_factor"]
    m5.metric("Profit Factor", f"{pf:.2f}" if np.isfinite(pf) else "\u221e")

    closed = trades_df.copy()
    closed["actual_return"] = pd.to_numeric(closed["actual_return"], errors="coerce")
    closed = closed.dropna(subset=["actual_return"])
    daily = daily_paper_returns(closed)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Equity Curve</div>', unsafe_allow_html=True)
        equity = (1 + daily).cumprod()
        st.plotly_chart(chart_equity_curve(equity, t), use_container_width=True, config={"displaylogo": False}, key="chart_equity_curve")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Drawdown</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_drawdown(daily, t), use_container_width=True, config={"displaylogo": False}, key="chart_drawdown")
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    date_col = "exit_date" if "exit_date" in closed.columns else ("target_date" if "target_date" in closed.columns else None)
    with c3:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Monthly Returns</div>', unsafe_allow_html=True)
        if date_col:
            st.plotly_chart(chart_monthly_returns(closed, date_col, t), use_container_width=True, config={"displaylogo": False}, key="chart_monthly_returns")
        else:
            st.caption("No date column available to build a monthly returns grid.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Return Distribution by Signal</div>', unsafe_allow_html=True)
        if "signal" in closed.columns:
            st.plotly_chart(chart_return_box(closed, t), use_container_width=True, config={"displaylogo": False}, key="chart_return_box")
        st.markdown("</div>", unsafe_allow_html=True)

    download_df(trades_df, "paper_trades.csv", key="dl_trades_4th_full")


def render_fifth_row(scan, model_summary, t):
    st.markdown('<div class="smai-section-title">\U0001F9E0 Model Confidence</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.4])
    with c1:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">AI Confidence Gauge</div>', unsafe_allow_html=True)
        if not scan.empty:
            conf = float((scan["prob_up_pct"] - 50).abs().mean() * 2)
            st.plotly_chart(chart_gauge(conf, "Avg conviction", t), use_container_width=True, config={"displaylogo": False}, key="chart_gauge_conviction")
        else:
            st.markdown('<div class="smai-empty">Run a scan to compute live confidence.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Model Status</div>', unsafe_allow_html=True)
        if model_summary:
            st.write("Model:", model_summary.get("general_model_name", "n/a"))
            st.write("Status:", model_summary.get("model_status", "n/a"))
            st.write("Training mode:", model_summary.get("training_mode", "n/a"))
            st.write("Prob thresholds:", f"UP \u2265 {model_summary.get('prob_up', 'n/a')}, DOWN \u2264 {model_summary.get('prob_down', 'n/a')}")
            st.write("Trained at:", str(model_summary.get("trained_at", "n/a"))[:19])
            _metrics = model_summary.get("metrics") or {}
            if _metrics:
                st.write(
                    "Risk-adjusted: "
                    f"Sharpe {float(_metrics.get('annualized_sharpe', 0)):.2f} · "
                    f"PSR {float(_metrics.get('psr', 0)):.2f} · "
                    f"DSR({int(_metrics.get('dsr_trials', 91))}) {float(_metrics.get('dsr', 0)):.2f}"
                )
        else:
            st.markdown('<div class="smai-empty">No trained model found.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="smai-panel"><div class="smai-panel-title">Probability Spread (live scan)</div>', unsafe_allow_html=True)
        if not scan.empty:
            st.plotly_chart(chart_probability_distribution(scan, t), use_container_width=True, config={"displaylogo": False}, key="chart_prob_dist_fifth_row")
            st.caption(
                "Feature importance / SHAP is intentionally omitted here: the backend has no function that "
                "returns live per-ticker attributions, so this panel shows model confidence derived from live "
                "probabilities instead of fabricating an importance chart."
            )
        else:
            st.markdown('<div class="smai-empty">Run a scan to see the live probability spread.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_bottom(scan, trades_df, t):
    st.markdown('<div class="smai-section-title">\U0001F4DC Trade History, Pattern Memory & Diagnostics</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Trade History", "Pattern Memory", "Diagnostics"])

    with tabs[0]:
        if trades_df.empty:
            st.markdown('<div class="smai-empty">No paper_trades.csv found.</div>', unsafe_allow_html=True)
        else:
            search = st.text_input("Search trade history by ticker", "", key="trade_history_search")
            view = trades_df.copy()
            if search and "ticker" in view.columns:
                view = view[view["ticker"].astype(str).str.upper().str.contains(search.upper())]
            try:
                styled = style_scan_table(view, t) if "signal" in view.columns else view
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(view, use_container_width=True, hide_index=True)
            download_df(trades_df, "trade_history.csv", key="dl_trade_history")

    with tabs[1]:
        if not PATTERN_MEMORY_FILE.exists():
            st.markdown('<div class="smai-empty">No pattern_memory.json found yet. Close some paper trades first.</div>', unsafe_allow_html=True)
        else:
            memory = ai.load_json_file(PATTERN_MEMORY_FILE, {"patterns": []})
            patterns = memory.get("patterns", [])
            if not patterns:
                st.markdown('<div class="smai-empty">Pattern memory is empty.</div>', unsafe_allow_html=True)
            else:
                df = pd.DataFrame(patterns)
                df["result"] = pd.to_numeric(df["result"], errors="coerce")
                c1, c2, c3 = st.columns(3)
                c1.metric("Patterns", len(df))
                c2.metric("Accuracy", f"{df['correct'].astype(bool).mean() * 100:.1f}%")
                c3.metric("Avg Result", f"{df['result'].mean() * 100:+.2f}%")
                st.dataframe(df.sort_values("closed_at", ascending=False), use_container_width=True, hide_index=True)

                tag_rows = []
                for _, row in df.iterrows():
                    tags = row.get("tags", [])
                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags)
                        except Exception:
                            tags = []
                    for tag in tags:
                        tag_rows.append({"tag": tag, "result": row["result"], "correct": bool(row["correct"])})
                if tag_rows:
                    tag_df = pd.DataFrame(tag_rows)
                    tag_perf = (
                        tag_df.groupby("tag")
                        .agg(count=("tag", "size"), avg_return=("result", "mean"), accuracy=("correct", "mean"))
                        .query("count >= 2")
                        .sort_values("avg_return", ascending=False)
                    )
                    if not tag_perf.empty:
                        tag_perf["avg_return"] *= 100
                        tag_perf["accuracy"] *= 100
                        cc1, cc2 = st.columns([1.3, 1])
                        with cc1:
                            st.plotly_chart(chart_tag_performance(tag_perf.head(12), t), use_container_width=True, config={"displaylogo": False}, key="chart_tag_performance")
                        with cc2:
                            st.dataframe(tag_perf, use_container_width=True)
                download_df(df, "pattern_memory.csv", key="dl_pattern_memory")

    with tabs[2]:
        payload = st.session_state.get("payload") or load_model_summary()
        st.write("Model file:", "Found" if MODEL_FILE.exists() else "Missing")
        st.write("Paper trades:", "Found" if PAPER_TRADES_FILE.exists() else "Missing")
        st.write("Pattern memory:", "Found" if PATTERN_MEMORY_FILE.exists() else "Missing")
        st.write("SQLite DB:", "Found" if DB_FILE.exists() else "Missing")
        st.write("Feature count:", len(ai.FEATURE_COLUMNS))
        st.write("Finnhub key:", "Loaded" if ai.finnhub_api_key() else "Missing")
        st.write("FinBERT:", "Enabled" if ai.ENABLE_FINBERT else "Disabled")
        if payload:
            st.json({k: str(v)[:300] for k, v in payload.items() if k != "model"})
        with st.expander("Feature columns"):
            st.write(ai.FEATURE_COLUMNS)


# =============================================================================
# MAIN
# =============================================================================


def main():
    theme_name, tickers_text, top_only, run_scan, include_corr = render_sidebar()
    t = THEMES[theme_name]
    inject_css(t)

    if run_scan:
        tickers = parse_tickers(tickers_text)
        scan, payload = run_live_scan(tickers)
        if top_only and not scan.empty:
            scan = scan[
                (scan["signal"].isin(["UP", "DOWN"]))
                & (scan["quality"].isin(["A", "B"]))
                & (scan["size"] >= 0.5)
            ]
        st.session_state["scan"] = scan
        st.session_state["payload"] = payload

    scan = st.session_state.get("scan", pd.DataFrame())
    model_summary = st.session_state.get("payload") or load_model_summary()
    trades_summary, trades_df = paper_trades_summary()

    render_header(scan, model_summary, t)
    render_kpi_row(scan, trades_summary, t)

    left, right = st.columns([1, 1.5])
    with left:
        render_left_column(scan, t)
    with right:
        render_right_column(scan, t)

    render_second_row(scan, include_corr, t)
    render_third_row(scan, t)
    render_fourth_row(trades_summary, trades_df, t)
    render_fifth_row(scan, model_summary, t)
    render_bottom(scan, trades_df, t)


if __name__ == "__main__":
    main()
else:
    main()