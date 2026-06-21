"""Agent Inc. — dashboard (P2).

Run it:

    uv run --with streamlit --with pandas streamlit run dashboard.py

Layout: a header, a stats strip, then four bordered section cards — Benchmark
(leaderboard), Live engagement, RL improvement (runway + training curve), and
Per-scenario breakdown. All data access goes through dashboard_data.py — this
file is the view (styling included). Theme lives in .streamlit/config.toml.
"""

import altair as alt
import pandas as pd
import streamlit as st

import dashboard_data as data
import live_run

ACCENT = "#4D9FFF"  # vivid chart series — color pops on the black SpaceX canvas

# ── styled Altair charts (clean, transparent, SpaceX-themed axes) ──────────────
_AX = {
    "labelColor": "#8a8a8a", "labelFont": "Saira", "labelFontSize": 11,
    "titleColor": "#8a8a8a", "domainColor": "#2a2a2a", "tickColor": "#2a2a2a",
}


def _pro_bar(df, x, y, *, height=300, ylim=(0.0, 1.0), sort="-y", label_fmt=".2f"):
    """Rounded bars + value labels + subtle dashed grid on a transparent canvas."""
    if ylim is None:
        ymax = float(df[y].max()) if len(df) else 1.0
        ylim = (0.0, ymax * 1.2 if ymax > 0 else 1.0)
    x_enc = alt.X(f"{x}:N", sort=sort, title=None,
                  axis=alt.Axis(labelAngle=-22, labelLimit=150, grid=False, ticks=False, **_AX))
    y_enc = alt.Y(f"{y}:Q", title=None, scale=alt.Scale(domain=list(ylim)),
                  axis=alt.Axis(grid=True, gridColor="#1a1a1a", gridDash=[2, 3], ticks=False,
                                domain=False, format=".1f", **_AX))
    base = alt.Chart(df)
    bars = base.mark_bar(color=ACCENT, opacity=0.9, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(x=x_enc, y=y_enc)
    labels = base.mark_text(dy=-8, color="#d8d8d8", font="Saira", fontSize=11).encode(
        x=x_enc, y=y_enc, text=alt.Text(f"{y}:Q", format=label_fmt))
    return ((bars + labels).properties(width="container", height=height)
            .configure_view(strokeWidth=0).configure(background="rgba(0,0,0,0)"))


def _pro_line(df, x, y, *, height=300):
    """Clean line + points for the training curve."""
    line = alt.Chart(df).mark_line(
        color=ACCENT, strokeWidth=2.5, point=alt.OverlayMarkDef(color=ACCENT, fill=ACCENT, size=55)
    ).encode(
        x=alt.X(f"{x}:Q", title=None, axis=alt.Axis(grid=False, ticks=False, **_AX)),
        y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(grid=True, gridColor="#1a1a1a", gridDash=[2, 3],
                                                     ticks=False, domain=False, format=".2f", **_AX)),
    )
    return (line.properties(width="container", height=height)
            .configure_view(strokeWidth=0).configure(background="rgba(0,0,0,0)"))


def _section(num, eyebrow, title):
    """A numbered section header (eyebrow + title) for the top of a card."""
    st.markdown(
        f'<div class="sec-head"><div class="eyebrow"><span class="enum">{num}</span>{eyebrow}</div>'
        f'<div class="sec-title">{title}</div></div>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Agent Inc.", page_icon="📊", layout="wide")

# ── presentation: typography, hide chrome, section cards, KPI + chart styling ────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Saira:wght@300;400;500;600;700&display=swap');
      html, body, [class*="css"], [data-testid="stMarkdownContainer"] { font-family: 'Saira', 'Helvetica Neue', Arial, sans-serif; }
      .stApp { background: #000; }
      #MainMenu, [data-testid="stToolbar"], [data-testid="stStatusWidget"], footer, [data-testid="stDecoration"] { display: none !important; }
      header[data-testid="stHeader"] { background: transparent; height: 0; }
      .block-container { padding-top: 2.4rem; padding-bottom: 3.5rem; max-width: 1320px; }

      /* section cards */
      [data-testid="stVerticalBlockBorderWrapper"] {
        background: #07080A; border: 1px solid #191c22 !important; border-radius: 8px;
        padding: 1.15rem 1.4rem 1.35rem; margin-bottom: 0.55rem;
      }
      /* numbered section heads */
      .sec-head { margin-bottom: 0.85rem; }
      .eyebrow { color: #4D9FFF; font-size: 0.62rem; letter-spacing: 0.24em; text-transform: uppercase; font-weight: 600; }
      .enum { color: #41505f; margin-right: 0.55rem; }
      .sec-title { color: #fff; font-size: 1.02rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.14em; margin-top: 0.2rem; }

      h3 { text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500 !important; font-size: 0.76rem !important; color: #8a8a8a !important; }
      [data-testid="stMetric"] { background: transparent; border: none; border-bottom: 1px solid #1a1a1a; border-radius: 0; padding: 0.3rem 0 0.5rem; }
      [data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.66rem !important; color: #6f6f6f; }
      [data-testid="stMetricValue"] { font-weight: 600; color: #fff; }
      [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #1a1a1a; border-radius: 0; }
      [data-testid="stWidgetLabel"] p { text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.7rem !important; color: #8a8a8a; }
      .stButton button { text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; border-radius: 0; }
      .stButton button[kind="primary"] { background: #fff; color: #000; border: none; }
      .stCaption, [data-testid="stCaptionContainer"] { color: #6f6f6f !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom:1.3rem; padding-bottom:1.2rem; border-bottom:1px solid #161616;">
      <div style="font-size:2.15rem; font-weight:700; letter-spacing:0.18em; color:#fff; text-transform:uppercase;">Agent&nbsp;Inc.</div>
      <div style="color:#7a7a7a; font-size:0.74rem; letter-spacing:0.22em; text-transform:uppercase; margin-top:0.55rem;">RL Environment&nbsp;·&nbsp;Autonomous Business Operations</div>
      <div style="color:#5c5c5c; font-size:0.85rem; margin-top:0.7rem; letter-spacing:0.01em;">SWE-bench taught models to code. Agent Inc. teaches them to run a business.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── stats strip ───────────────────────────────────────────────────────────────
with st.container(border=True):
    counts = data.scenario_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenarios", counts.get("total", 0))
    c2.metric("Easy", counts.get("easy", 0))
    c3.metric("Medium", counts.get("medium", 0))
    c4.metric("Hard", counts.get("hard", 0))

# ── 01 · benchmark / leaderboard ────────────────────────────────────────────────
with st.container(border=True):
    _section("01", "Benchmark", "Leaderboard")
    rows = data.leaderboard_rows()
    if not rows:
        st.info("No calibration data yet — run an eval to populate results/calibration.json.")
    else:
        lb = pd.DataFrame(rows)
        chart_col, table_col = st.columns([1, 1])
        with chart_col:
            st.altair_chart(_pro_bar(lb, "model", "mean_reward", height=300, ylim=(0.0, 1.0)), theme=None)
        with table_col:
            lb_disp = lb.copy()
            lb_disp["success_rate"] = lb_disp["success_rate"] * 100.0  # fraction -> percent
            st.dataframe(
                lb_disp[["model", "mean_reward", "std", "success_rate", "role"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "mean_reward": st.column_config.ProgressColumn(
                        "Mean reward", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "success_rate": st.column_config.NumberColumn("Success", format="%.1f%%"),
                    "std": st.column_config.NumberColumn("± std", format="%.3f"),
                },
            )

# ── 02 · live / run an engagement ───────────────────────────────────────────────
with st.container(border=True):
    _section("02", "Live", "Run an engagement")
    st.caption(
        "Run an agent through one scenario end-to-end — researched, priced, delivered, "
        "and scored by the grader. ~30 seconds."
    )
    _sids = data.scenario_ids()
    lr_a, lr_b, lr_c = st.columns([3, 2, 1])
    sel_sid = lr_a.selectbox(
        "Scenario", _sids, index=(_sids.index("easy_ticket_triage") if "easy_ticket_triage" in _sids else 0)
    )
    sel_model = lr_b.selectbox(
        "Model", ["claude-sonnet-4-6", "gemini-3.1-pro-preview", "Qwen/Qwen3.5-4B"], index=0
    )
    run_now = lr_c.button("Run", type="primary")
    if run_now:
        with st.spinner(f"Running {sel_sid} — {sel_model} (~30s)"):
            try:
                res = live_run.run_engagement(sel_sid, sel_model)
            except Exception as exc:  # gateway hiccup, etc. — fail soft in the demo
                res = None
                st.error(f"Run failed: {exc}")
        if res is not None:
            rc, bc = st.columns([1, 2])
            with rc:
                rw = res["reward"]
                st.metric("Reward", f"{rw:.3f}" if rw is not None else "—")
                st.caption(f"{res['model']} · {res['scenario_id']}")
            with bc:
                if res["subscores"]:
                    bd = pd.DataFrame(res["subscores"])
                    st.altair_chart(_pro_bar(bd, "name", "contribution", height=240, ylim=None), theme=None)
                    st.dataframe(bd, hide_index=True, width="stretch")
                else:
                    st.info("No per-criterion breakdown returned for this run.")

# ── 03 · training / RL improvement (runway + curve) ─────────────────────────────
with st.container(border=True):
    _section("03", "Training", "RL improvement")
    st.caption("The before / after story — the open model's reward, trained toward the frontier.")
    tc = data.training_curve()
    if not tc["points"]:
        st.info("No trained-model reward yet.")
    else:
        runway_col, note_col = st.columns([2, 1])
        with runway_col:
            bars: dict[str, float] = {"Qwen base": tc["points"][0]["reward"]}
            if tc["has_after"]:
                bars["Qwen + RL"] = tc["points"][-1]["reward"]
            bars["RL target"] = tc["target"]
            if tc["frontier_ceiling"] is not None:
                bars["Claude (ceiling)"] = tc["frontier_ceiling"]
            runway_df = pd.DataFrame({"stage": list(bars.keys()), "reward": list(bars.values())})
            st.altair_chart(
                _pro_bar(runway_df, "stage", "reward", height=320, ylim=(0.0, 1.0), sort=list(bars.keys())),
                theme=None,
            )
        with note_col:
            base = tc["points"][0]["reward"]
            st.metric("Qwen base", f"{base:.3f}")
            if tc["has_after"]:
                after = tc["points"][-1]["reward"]
                st.metric("Qwen + RL", f"{after:.3f}", delta=f"{after - base:+.3f}")
            else:
                st.metric("RL target", f"{tc['target']:.2f}", delta="pending training")
            if tc["has_after"]:
                b, a, t = tc["points"][0]["reward"], tc["points"][-1]["reward"], tc["target"]
                beat = "past" if a >= t else "toward"
                st.caption(f"Open model trained {b:.3f} → {a:.3f} ({a - b:+.3f}) — {beat} the {t:.2f} target.")
            elif tc["runway"]:
                st.caption(tc["runway"])

    prog = data.training_progress()
    st.subheader("Training curve — reward per RL step")
    if not prog["available"]:
        st.caption(
            "Awaiting the RL training run — this populates automatically once the "
            "per-step training rewards are written (baseline → GRPO steps → final)."
        )
    else:
        curve_col, stat_col = st.columns([3, 1])
        with curve_col:
            curve_df = pd.DataFrame(prog["points"])
            st.altair_chart(_pro_line(curve_df, "step", "reward", height=300), theme=None)
        with stat_col:
            if prog["baseline"] is not None:
                st.metric("Baseline", f"{prog['baseline']:.3f}")
            if prog["final"] is not None:
                st.metric(
                    "After training",
                    f"{prog['final']:.3f}",
                    delta=(f"{prog['delta']:+.3f}" if prog["delta"] is not None else None),
                )
            st.caption(f"{len(prog['points'])} step points")

# ── 04 · detail / per-scenario breakdown ────────────────────────────────────────
with st.container(border=True):
    _section("04", "Detail", "Per-scenario breakdown")
    runs = data.load_runs()
    if not runs:
        st.info("No per-run data found.")
    else:
        runs_df = pd.DataFrame(runs)

        # reward heatmap: scenario x model (matplotlib-free red->green coloring)
        def _heat(v):
            if pd.isna(v):
                return ""
            v = max(0.0, min(1.0, float(v)))
            r = int(255 * min(1.0, 2 * (1 - v)))  # red -> yellow -> green
            g = int(255 * min(1.0, 2 * v))
            return f"background-color: rgba({r}, {g}, 90, 0.6); color: #fff"

        pivot = runs_df.pivot_table(index="scenario_id", columns="model", values="reward")
        styler = pivot.style.format("{:.2f}", na_rep="—")
        styler = (styler.map if hasattr(styler, "map") else styler.applymap)(_heat)
        st.subheader("Reward by scenario × model")
        st.dataframe(styler, width="stretch")

        st.subheader("Criterion breakdown")
        sel1, sel2 = st.columns(2)
        model = sel1.selectbox("Model", sorted(runs_df["model"].unique()))
        scenario = sel2.selectbox("Scenario", sorted(runs_df["scenario_id"].unique()))
        match = [r for r in runs if r["model"] == model and r["scenario_id"] == scenario]
        if match:
            subs = match[0]["subscores"]
            bd = pd.DataFrame(
                [
                    {
                        "criterion": k,
                        "value": v.get("value", 0.0),
                        "weight": v.get("weight", 0.0),
                        "contribution": round(v.get("value", 0.0) * v.get("weight", 0.0), 3),
                    }
                    for k, v in subs.items()
                ]
            )
            bcol, tcol = st.columns([1, 1])
            with bcol:
                st.altair_chart(_pro_bar(bd, "criterion", "contribution", height=280, ylim=None), theme=None)
            with tcol:
                st.dataframe(bd, hide_index=True, width="stretch")
                st.metric("Total reward", f"{match[0]['reward']:.3f}")
        else:
            st.warning("No run for that model × scenario combination.")

# ── footer: data provenance ──────────────────────────────────────────────────────
status = data.data_status()
leaderboard_bit = "real calibration" if status["leaderboard_real"] else "missing"
runs_bit = (
    f"sample ({status['runs_file']})" if status["runs_are_sample"] else f"real ({status['runs_file']})"
)
st.markdown(
    f"""
    <div style="color:#5c5c5c; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; margin-top:0.8rem;">
      Leaderboard — <b style="color:#8a8a8a;">{leaderboard_bit}</b>
      &nbsp;·&nbsp; Per-run — <b style="color:#8a8a8a;">{runs_bit}</b>
    </div>
    """,
    unsafe_allow_html=True,
)
