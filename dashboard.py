"""Agent Inc. — dashboard (P2).

Run it:

    uv run --with streamlit --with pandas streamlit run dashboard.py

Sections: leaderboard · live engagement · RL runway + training curve · per-scenario
breakdown. All data access goes through dashboard_data.py — this file is the view
(styling included). Theme lives in .streamlit/config.toml.
"""

import pandas as pd
import streamlit as st

import dashboard_data as data
import live_run

ACCENT = "#6E8BFF"  # brand accent used for chart series

st.set_page_config(page_title="Agent Inc.", page_icon="📊", layout="wide")

# ── presentation: typography, spacing, hide Streamlit chrome, KPI cards ──────────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      html, body, [class*="css"], [data-testid="stMarkdownContainer"] { font-family: 'Inter', system-ui, sans-serif; }
      #MainMenu, [data-testid="stToolbar"], [data-testid="stStatusWidget"], footer, [data-testid="stDecoration"] { display: none !important; }
      header[data-testid="stHeader"] { background: transparent; height: 0; }
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1380px; }
      h2 { font-size: 1.28rem !important; font-weight: 600 !important; letter-spacing: -0.01em;
           border-left: 3px solid #6E8BFF; padding-left: 0.7rem !important; margin: 0.5rem 0 0.1rem; }
      h3 { font-size: 1.0rem !important; font-weight: 600 !important; color: #C3C9D8 !important; }
      hr { margin: 1.3rem 0 !important; border-color: #1E2533 !important; }
      [data-testid="stMetric"] { background: #141925; border: 1px solid #232A3A; border-radius: 10px; padding: 0.8rem 1rem; }
      [data-testid="stMetricLabel"] p { opacity: 0.6; font-size: 0.74rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
      [data-testid="stMetricValue"] { font-weight: 700; }
      [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #232A3A; border-radius: 8px; }
      .stCaption, [data-testid="stCaptionContainer"] { color: #7E869B !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom:1.1rem;">
      <div style="display:flex; align-items:baseline; gap:0.75rem; flex-wrap:wrap;">
        <span style="font-size:1.85rem; font-weight:700; letter-spacing:-0.02em; color:#F2F4FA;">Agent&nbsp;Inc.</span>
        <span style="color:#7E869B; font-size:0.92rem; font-weight:500;">An RL environment for autonomous business operations</span>
      </div>
      <div style="color:#737B90; font-size:0.9rem; margin-top:0.3rem;">
        SWE-bench taught models to code. Agent Inc. teaches them to run a business.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

counts = data.scenario_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Scenarios", counts.get("total", 0))
c2.metric("Easy", counts.get("easy", 0))
c3.metric("Medium", counts.get("medium", 0))
c4.metric("Hard", counts.get("hard", 0))

st.divider()

# ── leaderboard ───────────────────────────────────────────────────────────────
st.header("Leaderboard")
rows = data.leaderboard_rows()
if not rows:
    st.info("No calibration data yet — run an eval to populate results/calibration.json.")
else:
    lb = pd.DataFrame(rows)
    chart_col, table_col = st.columns([1, 1])
    with chart_col:
        st.bar_chart(lb.set_index("model")["mean_reward"], height=300, color=ACCENT)
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

st.divider()

# ── live: run one engagement end-to-end, scored on the spot ─────────────────────
st.header("Run a live engagement")
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
                st.bar_chart(bd.set_index("name")["contribution"], height=240, color=ACCENT)
                st.dataframe(bd, hide_index=True, width="stretch")
            else:
                st.info("No per-criterion breakdown returned for this run.")

st.divider()

# ── RL runway ───────────────────────────────────────────────────────────────────
st.header("RL runway")
st.caption("The before / after story — the open model's reward, trained toward the frontier.")
tc = data.training_curve()
if not tc["points"]:
    st.info("No trained-model reward yet.")
else:
    runway_col, note_col = st.columns([2, 1])
    with runway_col:
        # bars read clearly even with just the base point; becomes a before/after
        # story once a trained "Qwen + RL" number lands.
        bars: dict[str, float] = {"Qwen base": tc["points"][0]["reward"]}
        if tc["has_after"]:
            bars["Qwen + RL"] = tc["points"][-1]["reward"]
        bars["RL target"] = tc["target"]
        if tc["frontier_ceiling"] is not None:
            bars["Claude (ceiling)"] = tc["frontier_ceiling"]
        st.bar_chart(pd.DataFrame({"reward": bars}), height=320, color=ACCENT)
    with note_col:
        base = tc["points"][0]["reward"]
        st.metric("Qwen base", f"{base:.3f}")
        if tc["has_after"]:
            after = tc["points"][-1]["reward"]
            st.metric("Qwen + RL", f"{after:.3f}", delta=f"{after - base:+.3f}")
        else:
            st.metric("RL target", f"{tc['target']:.2f}", delta="pending training")
        if tc["runway"]:
            st.caption(tc["runway"])

# the per-step learning curve — fills in once scripts/rl_train.py runs
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
        cdf = pd.DataFrame(prog["points"]).set_index("step")["reward"]
        st.line_chart(cdf.to_frame("Qwen reward"), height=300, color=ACCENT)
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

st.divider()

# ── per-scenario breakdown ────────────────────────────────────────────────────
st.header("Per-scenario breakdown")
runs = data.load_runs()
if not runs:
    st.info("No per-run data found.")
else:
    runs_df = pd.DataFrame(runs)

    # reward heatmap: scenario x model (matplotlib-free red->green coloring)
    def _heat(v):
        if pd.isna(v):
            return ""
        r = int(255 * min(1.0, 2 * (1 - v)))
        g = int(255 * min(1.0, 2 * v))
        return f"background-color: rgba({r}, {g}, 90, 0.55)"

    pivot = runs_df.pivot_table(index="scenario_id", columns="model", values="reward")
    styler = pivot.style.format("{:.2f}", na_rep="—")
    styler = (styler.map if hasattr(styler, "map") else styler.applymap)(_heat)
    st.subheader("Reward by scenario × model")
    st.dataframe(styler, width="stretch")

    # per-criterion breakdown for a chosen run
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
            st.bar_chart(bd.set_index("criterion")["contribution"], height=280, color=ACCENT)
        with tcol:
            st.dataframe(bd, hide_index=True, width="stretch")
            st.metric("Total reward", f"{match[0]['reward']:.3f}")
    else:
        st.warning("No run for that model × scenario combination.")

# ── footer: data provenance ──────────────────────────────────────────────────────
st.divider()
status = data.data_status()
leaderboard_bit = "real calibration" if status["leaderboard_real"] else "missing"
runs_bit = (
    f"sample ({status['runs_file']})" if status["runs_are_sample"] else f"real ({status['runs_file']})"
)
st.markdown(
    f"""
    <div style="color:#6B7387; font-size:0.82rem; margin-top:0.3rem;">
      Leaderboard: <b style="color:#9AA3B8;">{leaderboard_bit}</b>
      &nbsp;·&nbsp; Per-run: <b style="color:#9AA3B8;">{runs_bit}</b>
    </div>
    """,
    unsafe_allow_html=True,
)
