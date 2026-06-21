"""Agent Inc. — live demo dashboard (P2).

Run it:

    uv run --with streamlit --with pandas streamlit run dashboard.py

Three panels:
  1. Leaderboard      — real Phase-1 calibration (results/calibration.json)
  2. RL runway        — Qwen base -> target, with the frontier ceiling
  3. Per-scenario     — reward heatmap + per-criterion breakdown (sample until
                        P1 exports real per-run results; see dashboard_data.py)

All data access goes through dashboard_data.py — this file is pure view.
"""

import pandas as pd
import streamlit as st

import dashboard_data as data

st.set_page_config(page_title="Agent Inc. — Dashboard", page_icon="📊", layout="wide")

# ── header ────────────────────────────────────────────────────────────────────
st.title("📊 Agent Inc.")
st.caption("SWE-bench taught models to code. **Agent Inc. teaches them to run a business.**")

counts = data.scenario_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Scenarios", counts.get("total", 0))
c2.metric("Easy", counts.get("easy", 0))
c3.metric("Medium", counts.get("medium", 0))
c4.metric("Hard", counts.get("hard", 0))

st.divider()

# ── 1. leaderboard ──────────────────────────────────────────────────────────────
st.header("🏆 Leaderboard")
rows = data.leaderboard_rows()
if not rows:
    st.info("No calibration data yet — run `hud eval` to populate results/calibration.json.")
else:
    lb = pd.DataFrame(rows)
    chart_col, table_col = st.columns([1, 1])
    with chart_col:
        st.bar_chart(lb.set_index("model")["mean_reward"], height=300)
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

# ── 2. RL runway ────────────────────────────────────────────────────────────────
st.header("📈 RL runway — the before / after story")
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
        st.bar_chart(pd.DataFrame({"reward": bars}), height=320)
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
        "⏳ Not run yet — auto-plots once `scripts/rl_train.py` writes "
        "`results/training_curve.jsonl` (baseline → GRPO steps → final)."
    )
else:
    curve_col, stat_col = st.columns([3, 1])
    with curve_col:
        cdf = pd.DataFrame(prog["points"]).set_index("step")["reward"]
        st.line_chart(cdf.to_frame("Qwen reward"), height=300)
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

# ── 3. per-scenario breakdown ────────────────────────────────────────────────────
st.header("🔬 Per-scenario breakdown")
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
            st.bar_chart(bd.set_index("criterion")["contribution"], height=280)
        with tcol:
            st.dataframe(bd, hide_index=True, width="stretch")
            st.metric("Total reward", f"{match[0]['reward']:.3f}")
    else:
        st.warning("No run for that model × scenario combination.")

# ── footer: data provenance ──────────────────────────────────────────────────────
st.divider()
status = data.data_status()
bits = []
bits.append("🟢 leaderboard: **real** calibration" if status["leaderboard_real"] else "🔴 leaderboard: missing")
bits.append(
    f"🟡 per-run: **sample** (`{status['runs_file']}`) — swap for real export"
    if status["runs_are_sample"]
    else f"🟢 per-run: real (`{status['runs_file']}`)"
)
st.caption(" · ".join(bits))
