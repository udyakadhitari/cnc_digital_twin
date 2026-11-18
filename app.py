import streamlit as st
import time
import pandas as pd
import altair as alt
import os

# Inject the OpenAI key from Streamlit secrets into the environment
# so agent.py can read it via os.getenv("OPENAI_API_KEY")
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from src.app.simulator_controller import SimulatorController
from src.llm_tools.twin_interfaces import TwinLLMInterface
from src.agent.agent import CNCAgent

# ---------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS  (built once per process, not per rerun)
# ---------------------------------------------------------------------------
SPEED_MAP = {"0.25x": 1.0, "1x": 0.2, "5x": 0.04, "10x": 0.01, "Fast Forward": 0.0}
TOOL_OPTIONS = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9]}
MACHINE_TOOL_MAP = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9]}
CHART_WINDOW = 300   # max data-points rendered in live charts

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CNC Digital Twin", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------------------
if "simulator" not in st.session_state:
    st.session_state.simulator = SimulatorController()
if "history" not in st.session_state:
    st.session_state.history = []
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame()
if "telemetry_df" not in st.session_state:
    st.session_state.telemetry_df = pd.DataFrame()
if "running" not in st.session_state:
    st.session_state.running = False
if "selected_speed" not in st.session_state:
    st.session_state.selected_speed = "1x"
if "show_actual" not in st.session_state:
    st.session_state.show_actual = False
if "agent" not in st.session_state:
    twin_interface = TwinLLMInterface(st.session_state.simulator.twin)
    st.session_state.agent = CNCAgent(twin_interface)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

sim = st.session_state.simulator
agent = st.session_state.agent

# ---------------------------------------------------------------------------
# SIDEBAR  (static — never reruns during simulation)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    st.markdown("### Simulator State")
    if st.session_state.running:
        st.success("🟢 LIVE - STREAMING")
    elif sim.current_idx > 0 and sim.has_next():
        st.warning("⏸ PAUSED")
    elif not sim.has_next():
        st.error("⏹ EOL REACHED")
    else:
        st.info("⚪ READY")

    st.markdown("---")

    st.markdown("### Controls")
    col1, col2, col3 = st.columns(3)
    if col1.button("▶ Start", disabled=st.session_state.running or not sim.has_next()):
        st.session_state.running = True
        st.rerun()
    if col2.button("⏸ Pause", disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()
    if col3.button("↻ Reset"):
        st.session_state.simulator = SimulatorController()
        st.session_state.history = []
        st.session_state.history_df = pd.DataFrame()
        st.session_state.telemetry_df = pd.DataFrame()
        st.session_state.running = False
        st.session_state.chat_history = []
        twin_interface = TwinLLMInterface(st.session_state.simulator.twin)
        st.session_state.agent = CNCAgent(twin_interface)
        st.rerun()

    selected_speed = st.selectbox(
        "Simulation Speed", list(SPEED_MAP.keys()),
        index=list(SPEED_MAP.keys()).index(st.session_state.selected_speed)
    )
    st.session_state.selected_speed = selected_speed

    st.markdown("---")
    st.markdown("### Twin Target")
    m_idx = [1, 2, 3].index(sim.live_machine) if sim.live_machine in [1, 2, 3] else 2
    sel_machine = st.selectbox("Machine", [1, 2, 3], index=m_idx)

    opts = TOOL_OPTIONS.get(sel_machine, [9])
    t_idx = opts.index(sim.live_tool) if sim.live_tool in opts else 0
    sel_tool = st.selectbox("Tool", opts, index=t_idx)

    if st.button("🔄 Swap Live Source", use_container_width=True):
        st.session_state.simulator = SimulatorController(machine_override=sel_machine, tool_override=sel_tool)
        st.session_state.history = []
        st.session_state.history_df = pd.DataFrame()
        st.session_state.telemetry_df = pd.DataFrame()
        st.session_state.running = False
        st.session_state.chat_history = []
        twin_interface = TwinLLMInterface(st.session_state.simulator.twin)
        st.session_state.agent = CNCAgent(twin_interface)
        st.rerun()

    st.markdown("---")
    st.markdown("### System Status")
    st.markdown(f"**Data Stream:** {len(sim.live_stream)} rows")
    st.markdown(f"**Digital Twin:** Healthy")
    st.markdown(f"**Model:** `{sim.twin.predictor.model_path.name}`")
    key_status = "Connected" if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("sk-placeholder") else "Mock / Fallback"
    st.markdown(f"**LLM API:** {key_status}")

    st.markdown("---")
    st.markdown("### Evaluation")
    st.session_state.show_actual = st.checkbox(
        "👁️ Reveal True Dataset Wear", value=st.session_state.show_actual,
        help="Plots the actual hidden dataset wear for visual comparison without leaking it to the AI."
    )

# ---------------------------------------------------------------------------
# MAIN HEADER  (static)
# ---------------------------------------------------------------------------
st.title("CNC Tool Wear Digital Twin")
st.markdown(f"**Machine:** {sim.live_machine} | **Tool:** {sim.live_tool} | **Model:** `{sim.twin.predictor.model_path.name}`")


# ---------------------------------------------------------------------------
# LIVE FRAGMENT — only this section reruns each tick, everything else frozen
# ---------------------------------------------------------------------------
@st.fragment(run_every=SPEED_MAP[st.session_state.selected_speed] if st.session_state.running else None)
def live_dashboard():
    sim_inner = st.session_state.simulator

    # ── Advance one simulation step ──────────────────────────────────────────
    if st.session_state.running:
        if sim_inner.has_next():
            current_idx = sim_inner.current_idx
            state = sim_inner.step()

            actuals = sim_inner.get_actual_metrics(current_idx)
            state["actual_wear"] = actuals.get("wear")
            state["actual_rul_150"] = actuals.get("rul_150_min")

            st.session_state.history.append(state)

            st.session_state.history_df = pd.concat(
                [st.session_state.history_df, pd.DataFrame([state])],
                ignore_index=True
            )
            if "latest_telemetry" in state:
                st.session_state.telemetry_df = pd.concat(
                    [st.session_state.telemetry_df, pd.DataFrame([state["latest_telemetry"]])],
                    ignore_index=True
                )
        else:
            st.session_state.running = False
            st.rerun()   # full rerun only on EOL to update sidebar badge

    # ── KPI Cards ────────────────────────────────────────────────────────────
    current_state = st.session_state.history[-1] if st.session_state.history else None

    if current_state:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Running Time",   f"{current_state['contact_time']:.1f} s")
        k2.metric("Predicted Wear", f"{current_state['predicted_wear']:.2f} µm")
        k3.metric("Wear Rate",      f"{current_state['wear_rate']:.3f} µm/s")
        rul = current_state['RUL_150']
        rul_str = f"{rul:.1f} m" if rul != float('inf') else "∞"
        k4.metric("RUL (150µm)",  rul_str)
        k5.metric("Health State", current_state['health_state'])
        k6.metric("Confidence",   current_state['confidence'])
    else:
        st.info("Start the simulation to stream CNC telemetry.")

    # ── Live Charts ───────────────────────────────────────────────────────────
    if not st.session_state.history_df.empty:
        df_hist = st.session_state.history_df.tail(CHART_WINDOW)

        g1, g2 = st.columns(2)

        with g1:
            st.markdown("### Live Wear Progression")
            base = alt.Chart(df_hist).mark_line(color='#1f77b4', strokeWidth=3).encode(
                x=alt.X('contact_time:Q', title="Contact Time (s)"),
                y=alt.Y('predicted_wear:Q', title="Predicted Wear (µm)",
                        scale=alt.Scale(domain=[0, max(160, float(df_hist['predicted_wear'].max()) + 10)]))
            )
            eol_line = alt.Chart(pd.DataFrame({'y': [150]})).mark_rule(
                color='red', strokeDash=[5, 5]
            ).encode(y='y:Q')

            chart_1 = base + eol_line
            if (st.session_state.show_actual
                    and "actual_wear" in df_hist.columns
                    and not df_hist["actual_wear"].isnull().all()):
                chart_1 += alt.Chart(df_hist).mark_line(
                    color='#2ca02c', strokeDash=[3, 3], strokeWidth=2
                ).encode(x='contact_time:Q', y='actual_wear:Q')

            st.altair_chart(chart_1, use_container_width=True)

        with g2:
            st.markdown("### Remaining Useful Life")
            df_rul = df_hist.copy()
            df_rul['RUL_150'] = df_rul['RUL_150'].clip(upper=500)

            rul_base = alt.Chart(df_rul).mark_area(color='#ff7f0e', opacity=0.3).encode(
                x=alt.X('contact_time:Q', title="Contact Time (s)"),
                y=alt.Y('RUL_150:Q', title="RUL (minutes)")
            )
            rul_line = alt.Chart(df_rul).mark_line(color='#ff7f0e', strokeWidth=2).encode(
                x='contact_time:Q', y='RUL_150:Q'
            )
            chart_2 = rul_base + rul_line
            if (st.session_state.show_actual
                    and "actual_rul_150" in df_rul.columns
                    and not df_rul["actual_rul_150"].isnull().all()):
                chart_2 += alt.Chart(df_rul).mark_line(
                    color='#2ca02c', strokeDash=[3, 3], strokeWidth=2
                ).encode(x='contact_time:Q', y='actual_rul_150:Q')

            st.altair_chart(chart_2, use_container_width=True)

    # ── Telemetry Expander ───────────────────────────────────────────────────
    if not st.session_state.telemetry_df.empty:
        with st.expander("Live Sensor Telemetry Data", expanded=False):
            st.dataframe(
                st.session_state.telemetry_df.iloc[::-1].head(50),
                use_container_width=True
            )


live_dashboard()


# ---------------------------------------------------------------------------
# BOTTOM PANELS  (static — rendered once, never reruns during simulation)
# ---------------------------------------------------------------------------
b1, b2 = st.columns([1, 1])

with b1:
    st.markdown("### Model Information")
    train_tools = [t for t in range(1, 10) if t != sim.live_tool]
    calib_tools = [t for t in MACHINE_TOOL_MAP[sim.live_machine] if t != sim.live_tool]

    st.markdown(f"- **Training Dataset**: Tools {train_tools}")
    st.markdown(f"- **Calibration Dataset**: Tools {calib_tools}")
    st.markdown(f"- **Held-out Test Target**: Tool {sim.live_tool}")
    st.markdown("- **Model Framework**: `ExtraTreesRegressor` + `IsotonicRegression`")

    st.markdown("### Feature Importance")
    try:
        # Model already in memory — zero disk I/O
        model    = sim.twin.predictor.model
        features = sim.twin.features
        imp      = model.feature_importances_
        fi_df    = (pd.DataFrame({"Feature": features, "Importance": imp})
                      .sort_values("Importance", ascending=False)
                      .head(5))
        fi_chart = alt.Chart(fi_df).mark_bar().encode(
            x=alt.X('Importance:Q'),
            y=alt.Y('Feature:N', sort='-x'),
            color=alt.Color('Importance:Q', scale=alt.Scale(scheme='blues'), legend=None)
        )
        st.altair_chart(fi_chart, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")

with b2:
    st.markdown("### AI Diagnostic Assistant")

    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about tool wear, RUL, or trends..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing Digital Twin..."):
                    response = agent.chat(prompt)
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
