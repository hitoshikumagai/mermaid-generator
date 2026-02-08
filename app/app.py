from pathlib import Path
import sys

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.state import StreamlitFlowState

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.mermaid_generator.graph_logic import (  # noqa: E402
    calculate_layout_positions,
    export_to_mermaid,
)
from src.mermaid_generator.orchestrator import (  # noqa: E402
    FlowchartOrchestrator,
)
from src.mermaid_generator.ui_mapper import (  # noqa: E402
    flow_items_to_graph_data,
    to_flow_edge_specs,
    to_flow_node_specs,
)


@st.cache_resource
def get_orchestrator() -> FlowchartOrchestrator:
    return FlowchartOrchestrator()


def to_flow_state(nodes_data: list, edges_data: list, positions: dict) -> StreamlitFlowState:
    node_specs = to_flow_node_specs(nodes_data, positions)
    edge_specs = to_flow_edge_specs(edges_data)
    flow_nodes = [StreamlitFlowNode(**spec) for spec in node_specs]
    flow_edges = [StreamlitFlowEdge(**spec) for spec in edge_specs]
    return StreamlitFlowState(nodes=flow_nodes, edges=flow_edges)


def ensure_state() -> None:
    if "flow_state" not in st.session_state:
        st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
    if "graph_data" not in st.session_state:
        st.session_state.graph_data = None
    if "scope_summary" not in st.session_state:
        st.session_state.scope_summary = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "impact" not in st.session_state:
        st.session_state.impact = {
            "phase": "initial",
            "message": "No diagram yet.",
            "impacted_node_ids": [],
            "impacted_edge_ids": [],
        }
    if "mode" not in st.session_state:
        st.session_state.mode = "scope"
    if "source" not in st.session_state:
        st.session_state.source = "fallback"


def run_agent_turn(user_message: str) -> None:
    orchestrator = get_orchestrator()
    turn = orchestrator.run_turn(
        user_message=user_message,
        chat_history=st.session_state.chat_history,
        current_scope=st.session_state.scope_summary,
        current_graph=st.session_state.graph_data,
    )

    st.session_state.chat_history.append({"role": "assistant", "content": turn.assistant_message})
    st.session_state.scope_summary = turn.scope_summary
    st.session_state.impact = turn.impact
    st.session_state.mode = turn.mode
    st.session_state.source = turn.source

    if turn.graph_data:
        st.session_state.graph_data = turn.graph_data
        positions = calculate_layout_positions(turn.graph_data["nodes"], turn.graph_data["edges"])
        st.session_state.flow_state = to_flow_state(
            turn.graph_data["nodes"], turn.graph_data["edges"], positions
        )


st.set_page_config(layout="wide")
st.title("Flowchart Copilot")
ensure_state()

with st.sidebar:
    st.markdown("### Agent Status")
    st.write(f"`mode`: {st.session_state.mode}")
    st.write(f"`source`: {st.session_state.source}")
    st.markdown("### Scope Summary")
    st.caption(st.session_state.scope_summary or "(empty)")

prompt = st.chat_input("要件を入力してください（初回はスコープ、2回目以降は変更指示）")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    run_agent_turn(prompt)

st.subheader("Copilot Chat")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

st.subheader("Editor")
st.caption("Drag nodes to refine layout")

curr_state = streamlit_flow(
    "flow",
    st.session_state.flow_state,
    fit_view=True,
    height=520,
)

current_nodes, current_edges = flow_items_to_graph_data(curr_state.nodes, curr_state.edges)
mermaid_text = export_to_mermaid(current_nodes, current_edges)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Mermaid Export")
    st.code(mermaid_text, language="mermaid")
with col2:
    st.markdown("### Impact Range")
    st.json(st.session_state.impact, expanded=False)
    st.markdown("### Debug")
    st.json({"node_count": len(curr_state.nodes), "edge_count": len(curr_state.edges)}, expanded=False)

st.caption("Set OPENAI_API_KEY to enable real LLM orchestration.")
