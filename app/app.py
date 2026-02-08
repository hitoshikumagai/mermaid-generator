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
    build_mock_graph,
    calculate_layout_positions,
    export_to_mermaid,
)
from src.mermaid_generator.ui_mapper import (  # noqa: E402
    flow_items_to_graph_data,
    to_flow_edge_specs,
    to_flow_node_specs,
)


def to_flow_state(nodes_data: list, edges_data: list, positions: dict) -> StreamlitFlowState:
    node_specs = to_flow_node_specs(nodes_data, positions)
    edge_specs = to_flow_edge_specs(edges_data)
    flow_nodes = [StreamlitFlowNode(**spec) for spec in node_specs]
    flow_edges = [StreamlitFlowEdge(**spec) for spec in edge_specs]
    return StreamlitFlowState(nodes=flow_nodes, edges=flow_edges)


st.set_page_config(layout="wide")
st.title("AI Flowchart Builder")

with st.sidebar:
    topic = st.text_input("Topic", "EC checkout flow")
    generate_btn = st.button("Generate")

if "flow_state" not in st.session_state:
    st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])

if generate_btn:
    with st.spinner("Generating..."):
        graph_data = build_mock_graph(topic)
        positions = calculate_layout_positions(graph_data["nodes"], graph_data["edges"])
        st.session_state.flow_state = to_flow_state(
            graph_data["nodes"], graph_data["edges"], positions
        )

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
    st.markdown("### Debug")
    st.json({"node_count": len(curr_state.nodes), "edge_count": len(curr_state.edges)})

st.caption("Tip: Replace the mock graph function with a real JSON-only LLM call.")
