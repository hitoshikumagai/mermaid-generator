from pathlib import Path
import sys
import html

import streamlit as st
import streamlit.components.v1 as components
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
    MermaidDiagramOrchestrator,
)
from src.mermaid_generator.templates import (  # noqa: E402
    DIAGRAM_TYPES,
    get_flowchart_template,
    get_mermaid_template,
    list_flowchart_templates,
    list_mermaid_templates,
)
from src.mermaid_generator.ui_mapper import (  # noqa: E402
    flow_items_to_graph_data,
    to_flow_edge_specs,
    to_flow_node_specs,
)


@st.cache_resource
def get_orchestrator() -> FlowchartOrchestrator:
    return FlowchartOrchestrator()


@st.cache_resource
def get_mermaid_orchestrator() -> MermaidDiagramOrchestrator:
    return MermaidDiagramOrchestrator()


def mermaid_editor_key(diagram_type: str) -> str:
    return f"mermaid_editor_{diagram_type.lower()}"


def to_flow_state(nodes_data: list, edges_data: list, positions: dict) -> StreamlitFlowState:
    node_specs = to_flow_node_specs(nodes_data, positions, edges_data)
    edge_specs = to_flow_edge_specs(edges_data, positions)
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
    if "diagram_type" not in st.session_state:
        st.session_state.diagram_type = "Flowchart"
    if "mermaid_code_by_type" not in st.session_state:
        st.session_state.mermaid_code_by_type = {}
    if "mermaid_chat_history_by_type" not in st.session_state:
        st.session_state.mermaid_chat_history_by_type = {}
    if "mermaid_agent_state_by_type" not in st.session_state:
        st.session_state.mermaid_agent_state_by_type = {}
    if "edit_mode_by_type" not in st.session_state:
        st.session_state.edit_mode_by_type = {}
    if "default_template_loaded" not in st.session_state:
        st.session_state.default_template_loaded = False

    for diagram_type in DIAGRAM_TYPES:
        if diagram_type != "Flowchart":
            if diagram_type not in st.session_state.mermaid_chat_history_by_type:
                st.session_state.mermaid_chat_history_by_type[diagram_type] = []
            if diagram_type not in st.session_state.mermaid_agent_state_by_type:
                st.session_state.mermaid_agent_state_by_type[diagram_type] = {
                    "phase": "initial",
                    "source": "fallback",
                    "message": "No Mermaid diagram yet.",
                }
            if diagram_type not in st.session_state.mermaid_code_by_type:
                templates = list_mermaid_templates(diagram_type)
                if templates:
                    st.session_state.mermaid_code_by_type[diagram_type] = get_mermaid_template(
                        diagram_type, templates[0]["id"]
                    )
                else:
                    st.session_state.mermaid_code_by_type[diagram_type] = ""
            editor_key = mermaid_editor_key(diagram_type)
            if editor_key not in st.session_state:
                st.session_state[editor_key] = st.session_state.mermaid_code_by_type[diagram_type]
        if diagram_type not in st.session_state.edit_mode_by_type:
            st.session_state.edit_mode_by_type[diagram_type] = (
                "Orchestration" if diagram_type == "Flowchart" else "Manual"
            )


def get_mermaid_code(diagram_type: str) -> str:
    return str(st.session_state.mermaid_code_by_type.get(diagram_type, ""))


def set_mermaid_code(diagram_type: str, code: str) -> None:
    st.session_state.mermaid_code_by_type[diagram_type] = code
    st.session_state[mermaid_editor_key(diagram_type)] = code


def apply_graph_data(graph_data: dict, impact_message: str) -> None:
    st.session_state.graph_data = graph_data
    positions = calculate_layout_positions(graph_data["nodes"], graph_data["edges"])
    st.session_state.flow_state = to_flow_state(graph_data["nodes"], graph_data["edges"], positions)
    st.session_state.impact = {
        "phase": "initial",
        "message": impact_message,
        "impacted_node_ids": sorted([node["id"] for node in graph_data["nodes"]]),
        "impacted_edge_ids": sorted([edge["id"] for edge in graph_data["edges"]]),
        "added_node_ids": sorted([node["id"] for node in graph_data["nodes"]]),
        "removed_node_ids": [],
        "changed_node_ids": [],
        "added_edge_ids": sorted([edge["id"] for edge in graph_data["edges"]]),
        "removed_edge_ids": [],
        "changed_edge_ids": [],
    }


def apply_flowchart_template(template_id: str) -> None:
    template = get_flowchart_template(template_id)
    st.session_state.scope_summary = f"Template: {template['name']} - {template['description']}"
    apply_graph_data(template["graph"], f"Template loaded: {template['name']}")


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


def run_mermaid_agent_turn(diagram_type: str, user_message: str) -> None:
    orchestrator = get_mermaid_orchestrator()
    history = st.session_state.mermaid_chat_history_by_type.setdefault(diagram_type, [])
    current_code = get_mermaid_code(diagram_type)
    turn = orchestrator.run_turn(
        diagram_type=diagram_type,
        user_message=user_message,
        chat_history=history,
        current_code=current_code,
    )
    history.append({"role": "assistant", "content": turn.assistant_message})
    set_mermaid_code(diagram_type, turn.mermaid_code)
    st.session_state.mermaid_agent_state_by_type[diagram_type] = {
        "phase": turn.phase,
        "source": turn.source,
        "message": turn.change_summary,
    }


def render_impact_summary(impact: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Added Nodes", len(impact.get("added_node_ids", [])))
    c2.metric("Changed Nodes", len(impact.get("changed_node_ids", [])))
    c3.metric("Added/Changed Edges", len(impact.get("added_edge_ids", [])) + len(impact.get("changed_edge_ids", [])))


def render_mermaid_preview(mermaid_code: str, height: int = 520) -> None:
    escaped = html.escape(mermaid_code or "")
    mermaid_html = f"""
<div style="padding: 8px;">
  <pre class="mermaid">{escaped}</pre>
  <div id="render_error" style="color:#b91c1c;font-family:monospace;"></div>
</div>
<script>
  function renderMermaid() {{
    try {{
      mermaid.initialize({{ startOnLoad: false, securityLevel: "loose" }});
      const nodes = document.querySelectorAll(".mermaid");
      mermaid.run({{ nodes }}).catch((err) => {{
        document.getElementById("render_error").textContent = "Mermaid render error: " + err;
      }});
    }} catch (err) {{
      document.getElementById("render_error").textContent = "Mermaid init error: " + err;
    }}
  }}

  if (window.mermaid) {{
    renderMermaid();
  }} else {{
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
    script.onload = renderMermaid;
    script.onerror = function() {{
      document.getElementById("render_error").textContent = "Failed to load Mermaid runtime.";
    }};
    document.head.appendChild(script);
  }}
</script>
"""
    components.html(mermaid_html, height=height, scrolling=True)


st.set_page_config(layout="wide")
st.title("Flowchart Copilot")
ensure_state()

if not st.session_state.default_template_loaded:
    apply_flowchart_template("ec_purchase")
    st.session_state.default_template_loaded = True

selected_mode = st.session_state.edit_mode_by_type.get(st.session_state.diagram_type, "Manual")

with st.sidebar:
    st.markdown("### Diagram Type")
    diagram_type = st.selectbox("Type", DIAGRAM_TYPES, key="diagram_type")
    current_mode = st.session_state.edit_mode_by_type.get(
        diagram_type, "Orchestration" if diagram_type == "Flowchart" else "Manual"
    )
    selected_mode = st.radio(
        "Work Mode",
        ["Orchestration", "Manual"],
        index=0 if current_mode == "Orchestration" else 1,
        horizontal=True,
    )
    st.session_state.edit_mode_by_type[diagram_type] = selected_mode

    if diagram_type == "Flowchart":
        templates = list_flowchart_templates()
        template_ids = [tpl["id"] for tpl in templates]
        template_lookup = {tpl["id"]: tpl for tpl in templates}

        if "selected_flowchart_template_id" not in st.session_state:
            st.session_state.selected_flowchart_template_id = "ec_purchase"

        selected_id = st.selectbox(
            "Common Template",
            template_ids,
            format_func=lambda x: template_lookup[x]["name"],
            key="selected_flowchart_template_id",
        )
        st.caption(template_lookup[selected_id]["description"])

        col_l, col_r = st.columns(2)
        if col_l.button("Load Template", use_container_width=True):
            apply_flowchart_template(selected_id)
        if col_r.button("Start Blank", use_container_width=True):
            st.session_state.graph_data = None
            st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
            st.session_state.scope_summary = ""
            st.session_state.impact = {
                "phase": "initial",
                "message": "Blank canvas.",
                "impacted_node_ids": [],
                "impacted_edge_ids": [],
            }

        if selected_mode == "Orchestration":
            st.markdown("### Agent Status")
            st.write(f"`mode`: {st.session_state.mode}")
            st.write(f"`source`: {st.session_state.source}")
            st.markdown("### Scope Summary")
            st.caption(st.session_state.scope_summary or "(empty)")
        else:
            st.caption("Manual mode: drag and edit nodes directly in the canvas.")
    else:
        mermaid_templates = list_mermaid_templates(diagram_type)
        if mermaid_templates:
            ids = [tpl["id"] for tpl in mermaid_templates]
            lookup = {tpl["id"]: tpl for tpl in mermaid_templates}
            key_name = f"selected_{diagram_type.lower()}_template_id"
            if key_name not in st.session_state:
                st.session_state[key_name] = ids[0]
            selected_mermaid_id = st.selectbox(
                "Common Template",
                ids,
                format_func=lambda x: lookup[x]["name"],
                key=key_name,
            )
            st.caption(lookup[selected_mermaid_id]["description"])
            if st.button("Load Mermaid Template", use_container_width=True):
                set_mermaid_code(diagram_type, get_mermaid_template(diagram_type, selected_mermaid_id))
        else:
            st.info("No predefined templates for this diagram type yet.")

        if selected_mode == "Orchestration":
            agent_state = st.session_state.mermaid_agent_state_by_type.get(
                diagram_type, {"phase": "initial", "source": "fallback", "message": ""}
            )
            st.markdown("### Agent Status")
            st.write(f"`phase`: {agent_state.get('phase', 'initial')}")
            st.write(f"`source`: {agent_state.get('source', 'fallback')}")
            st.caption(agent_state.get("message", ""))
        else:
            st.caption("Manual mode: edit Mermaid code directly.")

if st.session_state.diagram_type == "Flowchart":
    if selected_mode == "Orchestration":
        prompt = st.chat_input("初回はスコープ定義、2回目以降は変更指示を入力")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            run_agent_turn(prompt)

        st.subheader("Copilot Chat")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        st.subheader("Manual Flowchart")
        st.caption("Use template + direct drag editing without agent updates.")

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
        if selected_mode == "Orchestration":
            st.markdown("### Impact Summary")
            render_impact_summary(st.session_state.impact)
            st.caption(st.session_state.impact.get("message", ""))
            st.markdown("### Impact Range")
            st.json(st.session_state.impact, expanded=False)
        st.markdown("### Debug")
        st.json({"node_count": len(curr_state.nodes), "edge_count": len(curr_state.edges)}, expanded=False)

    if selected_mode == "Orchestration":
        st.caption("Set OPENAI_API_KEY to enable real LLM orchestration.")
else:
    diagram_type = st.session_state.diagram_type
    if selected_mode == "Orchestration":
        prompt = st.chat_input(f"{diagram_type} の変更指示を入力")
        if prompt:
            st.session_state.mermaid_chat_history_by_type[diagram_type].append(
                {"role": "user", "content": prompt}
            )
            run_mermaid_agent_turn(diagram_type, prompt)

        st.subheader(f"{diagram_type} Copilot Chat")
        for message in st.session_state.mermaid_chat_history_by_type[diagram_type]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        code = get_mermaid_code(diagram_type)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Mermaid Preview")
            render_mermaid_preview(code, height=540)
        with col2:
            agent_state = st.session_state.mermaid_agent_state_by_type.get(
                diagram_type, {"phase": "initial", "source": "fallback", "message": ""}
            )
            st.markdown("### Mermaid Code")
            st.code(code, language="mermaid")
            st.markdown("### Change Summary")
            st.caption(agent_state.get("message", ""))
            st.markdown("### Source")
            st.caption(agent_state.get("source", "fallback"))
        st.caption("Set OPENAI_API_KEY to enable real LLM orchestration.")
    else:
        st.subheader(f"{diagram_type} Mermaid Editor")
        edited_code = st.text_area(
            "Mermaid Code",
            key=mermaid_editor_key(diagram_type),
            height=520,
        )
        set_mermaid_code(diagram_type, edited_code)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Mermaid Preview")
            render_mermaid_preview(edited_code, height=540)
        with col2:
            st.markdown("### Mermaid Code")
            st.code(edited_code, language="mermaid")
            st.markdown("### Notes")
            st.caption("Switch to Orchestration mode to update this diagram through chat.")
