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
from src.mermaid_generator.canvas_graph import (  # noqa: E402
    graph_to_mermaid,
    parse_mermaid_to_graph,
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
from src.mermaid_generator.editor_policy import (  # noqa: E402
    get_editor_capabilities,
    get_export_filename,
)
from src.mermaid_generator.ui_mapper import (  # noqa: E402
    flow_items_to_graph_data,
    to_flow_edge_specs,
    to_flow_node_specs,
)
from src.mermaid_generator.property_editor import (  # noqa: E402
    apply_edge_properties,
    apply_node_properties,
    parse_edge_properties,
    parse_node_properties,
    upsert_class_subclass_relation,
)
from src.mermaid_generator.diagram_management import (  # noqa: E402
    ALLOWED_STATUS_TRANSITIONS,
    DiagramRepository,
)
from src.mermaid_generator.diagram_management_assistant import (  # noqa: E402
    DiagramDecisionAssistant,
)


@st.cache_resource
def get_orchestrator() -> FlowchartOrchestrator:
    return FlowchartOrchestrator()


@st.cache_resource
def get_mermaid_orchestrator() -> MermaidDiagramOrchestrator:
    return MermaidDiagramOrchestrator()


@st.cache_resource
def get_diagram_repository() -> DiagramRepository:
    return DiagramRepository(ROOT_DIR / ".diagram_data")


@st.cache_resource
def get_diagram_assistant() -> DiagramDecisionAssistant:
    return DiagramDecisionAssistant()


def to_flow_state(nodes_data: list, edges_data: list, positions: dict) -> StreamlitFlowState:
    node_specs = to_flow_node_specs(nodes_data, positions, edges_data)
    edge_specs = to_flow_edge_specs(edges_data, positions)
    flow_nodes = [StreamlitFlowNode(**spec) for spec in node_specs]
    flow_edges = [StreamlitFlowEdge(**spec) for spec in edge_specs]
    return StreamlitFlowState(nodes=flow_nodes, edges=flow_edges)


def coerce_flow_state(state_obj: object) -> StreamlitFlowState:
    if isinstance(state_obj, StreamlitFlowState):
        return state_obj

    if isinstance(state_obj, dict):
        raw_nodes = state_obj.get("nodes", [])
        raw_edges = state_obj.get("edges", [])
    else:
        raw_nodes = getattr(state_obj, "nodes", [])
        raw_edges = getattr(state_obj, "edges", [])

    nodes = []
    for raw in raw_nodes or []:
        if isinstance(raw, StreamlitFlowNode):
            nodes.append(raw)
            continue
        if isinstance(raw, dict):
            try:
                nodes.append(StreamlitFlowNode.from_dict(raw))
                continue
            except Exception:
                node_id = str(raw.get("id", "")).strip() or f"node_{len(nodes)+1}"
                node_type = str(raw.get("node_type", raw.get("type", "default")) or "default")
                pos = raw.get("pos", raw.get("position", (0.0, 0.0)))
                if isinstance(pos, dict):
                    x = float(pos.get("x", 0.0))
                    y = float(pos.get("y", 0.0))
                else:
                    x = float(pos[0]) if len(pos) > 0 else 0.0
                    y = float(pos[1]) if len(pos) > 1 else 0.0
                data = raw.get("data", {"content": raw.get("label", node_id)})
                nodes.append(
                    StreamlitFlowNode(
                        id=node_id,
                        pos=(x, y),
                        data=data if isinstance(data, dict) else {"content": str(data)},
                        node_type=node_type if node_type in {"input", "default", "output"} else "default",
                    )
                )

    edges = []
    for raw in raw_edges or []:
        if isinstance(raw, StreamlitFlowEdge):
            edges.append(raw)
            continue
        if isinstance(raw, dict):
            try:
                edges.append(StreamlitFlowEdge.from_dict(raw))
                continue
            except Exception:
                edge_id = str(raw.get("id", "")).strip() or f"e{len(edges)+1}"
                source = str(raw.get("source", "")).strip()
                target = str(raw.get("target", "")).strip()
                if source and target:
                    edges.append(
                        StreamlitFlowEdge(
                            id=edge_id,
                            source=source,
                            target=target,
                            label=str(raw.get("label", "") or ""),
                        )
                    )

    return StreamlitFlowState(nodes=nodes, edges=edges)


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
    if "canvas_graph_by_type" not in st.session_state:
        st.session_state.canvas_graph_by_type = {}
    if "edit_mode_by_type" not in st.session_state:
        st.session_state.edit_mode_by_type = {}
    if "default_template_loaded" not in st.session_state:
        st.session_state.default_template_loaded = False

    st.session_state.flow_state = coerce_flow_state(st.session_state.flow_state)

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
            if diagram_type not in st.session_state.canvas_graph_by_type:
                st.session_state.canvas_graph_by_type[diagram_type] = parse_mermaid_to_graph(
                    diagram_type, st.session_state.mermaid_code_by_type[diagram_type]
                )
        if diagram_type not in st.session_state.edit_mode_by_type:
            st.session_state.edit_mode_by_type[diagram_type] = (
                "Orchestration" if diagram_type == "Flowchart" else "Manual"
            )


def get_mermaid_code(diagram_type: str) -> str:
    return str(st.session_state.mermaid_code_by_type.get(diagram_type, ""))


def set_mermaid_code(diagram_type: str, code: str, sync_editor: bool = False) -> None:
    st.session_state.mermaid_code_by_type[diagram_type] = code
    _ = sync_editor  # reserved for compatibility


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


def get_canvas_graph(diagram_type: str) -> dict:
    return st.session_state.canvas_graph_by_type.get(diagram_type, {"nodes": [], "edges": []})


def set_canvas_graph(diagram_type: str, graph_data: dict) -> None:
    st.session_state.canvas_graph_by_type[diagram_type] = graph_data


def apply_managed_diagram(diagram: dict) -> None:
    diagram_type = str(diagram.get("diagram_type", "") or "")
    mermaid_code = str(diagram.get("mermaid_code", "") or "")
    graph_data = diagram.get("graph_data")
    if not isinstance(graph_data, dict):
        graph_data = parse_mermaid_to_graph(diagram_type, mermaid_code)

    if diagram_type == "Flowchart":
        apply_graph_data(graph_data, f"Loaded candidate: {diagram.get('title', diagram.get('id', ''))}")
    else:
        set_canvas_graph(diagram_type, graph_data)
        if mermaid_code.strip():
            set_mermaid_code(diagram_type, mermaid_code, sync_editor=True)
        else:
            set_mermaid_code(diagram_type, graph_to_mermaid(diagram_type, graph_data), sync_editor=True)
        st.session_state.mermaid_agent_state_by_type[diagram_type] = {
            "phase": "loaded",
            "source": "stored",
            "message": f"Loaded candidate: {diagram.get('title', diagram.get('id', ''))}",
        }


def render_candidate_manager(diagram_type: str, mermaid_code: str, graph_data: dict) -> None:
    repository = get_diagram_repository()
    st.markdown("### Candidate Management")
    st.caption("Create, review, and archive diagram candidates with decision history.")

    include_archived = st.checkbox(
        "Include archived",
        value=False,
        key=f"candidate_include_archived_{diagram_type.lower()}",
    )
    all_diagrams = repository.list_diagrams(include_archived=include_archived)
    diagrams = [d for d in all_diagrams if str(d.get("diagram_type", "")) == diagram_type]
    option_ids = [d["id"] for d in diagrams]

    def format_candidate(candidate_id: str) -> str:
        candidate = next((item for item in diagrams if item["id"] == candidate_id), None)
        if not candidate:
            return candidate_id
        return f"{candidate['title']} [{candidate['status']}]"

    if option_ids:
        selected_id = st.selectbox(
            "Saved candidates",
            options=option_ids,
            format_func=format_candidate,
            key=f"candidate_select_{diagram_type.lower()}",
            index=0,
        )
    else:
        st.caption("No saved candidates for this diagram type.")
        selected_id = ""

    default_title = f"{diagram_type} Candidate"
    title = st.text_input(
        "New candidate title",
        value=default_title,
        key=f"candidate_title_{diagram_type.lower()}",
    )
    if st.button("Save New Candidate", key=f"candidate_save_new_{diagram_type.lower()}"):
        created = repository.create_diagram(
            title=title,
            diagram_type=diagram_type,
            mermaid_code=mermaid_code,
            graph_data=graph_data,
            actor="user",
            tags=[diagram_type.lower()],
        )
        st.success(f"Saved: {created['title']}")
        st.rerun()

    if not selected_id:
        return

    selected_diagram = repository.get_diagram(selected_id)
    if not selected_diagram:
        st.warning("Selected candidate was not found.")
        return

    row_load, row_update = st.columns(2)
    if row_load.button("Load Candidate", key=f"candidate_load_{diagram_type.lower()}"):
        apply_managed_diagram(selected_diagram)
        st.success(f"Loaded: {selected_diagram['title']}")
        st.rerun()
    if row_update.button("Save Update", key=f"candidate_save_update_{diagram_type.lower()}"):
        repository.update_diagram_content(
            diagram_id=selected_id,
            mermaid_code=mermaid_code,
            graph_data=graph_data,
            actor="user",
        )
        st.success("Candidate updated.")
        st.rerun()

    current_status = str(selected_diagram.get("status", "active"))
    next_statuses = sorted(ALLOWED_STATUS_TRANSITIONS.get(current_status, set()))
    if next_statuses:
        next_status = st.selectbox(
            "Transition status",
            options=next_statuses,
            key=f"candidate_next_status_{diagram_type.lower()}",
        )
        status_reason = st.text_input(
            "Status reason",
            value="",
            key=f"candidate_status_reason_{diagram_type.lower()}",
        )
        if st.button("Apply Status", key=f"candidate_apply_status_{diagram_type.lower()}"):
            repository.set_status(
                diagram_id=selected_id,
                status=next_status,
                actor="user",
                reason=status_reason,
            )
            st.success(f"Status changed to {next_status}.")
            st.rerun()

    st.markdown("#### LLM Decision Note")
    llm_prompt = st.text_area(
        "Decision prompt",
        value="Summarize rationale, impact range, and next action.",
        height=90,
        key=f"candidate_llm_prompt_{diagram_type.lower()}",
    )
    if st.button("Auto-Log Note (LLM/Fallback)", key=f"candidate_auto_log_{diagram_type.lower()}"):
        assistant = get_diagram_assistant()
        recent_events = repository.list_decision_events(selected_id)
        draft = assistant.draft_note(
            diagram=selected_diagram,
            user_request=llm_prompt,
            recent_events=recent_events,
        )
        repository.append_decision_event(
            diagram_id=selected_id,
            actor="assistant" if draft.get("source") == "llm" else "fallback",
            stage="llm_note",
            summary=str(draft.get("summary", "Decision note")),
            markdown_comment=str(draft.get("markdown_comment", "")),
            tags=list(draft.get("tags", [])),
        )
        if draft.get("source") == "llm":
            st.success("Decision note logged from LLM output.")
        else:
            st.info("LLM unavailable. Markdown fallback note logged.")
        st.rerun()

    note_summary = st.text_input(
        "Decision summary",
        value="Review note",
        key=f"candidate_note_summary_{diagram_type.lower()}",
    )
    note_markdown = st.text_area(
        "Decision markdown",
        value="",
        height=120,
        key=f"candidate_note_markdown_{diagram_type.lower()}",
    )
    if st.button("Add Decision Note", key=f"candidate_add_note_{diagram_type.lower()}"):
        if not note_markdown.strip():
            st.warning("Decision markdown is required.")
        else:
            repository.append_decision_event(
                diagram_id=selected_id,
                actor="user",
                stage="review",
                summary=note_summary,
                markdown_comment=note_markdown,
                tags=["review", diagram_type.lower()],
            )
            st.success("Decision note recorded.")
            st.rerun()

    events = repository.list_decision_events(selected_id)
    vectors = repository.list_vector_records(selected_id)
    st.caption(f"Decision events: {len(events)} / Vector records: {len(vectors)}")
    for event in reversed(events[-5:]):
        st.markdown(f"**{event.get('stage', '').upper()}** {event.get('summary', '')}")
        st.caption(event.get("created_at", ""))
        comment = str(event.get("markdown_comment", "")).strip()
        if comment:
            st.markdown(comment)


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
    set_mermaid_code(diagram_type, turn.mermaid_code, sync_editor=True)
    set_canvas_graph(diagram_type, parse_mermaid_to_graph(diagram_type, turn.mermaid_code))
    st.session_state.mermaid_agent_state_by_type[diagram_type] = {
        "phase": turn.phase,
        "source": turn.source,
        "message": turn.change_summary,
    }


def render_property_panel(diagram_type: str, graph_data: dict) -> dict:
    st.markdown("### Properties")
    if diagram_type not in {"Class", "ER", "State", "Sequence"}:
        st.caption("Type-specific property panel will be expanded in a follow-up issue.")
        return graph_data

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    if not nodes:
        st.caption("No nodes available.")
        return graph_data

    node_options = [node["id"] for node in nodes]
    selected_node_id = st.selectbox(
        "Node",
        node_options,
        key=f"prop_node_{diagram_type.lower()}",
    )
    selected_node = next(node for node in nodes if node["id"] == selected_node_id)
    node_props = parse_node_properties(diagram_type, selected_node["id"], selected_node.get("label", ""))
    updated_graph = graph_data

    if diagram_type == "Class":
        node_name = st.text_input(
            "Class Name", value=node_props.get("name", ""), key=f"class_name_{diagram_type}_{selected_node_id}"
        )
        node_attrs = st.text_input(
            "Attributes",
            value=node_props.get("attributes", ""),
            key=f"class_attrs_{diagram_type}_{selected_node_id}",
        )
        node_methods = st.text_input(
            "Methods",
            value=node_props.get("methods", ""),
            key=f"class_methods_{diagram_type}_{selected_node_id}",
        )
        if st.button("Apply Node Properties", key=f"apply_node_props_{diagram_type}_{selected_node_id}"):
            updated_graph = apply_node_properties(
                diagram_type,
                updated_graph,
                selected_node_id,
                {"name": node_name, "attributes": node_attrs, "methods": node_methods},
            )

        st.markdown("#### Subclass Relation")
        child_id = st.selectbox("Child Class", node_options, key=f"sub_child_{diagram_type}")
        parent_candidates = [node_id for node_id in node_options if node_id != child_id]
        if parent_candidates:
            parent_id = st.selectbox("Parent Class", parent_candidates, key=f"sub_parent_{diagram_type}")
            if st.button("Apply Subclass", key=f"apply_subclass_{diagram_type}"):
                updated_graph = upsert_class_subclass_relation(updated_graph, child_id, parent_id)
        else:
            st.caption("Add at least two classes to define subclass relation.")

    if diagram_type == "ER":
        er_name = st.text_input(
            "Entity Name", value=node_props.get("name", ""), key=f"er_name_{diagram_type}_{selected_node_id}"
        )
        er_pk = st.text_input(
            "Primary Key", value=node_props.get("primary_key", ""), key=f"er_pk_{diagram_type}_{selected_node_id}"
        )
        er_fk = st.text_input(
            "Foreign Key", value=node_props.get("foreign_key", ""), key=f"er_fk_{diagram_type}_{selected_node_id}"
        )
        if st.button("Apply Node Properties", key=f"apply_node_props_{diagram_type}_{selected_node_id}"):
            updated_graph = apply_node_properties(
                diagram_type,
                updated_graph,
                selected_node_id,
                {"name": er_name, "primary_key": er_pk, "foreign_key": er_fk},
            )

    if diagram_type == "State":
        state_name = st.text_input(
            "State Name", value=node_props.get("name", ""), key=f"state_name_{diagram_type}_{selected_node_id}"
        )
        if st.button("Apply Node Properties", key=f"apply_node_props_{diagram_type}_{selected_node_id}"):
            updated_graph = apply_node_properties(
                diagram_type, updated_graph, selected_node_id, {"name": state_name}
            )

    if diagram_type == "Sequence":
        alias = st.text_input(
            "Participant Alias", value=node_props.get("alias", ""), key=f"seq_alias_{diagram_type}_{selected_node_id}"
        )
        if st.button("Apply Node Properties", key=f"apply_node_props_{diagram_type}_{selected_node_id}"):
            updated_graph = apply_node_properties(
                diagram_type, updated_graph, selected_node_id, {"alias": alias}
            )

    if edges:
        st.markdown("#### Edge Properties")
        edge_options = [edge["id"] for edge in edges]
        selected_edge_id = st.selectbox(
            "Edge",
            edge_options,
            key=f"prop_edge_{diagram_type.lower()}",
        )
        selected_edge = next(edge for edge in edges if edge["id"] == selected_edge_id)
        edge_props = parse_edge_properties(diagram_type, selected_edge.get("label", ""))

        if diagram_type == "Class":
            relation_type = st.selectbox(
                "Relation Type",
                ["association", "extends", "implements", "composition", "aggregation"],
                index=["association", "extends", "implements", "composition", "aggregation"].index(
                    edge_props.get("relation_type", "association")
                )
                if edge_props.get("relation_type", "association")
                in {"association", "extends", "implements", "composition", "aggregation"}
                else 0,
                key=f"class_relation_type_{diagram_type}_{selected_edge_id}",
            )
            relation_text = st.text_input(
                "Relation Label",
                value=edge_props.get("relation", ""),
                key=f"class_relation_text_{diagram_type}_{selected_edge_id}",
            )
            if st.button("Apply Edge Properties", key=f"apply_edge_props_{diagram_type}_{selected_edge_id}"):
                updated_graph = apply_edge_properties(
                    diagram_type,
                    updated_graph,
                    selected_edge_id,
                    {"relation_type": relation_type, "relation": relation_text},
                )

        if diagram_type == "ER":
            card_options = ["1-1", "1-N", "N-1", "N-N"]
            cardinality = st.selectbox(
                "Cardinality",
                card_options,
                index=card_options.index(edge_props.get("cardinality"))
                if edge_props.get("cardinality") in card_options
                else 0,
                key=f"er_cardinality_{diagram_type}_{selected_edge_id}",
            )
            relation = st.text_input(
                "Relation",
                value=edge_props.get("relation", ""),
                key=f"er_relation_{diagram_type}_{selected_edge_id}",
            )
            if st.button("Apply Edge Properties", key=f"apply_edge_props_{diagram_type}_{selected_edge_id}"):
                updated_graph = apply_edge_properties(
                    diagram_type,
                    updated_graph,
                    selected_edge_id,
                    {"cardinality": cardinality, "relation": relation},
                )

        if diagram_type == "State":
            event = st.text_input(
                "Event",
                value=edge_props.get("event", ""),
                key=f"state_event_{diagram_type}_{selected_edge_id}",
            )
            guard = st.text_input(
                "Guard",
                value=edge_props.get("guard", ""),
                key=f"state_guard_{diagram_type}_{selected_edge_id}",
            )
            if st.button("Apply Edge Properties", key=f"apply_edge_props_{diagram_type}_{selected_edge_id}"):
                updated_graph = apply_edge_properties(
                    diagram_type, updated_graph, selected_edge_id, {"event": event, "guard": guard}
                )

        if diagram_type == "Sequence":
            msg_options = ["sync", "async", "return"]
            message_type = st.selectbox(
                "Message Type",
                msg_options,
                index=msg_options.index(edge_props.get("message_type"))
                if edge_props.get("message_type") in msg_options
                else 0,
                key=f"seq_msg_type_{diagram_type}_{selected_edge_id}",
            )
            message = st.text_input(
                "Message",
                value=edge_props.get("message", ""),
                key=f"seq_msg_{diagram_type}_{selected_edge_id}",
            )
            if st.button("Apply Edge Properties", key=f"apply_edge_props_{diagram_type}_{selected_edge_id}"):
                updated_graph = apply_edge_properties(
                    diagram_type,
                    updated_graph,
                    selected_edge_id,
                    {"message_type": message_type, "message": message},
                )

    return updated_graph


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
                template_code = get_mermaid_template(diagram_type, selected_mermaid_id)
                set_mermaid_code(
                    diagram_type,
                    template_code,
                    sync_editor=True,
                )
                set_canvas_graph(diagram_type, parse_mermaid_to_graph(diagram_type, template_code))
        else:
            st.info("No predefined templates for this diagram type yet.")

        with st.expander("Import Mermaid", expanded=False):
            import_key = f"import_mermaid_{diagram_type.lower()}"
            if import_key not in st.session_state:
                st.session_state[import_key] = get_mermaid_code(diagram_type)
            import_code = st.text_area(
                "Paste Mermaid Code",
                key=import_key,
                height=180,
            )
            if st.button("Load From Mermaid", key=f"load_mermaid_{diagram_type.lower()}"):
                set_mermaid_code(diagram_type, import_code, sync_editor=True)
                set_canvas_graph(diagram_type, parse_mermaid_to_graph(diagram_type, import_code))

        if selected_mode == "Orchestration":
            agent_state = st.session_state.mermaid_agent_state_by_type.get(
                diagram_type, {"phase": "initial", "source": "fallback", "message": ""}
            )
            st.markdown("### Agent Status")
            st.write(f"`phase`: {agent_state.get('phase', 'initial')}")
            st.write(f"`source`: {agent_state.get('source', 'fallback')}")
            st.caption(agent_state.get("message", ""))
        else:
            st.caption("Manual mode: drag and edit nodes directly in the canvas.")

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
        coerce_flow_state(st.session_state.flow_state),
        fit_view=True,
        height=520,
    )

    current_nodes, current_edges = flow_items_to_graph_data(curr_state.nodes, curr_state.edges)
    mermaid_text = export_to_mermaid(current_nodes, current_edges)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Flowchart Preview")
        render_mermaid_preview(mermaid_text, height=420)
        st.markdown("### Mermaid Export")
        st.code(mermaid_text, language="mermaid")
        st.download_button(
            "Export Mermaid (.mmd)",
            data=mermaid_text,
            file_name=get_export_filename("Flowchart"),
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        if selected_mode == "Orchestration":
            st.markdown("### Impact Summary")
            render_impact_summary(st.session_state.impact)
            st.caption(st.session_state.impact.get("message", ""))
            st.markdown("### Impact Range")
            st.json(st.session_state.impact, expanded=False)
        st.markdown("### Debug")
        st.json({"node_count": len(curr_state.nodes), "edge_count": len(curr_state.edges)}, expanded=False)
        render_candidate_manager(
            "Flowchart",
            mermaid_text,
            {"nodes": current_nodes, "edges": current_edges},
        )

    if selected_mode == "Orchestration":
        st.caption("Set OPENAI_API_KEY to enable real LLM orchestration.")
else:
    diagram_type = st.session_state.diagram_type
    capabilities = get_editor_capabilities(diagram_type, selected_mode)

    if capabilities["chat"]:
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

    st.subheader(f"{diagram_type} Editor")
    st.caption("Shared editor surface: edit, preview, export.")
    graph_data = get_canvas_graph(diagram_type)
    positions = calculate_layout_positions(graph_data["nodes"], graph_data["edges"])
    flow_state = to_flow_state(graph_data["nodes"], graph_data["edges"], positions)
    curr_state = streamlit_flow(
        f"flow_{diagram_type.lower()}",
        flow_state,
        fit_view=True,
        height=520,
    )
    current_nodes, current_edges = flow_items_to_graph_data(curr_state.nodes, curr_state.edges)
    current_graph = {"nodes": current_nodes, "edges": current_edges}
    set_canvas_graph(diagram_type, current_graph)
    code = graph_to_mermaid(diagram_type, current_graph)
    set_mermaid_code(diagram_type, code)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Mermaid Preview")
        render_mermaid_preview(code, height=540)
    with col2:
        st.markdown("### Mermaid Export")
        st.code(code, language="mermaid")
        st.download_button(
            "Export Mermaid (.mmd)",
            data=code,
            file_name=get_export_filename(diagram_type),
            mime="text/plain",
            use_container_width=True,
        )
        updated_graph = render_property_panel(diagram_type, current_graph)
        if updated_graph != current_graph:
            set_canvas_graph(diagram_type, updated_graph)
            updated_code = graph_to_mermaid(diagram_type, updated_graph)
            set_mermaid_code(diagram_type, updated_code)
            st.rerun()
        render_candidate_manager(diagram_type, code, current_graph)
        if capabilities["chat"]:
            agent_state = st.session_state.mermaid_agent_state_by_type.get(
                diagram_type, {"phase": "initial", "source": "fallback", "message": ""}
            )
            st.markdown("### Change Summary")
            st.caption(agent_state.get("message", ""))
            st.markdown("### Source")
            st.caption(agent_state.get("source", "fallback"))
            st.caption("Set OPENAI_API_KEY to enable real LLM orchestration.")
