# Mermaid Generator (LLM + Python Layout + Streamlit-Flow)

This repository provides a Python-only flowchart generator with a GUI editor. The core idea is:

1. LLM generates logical structure only (nodes/edges as JSON, no coordinates).
2. Python computes node coordinates using a layout engine (NetworkX).
3. Streamlit-Flow renders the graph and allows manual adjustments.
4. A lightweight orchestrator changes behavior by turn:
   - first turn: scope clarification + initial diagram generation
   - second turn onward: graph update + impact range detection
5. Common Mermaid templates are selectable by default:
   - Flowchart templates (GUI editable)
   - Sequence/State/ER/Class/Gantt templates (code mode)
6. Editor UX parity across all diagram types:
   - shared editor surface
   - edit, preview, export support
   - optional orchestration chat mode

The app lives under `app/` and business logic lives under `src/`.
The `testbed/` directory is reserved for tests to support TDD.

## Quickstart (Conda)

Create and activate environment:

```bash
conda env create -f environment.yml
conda activate mermaid-generator
```

Run the app:

```bash
streamlit run app/app.py
```

Enable real LLM mode (optional):

```bash
export OPENAI_API_KEY=\"your_api_key\"
# optional: export OPENAI_MODEL=\"gpt-4o-mini\"
```

Run tests:

```bash
pytest -q testbed
```

### Optional (manual install in existing conda env)

```bash
conda install -c conda-forge streamlit networkx
pip install streamlit-flow-component==1.6.1
```

## Architecture

- LLM: returns JSON with `nodes` and `edges`.
- Orchestrator: manages first-turn scope flow and update-turn impact analysis.
- Templates: provides common diagrams as defaults for faster start.
- Layout: uses NetworkX to compute a simple hierarchical layout.
- UI: Streamlit-Flow renders nodes/edges and supports drag-and-drop editing.
- Export: UI state is converted to Mermaid syntax.
- Editor policy: all types support `edit + preview + export` with consistent controls.

## Editor UX Parity

Every diagram type now uses a consistent Editor workflow.

- Types: `Flowchart`, `Sequence`, `State`, `ER`, `Class`, `Gantt`
- Shared controls: load template, edit, preview, export
- Work modes: `Orchestration` or `Manual`
- Type-specific behavior:
  - `Flowchart`: node/edge canvas editing + Mermaid export
  - non-flowchart types: Mermaid text editing + Mermaid render preview + export

## JSON Schema (LLM Output)

```json
{
  "nodes": [
    {"id": "string", "label": "string", "type": "input|default|output"}
  ],
  "edges": [
    {"id": "string", "source": "node_id", "target": "node_id", "label": "string"}
  ]
}
```

## Mermaid Export Example

```mermaid
graph TD;
    start["Start"];
    process["Do work"];
    end["Done"];
    start --> process;
    process --> end;
```

## Project Layout

- `README.md`: project overview and usage
- `ISSUE.md`: issue template
- `environment.yml`: conda environment definition
- `app/app.py`: Streamlit UI entrypoint
- `src/mermaid_generator/graph_logic.py`: business logic
- `src/mermaid_generator/orchestrator.py`: LLM orchestration and impact range detection
- `src/mermaid_generator/templates.py`: common Mermaid template catalog
- `src/mermaid_generator/ui_mapper.py`: UI conversion functions (testable)
- `testbed/test_graph_logic.py`: tests for TDD workflow
- `testbed/test_orchestrator.py`: tests for first-turn/update-turn behavior
- `testbed/test_templates.py`: tests for default template catalog
- `testbed/test_ui_mapper.py`: tests for UI conversion

## Notes

- If you plan to integrate a real LLM, enforce JSON-only output.
- For more stable layouts, consider Graphviz if available.

## License

See `LICENSE`.
