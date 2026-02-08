from copy import deepcopy
from typing import Any, Dict, List

from .graph_logic import GraphData

DIAGRAM_TYPES = ["Flowchart", "Sequence", "State", "ER", "Class", "Gantt"]

FLOWCHART_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "ec_purchase": {
        "id": "ec_purchase",
        "name": "EC Purchase Flow",
        "description": "Typical checkout process from cart to fulfillment.",
        "graph": {
            "nodes": [
                {"id": "start", "label": "Start", "type": "input"},
                {"id": "cart", "label": "Add to Cart", "type": "default"},
                {"id": "checkout", "label": "Checkout", "type": "default"},
                {"id": "payment", "label": "Payment", "type": "default"},
                {"id": "confirm", "label": "Order Confirmed", "type": "output"},
                {"id": "fail", "label": "Payment Failed", "type": "output"},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "cart", "label": ""},
                {"id": "e2", "source": "cart", "target": "checkout", "label": ""},
                {"id": "e3", "source": "checkout", "target": "payment", "label": ""},
                {"id": "e4", "source": "payment", "target": "confirm", "label": "OK"},
                {"id": "e5", "source": "payment", "target": "fail", "label": "NG"},
            ],
        },
    },
    "approval": {
        "id": "approval",
        "name": "Approval Workflow",
        "description": "Request, review, approve/reject process.",
        "graph": {
            "nodes": [
                {"id": "start", "label": "Submit Request", "type": "input"},
                {"id": "review", "label": "Manager Review", "type": "default"},
                {"id": "approve", "label": "Approved", "type": "output"},
                {"id": "reject", "label": "Rejected", "type": "output"},
                {"id": "rework", "label": "Rework", "type": "default"},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "review", "label": ""},
                {"id": "e2", "source": "review", "target": "approve", "label": "Approve"},
                {"id": "e3", "source": "review", "target": "reject", "label": "Reject"},
                {"id": "e4", "source": "reject", "target": "rework", "label": "Fix"},
                {"id": "e5", "source": "rework", "target": "review", "label": "Resubmit"},
            ],
        },
    },
    "incident": {
        "id": "incident",
        "name": "Incident Response",
        "description": "Detect, triage, mitigate, and postmortem.",
        "graph": {
            "nodes": [
                {"id": "detect", "label": "Alert Detected", "type": "input"},
                {"id": "triage", "label": "Triage", "type": "default"},
                {"id": "sev", "label": "Severity Decision", "type": "default"},
                {"id": "mitigate", "label": "Mitigate", "type": "default"},
                {"id": "resolve", "label": "Resolve", "type": "output"},
                {"id": "post", "label": "Postmortem", "type": "output"},
            ],
            "edges": [
                {"id": "e1", "source": "detect", "target": "triage", "label": ""},
                {"id": "e2", "source": "triage", "target": "sev", "label": ""},
                {"id": "e3", "source": "sev", "target": "mitigate", "label": "High"},
                {"id": "e4", "source": "sev", "target": "resolve", "label": "Low"},
                {"id": "e5", "source": "mitigate", "target": "resolve", "label": ""},
                {"id": "e6", "source": "resolve", "target": "post", "label": ""},
            ],
        },
    },
    "support": {
        "id": "support",
        "name": "Customer Support",
        "description": "Ticket intake to closure with escalation path.",
        "graph": {
            "nodes": [
                {"id": "intake", "label": "Ticket Intake", "type": "input"},
                {"id": "classify", "label": "Classify", "type": "default"},
                {"id": "resolve", "label": "L1 Resolve", "type": "default"},
                {"id": "escalate", "label": "Escalate to L2", "type": "default"},
                {"id": "close", "label": "Close Ticket", "type": "output"},
            ],
            "edges": [
                {"id": "e1", "source": "intake", "target": "classify", "label": ""},
                {"id": "e2", "source": "classify", "target": "resolve", "label": "Simple"},
                {"id": "e3", "source": "classify", "target": "escalate", "label": "Complex"},
                {"id": "e4", "source": "resolve", "target": "close", "label": ""},
                {"id": "e5", "source": "escalate", "target": "close", "label": "Resolved"},
            ],
        },
    },
    "onboarding": {
        "id": "onboarding",
        "name": "Employee Onboarding",
        "description": "Hiring to productive state.",
        "graph": {
            "nodes": [
                {"id": "offer", "label": "Offer Accepted", "type": "input"},
                {"id": "setup", "label": "Account Setup", "type": "default"},
                {"id": "equip", "label": "Equipment Provision", "type": "default"},
                {"id": "training", "label": "Orientation & Training", "type": "default"},
                {"id": "productive", "label": "Productive", "type": "output"},
            ],
            "edges": [
                {"id": "e1", "source": "offer", "target": "setup", "label": ""},
                {"id": "e2", "source": "setup", "target": "equip", "label": ""},
                {"id": "e3", "source": "equip", "target": "training", "label": ""},
                {"id": "e4", "source": "training", "target": "productive", "label": ""},
            ],
        },
    },
    "bugfix": {
        "id": "bugfix",
        "name": "Bug Fix Workflow",
        "description": "Issue report to deployment and monitoring.",
        "graph": {
            "nodes": [
                {"id": "report", "label": "Bug Report", "type": "input"},
                {"id": "repro", "label": "Reproduce", "type": "default"},
                {"id": "fix", "label": "Implement Fix", "type": "default"},
                {"id": "test", "label": "Regression Test", "type": "default"},
                {"id": "deploy", "label": "Deploy", "type": "output"},
                {"id": "reopen", "label": "Reopen", "type": "output"},
            ],
            "edges": [
                {"id": "e1", "source": "report", "target": "repro", "label": ""},
                {"id": "e2", "source": "repro", "target": "fix", "label": "Confirmed"},
                {"id": "e3", "source": "fix", "target": "test", "label": ""},
                {"id": "e4", "source": "test", "target": "deploy", "label": "Pass"},
                {"id": "e5", "source": "test", "target": "reopen", "label": "Fail"},
            ],
        },
    },
}

MERMAID_TEMPLATES: Dict[str, List[Dict[str, str]]] = {
    "Sequence": [
        {
            "id": "api_sequence",
            "name": "API Request",
            "description": "Client to API to DB sequence.",
            "code": (
                "sequenceDiagram\n"
                "    participant U as User\n"
                "    participant C as Client\n"
                "    participant A as API\n"
                "    participant D as DB\n"
                "    U->>C: Submit form\n"
                "    C->>A: POST /orders\n"
                "    A->>D: Insert order\n"
                "    D-->>A: OK\n"
                "    A-->>C: 201 Created\n"
                "    C-->>U: Confirmation\n"
            ),
        }
    ],
    "State": [
        {
            "id": "order_state",
            "name": "Order State",
            "description": "Order lifecycle state machine.",
            "code": (
                "stateDiagram-v2\n"
                "    [*] --> Created\n"
                "    Created --> Paid\n"
                "    Paid --> Shipped\n"
                "    Shipped --> Delivered\n"
                "    Paid --> Cancelled\n"
                "    Created --> Cancelled\n"
            ),
        }
    ],
    "ER": [
        {
            "id": "ec_er",
            "name": "EC Minimal ER",
            "description": "Basic user-order-item schema.",
            "code": (
                "erDiagram\n"
                "    USER ||--o{ ORDER : places\n"
                "    ORDER ||--|{ ORDER_ITEM : contains\n"
                "    PRODUCT ||--o{ ORDER_ITEM : referenced_by\n"
                "    USER {\n"
                "        int id PK\n"
                "        string email\n"
                "    }\n"
                "    ORDER {\n"
                "        int id PK\n"
                "        int user_id FK\n"
                "    }\n"
            ),
        }
    ],
    "Class": [
        {
            "id": "service_class",
            "name": "Service Class Diagram",
            "description": "App service and repository structure.",
            "code": (
                "classDiagram\n"
                "    class OrderService {\n"
                "      +createOrder()\n"
                "      +cancelOrder()\n"
                "    }\n"
                "    class OrderRepository {\n"
                "      +save()\n"
                "      +findById()\n"
                "    }\n"
                "    class PaymentGateway {\n"
                "      +charge()\n"
                "    }\n"
                "    OrderService --> OrderRepository\n"
                "    OrderService --> PaymentGateway\n"
            ),
        }
    ],
    "Gantt": [
        {
            "id": "release_gantt",
            "name": "Release Plan",
            "description": "Simple weekly release timeline.",
            "code": (
                "gantt\n"
                "    title Release Plan\n"
                "    dateFormat  YYYY-MM-DD\n"
                "    section Planning\n"
                "    Scope Freeze     :a1, 2026-02-10, 3d\n"
                "    section Build\n"
                "    Implementation   :a2, after a1, 5d\n"
                "    section Validate\n"
                "    QA              :a3, after a2, 3d\n"
                "    Release         :milestone, after a3, 0d\n"
            ),
        }
    ],
}


def list_flowchart_templates() -> List[Dict[str, str]]:
    return [
        {
            "id": template["id"],
            "name": template["name"],
            "description": template["description"],
        }
        for template in FLOWCHART_TEMPLATES.values()
    ]


def get_flowchart_template(template_id: str) -> Dict[str, Any]:
    template = FLOWCHART_TEMPLATES.get(template_id)
    if not template:
        template = FLOWCHART_TEMPLATES["ec_purchase"]
    return {
        "id": template["id"],
        "name": template["name"],
        "description": template["description"],
        "graph": deepcopy(template["graph"]),
    }


def list_mermaid_templates(diagram_type: str) -> List[Dict[str, str]]:
    templates = MERMAID_TEMPLATES.get(diagram_type, [])
    return [
        {"id": t["id"], "name": t["name"], "description": t["description"]}
        for t in templates
    ]


def get_mermaid_template(diagram_type: str, template_id: str) -> str:
    templates = MERMAID_TEMPLATES.get(diagram_type, [])
    for template in templates:
        if template["id"] == template_id:
            return template["code"]
    return ""


def get_flowchart_template_graph(template_id: str) -> GraphData:
    return get_flowchart_template(template_id)["graph"]
