"""Evidence graph construction and rendering.

This is a VISUALISATION of the adjudication trace, not GraphRAG. Nothing is
retrieved from the graph; it is built after the fact so a human can audit why
one source beat another.

Node types: query, claim, conflict, adjudication, answer
Edge types: SUPPORT, CONTRADICTION, DIFFERENT_SCOPE, DERIVED_FROM, SELECTED,
REJECTED. Relation names intentionally match the conflict detector exactly.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

NODE_COLOURS = {
    "query": "#4C78A8",
    "claim": "#72B7B2",
    "claim_selected": "#54A24B",
    "claim_rejected": "#E45756",
    "conflict": "#F58518",
    "adjudication": "#B279A2",
    "answer": "#54A24B",
}

EDGE_COLOURS = {
    "SUPPORT": "#54A24B",
    "CONTRADICTION": "#E45756",
    "DIFFERENT_SCOPE": "#F58518",
    "DERIVED_FROM": "#9C9C9C",
    "SELECTED": "#54A24B",
    "REJECTED": "#E45756",
}


def build_evidence_graph(
    question: str,
    evidence: list,
    conflicts: list,
    adjudication: Any | None,
) -> nx.DiGraph:
    """Assemble the adjudication trace as a directed graph."""
    graph = nx.DiGraph()

    q_label = question if len(question) <= 60 else question[:57] + "..."
    graph.add_node("query", kind="query", label=q_label)

    selected = getattr(adjudication, "selected_claim_id", None)
    rejected = set(getattr(adjudication, "rejected_claim_ids", []) or [])

    for ev in evidence:
        if not ev.supports_question:
            continue

        node = f"claim_{ev.claim_id}"
        if ev.claim_id == selected:
            kind = "claim_selected"
        elif ev.claim_id in rejected:
            kind = "claim_rejected"
        else:
            kind = "claim"

        graph.add_node(
            node,
            kind=kind,
            label=f"[{ev.claim_id}] {ev.source}\n{ev.value or 'no value'}",
            source=ev.source,
            source_type=ev.source_type,
            date=ev.date,
            value=ev.value,
            scope=ev.scope,
        )
        graph.add_edge("query", node, kind="DERIVED_FROM", label="")

    # Conflict and support edges between claims.
    for rel in conflicts:
        if rel.relationship == "IRRELEVANT":
            continue
        a, b = rel.claim_ids
        node_a, node_b = f"claim_{a}", f"claim_{b}"
        if node_a not in graph or node_b not in graph:
            continue
        graph.add_edge(
            node_a,
            node_b,
            kind=rel.relationship,
            label=rel.relationship.replace("_", " ").title(),
        )

    if adjudication is not None and getattr(adjudication, "criterion", None):
        graph.add_node(
            "adjudication",
            kind="adjudication",
            label=f"ADJUDICATOR\n{adjudication.criterion.replace('_', ' ')}",
        )
        for ev in evidence:
            if ev.claim_id == selected or ev.claim_id in rejected:
                node = f"claim_{ev.claim_id}"
                if node in graph:
                    graph.add_edge(node, "adjudication", kind="DERIVED_FROM")

        if selected is not None:
            graph.add_node("answer", kind="answer", label="FINAL ANSWER")
            graph.add_edge(
                "adjudication", "answer", kind="SELECTED", label="selected"
            )

    return graph


def graph_summary(graph: nx.DiGraph) -> dict[str, Any]:
    """Counts used by the UI and by tests."""
    edges = [d.get("kind", "") for _, _, d in graph.edges(data=True)]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "contradictions": edges.count("CONTRADICTION"),
        "supports": edges.count("SUPPORT"),
        "different_scope": edges.count("DIFFERENT_SCOPE"),
    }


def render_matplotlib(graph: nx.DiGraph, figsize=(12, 7)):
    """Render the audit graph with a stable, role-based layered layout.

    Conflict/support edges form a dense subgraph among claim nodes. Letting a
    force/Graphviz layout optimise those edges can push the query away from the
    top-to-bottom pipeline and make the trace harder to read. Layout therefore
    follows semantic roles first: query -> claims -> adjudicator -> answer.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)

    layer = {
        "query": 0,
        "claim": 1,
        "claim_selected": 1,
        "claim_rejected": 1,
        "conflict": 1,
        "adjudication": 2,
        "answer": 3,
    }
    rows: dict[int, list] = {}
    for node in graph.nodes:
        depth = layer.get(graph.nodes[node].get("kind", "claim"), 1)
        rows.setdefault(depth, []).append(node)

    pos = {}
    y_by_depth = {0: 0.0, 1: -2.2, 2: -4.6, 3: -6.7}
    for depth, nodes in rows.items():
        ordered = sorted(
            nodes,
            key=lambda n: (
                0 if n.startswith("claim_") else 1,
                int(n.split("_")[-1]) if n.startswith("claim_") else 0,
                n,
            ),
        )
        span = len(ordered)
        spacing = 3.0 if span <= 5 else 2.5
        for i, node in enumerate(ordered):
            x = (i - (span - 1) / 2) * spacing
            pos[node] = (x, y_by_depth.get(depth, -depth * 2.2))

    colours = [
        NODE_COLOURS.get(graph.nodes[n].get("kind", "claim"), "#999999")
        for n in graph.nodes
    ]
    nx.draw_networkx_nodes(
        graph, pos, node_color=colours, node_size=3000, alpha=0.92, ax=ax
    )

    relation_kinds = {"SUPPORT", "CONTRADICTION", "DIFFERENT_SCOPE"}

    # Draw structural/provenance edges first.
    for kind in {
        d.get("kind", "") for _, _, d in graph.edges(data=True)
    } - relation_kinds:
        subset = [
            (u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == kind
        ]
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=subset,
            edge_color=EDGE_COLOURS.get(kind, "#9C9C9C"),
            width=1.3,
            style="dashed" if kind == "DERIVED_FROM" else "solid",
            arrowsize=15,
            alpha=0.8,
            ax=ax,
        )

    # Claim-to-claim relations are curved so the complete pairwise comparison
    # does not collapse into one horizontal line through the claim nodes.
    claim_order = {
        node: i
        for i, node in enumerate(
            sorted(n for n in graph.nodes if n.startswith("claim_"))
        )
    }
    for u, v, data in graph.edges(data=True):
        kind = data.get("kind", "")
        if kind not in relation_kinds:
            continue
        distance = abs(claim_order.get(u, 0) - claim_order.get(v, 0))
        direction = 1 if (claim_order.get(u, 0) + claim_order.get(v, 0)) % 2 == 0 else -1
        radius = direction * (0.10 + 0.035 * max(0, distance - 1))
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(u, v)],
            edge_color=EDGE_COLOURS.get(kind, "#9C9C9C"),
            width=2.2 if kind == "CONTRADICTION" else 1.6,
            arrowsize=12,
            alpha=0.85,
            connectionstyle=f"arc3,rad={radius}",
            ax=ax,
        )

    labels = {n: graph.nodes[n].get("label", n) for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels, font_size=7, ax=ax)

    # Structural labels remain useful; relation meaning is clearer as a legend
    # than as ten overlapping labels on a five-claim complete graph.
    edge_labels = {
        (u, v): d.get("label", "")
        for u, v, d in graph.edges(data=True)
        if d.get("label") and d.get("kind") not in relation_kinds
    }
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels, font_size=6, ax=ax
    )

    legend_items = [
        Line2D([0], [0], color=EDGE_COLOURS["CONTRADICTION"], lw=2, label="Contradiction"),
        Line2D([0], [0], color=EDGE_COLOURS["SUPPORT"], lw=2, label="Support"),
        Line2D([0], [0], color=EDGE_COLOURS["DIFFERENT_SCOPE"], lw=2, label="Different scope"),
        Line2D(
            [0], [0], color=EDGE_COLOURS["DERIVED_FROM"], lw=1.5, ls="--",
            label="Evidence flow",
        ),
    ]
    ax.legend(handles=legend_items, loc="lower right", frameon=False, fontsize=7)

    # Node coordinates do not account for label width; leave horizontal room
    # so long source titles at the outer claims are not clipped in saved PNGs.
    max_x = max((abs(x) for x, _ in pos.values()), default=1.0)
    ax.set_xlim(-max_x - 2.4, max_x + 2.4)
    ax.set_ylim(-7.7, 0.9)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def to_dot(graph: nx.DiGraph) -> str:
    """Export as Graphviz DOT for the README or a report."""
    lines = ["digraph EvidenceGraph {", "  rankdir=TB;", "  node [shape=box];"]
    for node, data in graph.nodes(data=True):
        label = str(data.get("label", node)).replace('"', "'").replace("\n", "\\n")
        colour = NODE_COLOURS.get(data.get("kind", "claim"), "#999999")
        lines.append(
            f'  "{node}" [label="{label}", style=filled, fillcolor="{colour}"];'
        )
    for u, v, data in graph.edges(data=True):
        kind = data.get("kind", "")
        colour = EDGE_COLOURS.get(kind, "#9C9C9C")
        lines.append(f'  "{u}" -> "{v}" [label="{kind}", color="{colour}"];')
    lines.append("}")
    return "\n".join(lines)
