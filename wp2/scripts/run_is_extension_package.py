"""Host-anchor selection used by the fixed-carrier benchmark."""
from __future__ import annotations

import networkx as nx


def select_host_anchors(graph: nx.Graph, count: int = 4) -> tuple[str, ...]:
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    degrees = dict(graph.degree())
    nodes = list(graph.nodes())
    first = max(nodes, key=lambda node: (degrees[node], str(node)))
    chosen = [first]
    while len(chosen) < count:
        remaining = [node for node in nodes if node not in chosen]
        best = max(
            remaining,
            key=lambda node: (
                min(lengths[node][anchor] for anchor in chosen),
                degrees[node],
                str(node),
            ),
        )
        chosen.append(best)
    return tuple(chosen)

