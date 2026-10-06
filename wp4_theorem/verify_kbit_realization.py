#!/usr/bin/env python3
"""
WP4 de-risking: exact finite verification of a k-bit rooted-cell graph realization
of the parity-class carrier construction for k in {3,4,5,6}.

This mirrors the methodology of the existing exact k=3 audits in empirical_artifacts/:
a rooted-cell family is assembled, a *local radius-r ball* matcher recovers a k-bit code
at each root, and we verify the four proof obligations exactly and finitely.

Encoding note (why this is not a trivial copy of the k=3 triadic cell):
the original construction encodes the 3-bit code as the three possible edges among three
arm-endpoints; that works only because C(3,2) = 3 = k at k=3. A genuine k-bit carrier needs
k INDEPENDENT local features. We therefore encode bit i by the presence/absence of a pendant
"flag" at the endpoint of the i-th arm (arms have distinct lengths, so flags sit at distinct
depths and every code yields a distinct rooted ball). At k=3 this reduces to a valid frontier-3
witness, confirming the generalized construction is correct at the known dimension.

Obligations checked (per k, per parity class):
  L1 realizability   : the local matcher recovers, at each planted root, exactly the intended code.
  L2 non-isomorphism : all 2^k rooted-ball templates are pairwise non-isomorphic.
  L3 isolation       : NO non-root node has a radius-r ball isomorphic to any template
                       (deterministic spacer; no rejection sampling).
  L4 frontier        : squarefree monomial counts of the matcher-recovered codes agree through
                       order k-1 and first differ at order k  ->  exact separating order = k.

No value is hand-entered; every reported number is computed from the assembled graph.
"""
from __future__ import annotations
import json, itertools, sys
from pathlib import Path
import networkx as nx
from networkx.algorithms import isomorphism as iso

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------- parity-class carrier algebra ----------
def parity_classes(k):
    even, odd = [], []
    for x in range(2 ** k):
        bits = tuple((x >> i) & 1 for i in range(k))
        (odd if sum(bits) % 2 else even).append(bits)
    return even, odd


def squarefree_monomial_counts(codes, k):
    """For every subset S of [k], count codes with prod_{i in S} x_i = 1, grouped by |S|."""
    per_order = {}
    detail = {}
    for r in range(k + 1):
        for S in itertools.combinations(range(k), r):
            c = sum(all(code[i] for i in S) for code in codes)
            detail[S] = c
            per_order.setdefault(r, []).append((S, c))
    return per_order, detail


def frontier_order(codes_pos, codes_neg, k):
    _, dp = squarefree_monomial_counts(codes_pos, k)
    _, dn = squarefree_monomial_counts(codes_neg, k)
    for r in range(k + 1):
        for S in itertools.combinations(range(k), r):
            if dp[S] != dn[S]:
                return r
    return None


# ---------- graph construction (k-adic rooted cell, flag encoding) ----------
def arm_lengths(k):
    return list(range(2, k + 2))          # k arms of distinct lengths 2..k+1


def radius(k):
    return max(arm_lengths(k)) + 1        # r = k+2 : deep enough to see every flag


def build_cell(code, prefix, k):
    G = nx.Graph()
    root = f"{prefix}r"
    G.add_node(root)
    for i, L in enumerate(arm_lengths(k)):
        prev = root
        for d in range(1, L + 1):
            nd = f"{prefix}a{i}_{d}"
            G.add_edge(prev, nd)
            prev = nd
        if code[i]:                        # bit i -> pendant flag at endpoint of arm i
            G.add_edge(prev, f"{prefix}flag{i}")
    conn_len = radius(k) + 1               # connector fills the ball; spacer stays outside radius
    prev = root
    for d in range(1, conn_len + 1):
        nd = f"{prefix}z{d}"
        G.add_edge(prev, nd)
        prev = nd
    return G, root, prev                    # graph, root, connector-endpoint


def radius_ball(graph, center, r):
    nodes = list(nx.single_source_shortest_path_length(graph, center, cutoff=r))
    ball = graph.subgraph(nodes).copy()
    for n in ball.nodes:
        ball.nodes[n]["is_root"] = (n == center)
    return ball


def root_node_match(a, b):
    return bool(a.get("is_root", False)) == bool(b.get("is_root", False))


def templates(k):
    tpl = {}
    for code in itertools.product((0, 1), repeat=k):
        G, root, _ = build_cell(code, f"tpl_{''.join(map(str,code))}_", k)
        tpl[code] = radius_ball(G, root, radius(k))
    return tpl


def assemble(codes, k, spacer_len):
    """Plant one cell per code along a single deterministic spacer path."""
    G = nx.Graph()
    roots = {}
    connectors = []
    for idx, code in enumerate(codes):
        cell, root, conn = build_cell(code, f"c{idx}_", k)
        G = nx.compose(G, cell)
        roots[root] = tuple(code)
        connectors.append(conn)
    for j in range(len(connectors) - 1):     # deterministic spacer, no regeneration
        prev = connectors[j]
        for s in range(1, spacer_len):
            nd = f"sp{j}_{s}"
            G.add_edge(prev, nd)
            prev = nd
        G.add_edge(prev, connectors[j + 1])
    return G, roots


def match_code(graph, node, tpl, k):
    if graph.degree[node] != k + 1:          # only roots have degree k+1
        return None
    ball = radius_ball(graph, node, radius(k))
    hits = [code for code, t in tpl.items()
            if iso.GraphMatcher(ball, t, node_match=root_node_match).is_isomorphic()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None
    raise RuntimeError(f"ambiguous match at {node}: {hits}")


def isolation_violations(graph, tpl, k, planted_roots):
    """Count non-root nodes whose radius-r ball is isomorphic to some template."""
    bad = 0
    for node in graph.nodes:
        if node in planted_roots:
            continue
        ball = radius_ball(graph, node, radius(k))
        for t in tpl.values():
            if iso.GraphMatcher(ball, t, node_match=root_node_match).is_isomorphic():
                bad += 1
                break
    return bad


def pairwise_non_isomorphic(tpl):
    items = list(tpl.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            gm = iso.GraphMatcher(items[i][1], items[j][1], node_match=root_node_match)
            if gm.is_isomorphic():
                return False, (items[i][0], items[j][0])
    return True, None


# ---------- driver ----------
def run_for_k(k):
    even, odd = parity_classes(k)
    tpl = templates(k)
    r = radius(k)
    spacer_len = 2 * r + 1                    # deterministic; guarantees disjoint cell balls

    non_iso, collision = pairwise_non_isomorphic(tpl)

    out = {"k": k, "radius": r, "spacer_length": spacer_len,
           "num_templates": len(tpl),
           "class_sizes": {"even": len(even), "odd": len(odd)},
           "L2_pairwise_non_isomorphic": non_iso,
           "L2_collision": collision, "classes": {}}

    recovered = {}
    for name, codes in (("even", even), ("odd", odd)):
        G, roots = assemble(codes, k, spacer_len)
        rec = []
        correct = True
        for root, intended in roots.items():
            got = match_code(G, root, tpl, k)
            rec.append(got)
            if got != intended:
                correct = False
        iso_bad = isolation_violations(G, tpl, k, set(roots))
        recovered[name] = [c for c in rec if c is not None]
        degs = [d for _, d in G.degree()]
        cell_sizes = []
        for code in codes:
            cg, _, _ = build_cell(code, "m_", k)
            cell_sizes.append(cg.number_of_nodes())
        out["classes"][name] = {
            "planted_roots": len(roots),
            "matched_roots": sum(c is not None for c in rec),
            "L1_all_codes_recovered_correctly": correct,
            "recovered_multiset_equals_class":
                sorted(recovered[name]) == sorted(tuple(c) for c in codes),
            "L3_nonroot_accidental_matches": iso_bad,
            "witness_nodes": G.number_of_nodes(),
            "witness_edges": G.number_of_edges(),
            "max_degree": max(degs),
            "per_cell_nodes_min_max": [min(cell_sizes), max(cell_sizes)],
        }

    f = frontier_order([tuple(c) for c in recovered["odd"]],
                       [tuple(c) for c in recovered["even"]], k)
    out["L4_frontier_order_from_recovered_codes"] = f
    out["L4_frontier_equals_k"] = (f == k)

    obligations = {
        "L1_realizability": all(out["classes"][n]["L1_all_codes_recovered_correctly"]
                                and out["classes"][n]["recovered_multiset_equals_class"]
                                for n in ("even", "odd")),
        "L2_non_isomorphism": non_iso,
        "L3_isolation": all(out["classes"][n]["L3_nonroot_accidental_matches"] == 0
                            for n in ("even", "odd")),
        "L4_frontier": out["L4_frontier_equals_k"],
    }
    out["obligations"] = obligations
    out["PASS"] = all(obligations.values())
    return out


def main():
    report = {"stage": "wp4_kbit_realization_exact_verification",
              "results": [run_for_k(k) for k in (3, 4, 5, 6)]}
    (RESULTS / "kbit_realization_audit.json").write_text(json.dumps(report, indent=2))

    lines = ["# WP4 exact verification — k-bit parity-class graph realization", ""]
    for res in report["results"]:
        lines.append(f"## k = {res['k']}  ->  {'PASS' if res['PASS'] else 'FAIL'}")
        lines.append(f"- radius r = {res['radius']}, deterministic spacer = {res['spacer_length']}, "
                     f"templates = {res['num_templates']}")
        for L, v in res["obligations"].items():
            lines.append(f"- {L}: {'ok' if v else 'FAIL'}")
        lines.append(f"- frontier order recovered from matcher codes = "
                     f"{res['L4_frontier_order_from_recovered_codes']} (target {res['k']})")
        for n in ("even", "odd"):
            c = res["classes"][n]
            lines.append(f"  - {n}: {c['matched_roots']}/{c['planted_roots']} roots matched, "
                         f"multiset==class {c['recovered_multiset_equals_class']}, "
                         f"accidental matches {c['L3_nonroot_accidental_matches']}, "
                         f"witness {c['witness_nodes']} nodes / max-deg {c['max_degree']}")
        lines.append("")
    (RESULTS / "kbit_realization_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("overall:", "ALL PASS" if all(r["PASS"] for r in report["results"]) else "SOME FAIL")


if __name__ == "__main__":
    sys.exit(main())
