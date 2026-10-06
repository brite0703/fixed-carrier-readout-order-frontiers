from __future__ import annotations

import json
import random
import statistics
import warnings
from collections import defaultdict
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism as iso
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

Pattern = tuple[int, int, int]
ZERO_PATTERN: Pattern = (0, 0, 0)
POSITIVE_PATTERNS: tuple[Pattern, ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
NEGATIVE_PATTERNS: tuple[Pattern, ...] = ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))
ALL_PATTERNS: tuple[Pattern, ...] = tuple(product((0, 1), repeat=3))
DEFAULT_SPLITS: dict[str, tuple[int, ...]] = {
    "train": tuple(range(12, 21)),
    "validation": tuple(range(21, 25)),
    "test": tuple(range(25, 37)),
}
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))
EXTRACTORS: tuple[str, ...] = ("isomorphism", "explicit")


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pattern_to_text(pattern: Pattern) -> str:
    return "".join(str(bit) for bit in pattern)


def pattern_from_text(text: str) -> Pattern:
    return tuple(int(bit) for bit in text)  # type: ignore[return-value]


def subset_key(subset: frozenset[int]) -> str:
    if not subset:
        return "empty"
    return "".join(str(index) for index in sorted(subset))


def subsets_of(items: frozenset[int]) -> tuple[frozenset[int], ...]:
    members = tuple(sorted(items))
    return tuple(
        frozenset(choice)
        for size in range(len(members) + 1)
        for choice in combinations(members, size)
    )


def rooted_node_match(left: dict, right: dict) -> bool:
    return bool(left.get("is_root", False)) == bool(right.get("is_root", False))


def radius_ball(graph: nx.Graph, center: str, radius: int = 4) -> nx.Graph:
    nodes = list(nx.single_source_shortest_path_length(graph, center, cutoff=radius))
    ball = graph.subgraph(nodes).copy()
    for node in ball.nodes:
        ball.nodes[node]["is_root"] = node == center
    return ball


def build_rooted_triadic_cell(pattern: Pattern, prefix: str) -> tuple[nx.Graph, str, str]:
    graph = nx.Graph()
    root = f"{prefix}r"
    graph.add_node(root)

    arm_a = [root, f"{prefix}u1", f"{prefix}a"]
    arm_b = [root, f"{prefix}v1", f"{prefix}v2", f"{prefix}b"]
    arm_c = [root, f"{prefix}w1", f"{prefix}w2", f"{prefix}w3", f"{prefix}c"]
    connector_stem = [root, f"{prefix}z1", f"{prefix}z2", f"{prefix}z3", f"{prefix}z4", f"{prefix}q"]

    for path in (arm_a, arm_b, arm_c, connector_stem):
        graph.add_nodes_from(path)
        graph.add_edges_from(zip(path, path[1:]))

    a, b, c = arm_a[-1], arm_b[-1], arm_c[-1]
    if pattern[0]:
        graph.add_edge(a, b)
    if pattern[1]:
        graph.add_edge(a, c)
    if pattern[2]:
        graph.add_edge(b, c)

    return graph, root, connector_stem[-1]


def connect_with_spacer(graph: nx.Graph, left: str, right: str, length: int, prefix: str) -> None:
    if length < 1:
        raise ValueError("Spacer length must be at least 1.")
    previous = left
    for index in range(1, length):
        current = f"{prefix}{index}"
        graph.add_edge(previous, current)
        previous = current
    graph.add_edge(previous, right)


def nuisance_anchor_positions(spacer_length: int, intensity: int) -> list[int]:
    if intensity <= 0:
        return []
    if spacer_length <= 6:
        usable = list(range(1, spacer_length))
    else:
        usable = list(range(3, spacer_length - 2))
    if not usable:
        return []
    if intensity >= len(usable):
        return usable
    return sorted({usable[(index * len(usable)) // intensity] for index in range(intensity)})


def add_nuisance_decorations(graph: nx.Graph, spacer_length: int, intensity: int = 1) -> None:
    for spacer_index in range(3):
        for local_index, anchor_position in enumerate(nuisance_anchor_positions(spacer_length, intensity)):
            anchor = f"sp{spacer_index}_{anchor_position}"
            tail_length = 1 + ((local_index + spacer_index) % 3)
            previous = anchor
            for depth in range(1, tail_length + 1):
                current = f"dec{spacer_index}_{anchor_position}_{depth}"
                graph.add_edge(previous, current)
                previous = current


def build_witness_graph(
    patterns: Iterable[Pattern],
    spacer_length: int,
    nuisance_intensity: int = 0,
) -> dict:
    graph = nx.Graph()
    roots: list[str] = []
    connectors: list[str] = []
    root_patterns: dict[str, Pattern] = {}

    for cell_index, pattern in enumerate(patterns):
        prefix = f"c{cell_index}_"
        cell_graph, root, connector = build_rooted_triadic_cell(pattern, prefix=prefix)
        graph = nx.compose(graph, cell_graph)
        roots.append(root)
        connectors.append(connector)
        root_patterns[root] = pattern

    for spacer_index in range(len(connectors) - 1):
        connect_with_spacer(
            graph,
            connectors[spacer_index],
            connectors[spacer_index + 1],
            length=spacer_length,
            prefix=f"sp{spacer_index}_",
        )

    if nuisance_intensity > 0:
        add_nuisance_decorations(graph, spacer_length=spacer_length, intensity=nuisance_intensity)

    return {
        "graph": graph,
        "roots": tuple(roots),
        "connectors": tuple(connectors),
        "root_patterns": root_patterns,
    }


@lru_cache(maxsize=1)
def template_graphs() -> dict[Pattern, nx.Graph]:
    templates: dict[Pattern, nx.Graph] = {}
    for pattern in ALL_PATTERNS:
        graph, root, _ = build_rooted_triadic_cell(pattern, prefix=f"tpl_{pattern_to_text(pattern)}_")
        templates[pattern] = radius_ball(graph, root, radius=4)
    return templates


def matched_template_isomorphism(graph: nx.Graph, node: str) -> Pattern | None:
    if graph.degree[node] != 4:
        return None

    rooted_ball = radius_ball(graph, node, radius=4)
    matches: list[Pattern] = []
    for pattern, template in template_graphs().items():
        matcher = iso.GraphMatcher(rooted_ball, template, node_match=rooted_node_match)
        if matcher.is_isomorphic():
            matches.append(pattern)

    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected a unique rooted-template match for {node}, found {matches}.")
    return matches[0]


def rooted_layer_signature(graph: nx.Graph, node: str, radius: int = 4) -> tuple:
    ball = radius_ball(graph, node, radius=radius)
    distances = nx.single_source_shortest_path_length(ball, node, cutoff=radius)
    depth_counts = tuple(sum(1 for depth in distances.values() if depth == layer) for layer in range(radius + 1))

    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for left, right in ball.edges():
        left_depth = distances[left]
        right_depth = distances[right]
        edge_counts[tuple(sorted((left_depth, right_depth)))] += 1

    degree_signature = tuple(
        tuple(sorted(ball.degree[current] for current, depth in distances.items() if depth == layer))
        for layer in range(radius + 1)
    )
    return (
        depth_counts,
        tuple(sorted(edge_counts.items())),
        degree_signature,
    )


@lru_cache(maxsize=1)
def template_signature_map() -> dict[tuple, Pattern]:
    mapping: dict[tuple, Pattern] = {}
    for pattern in ALL_PATTERNS:
        graph, root, _ = build_rooted_triadic_cell(pattern, prefix=f"sig_{pattern_to_text(pattern)}_")
        signature = rooted_layer_signature(graph, root, radius=4)
        if signature in mapping:
            raise RuntimeError(f"Template signature collision for {pattern} and {mapping[signature]}.")
        mapping[signature] = pattern
    return mapping


def matched_template_explicit(graph: nx.Graph, node: str) -> Pattern | None:
    if graph.degree[node] != 4:
        return None
    return template_signature_map().get(rooted_layer_signature(graph, node, radius=4))


def matched_template(graph: nx.Graph, node: str, extractor: str = "isomorphism") -> Pattern | None:
    if extractor == "isomorphism":
        return matched_template_isomorphism(graph, node)
    if extractor == "explicit":
        return matched_template_explicit(graph, node)
    raise ValueError(f"Unknown extractor '{extractor}'. Expected one of {EXTRACTORS}.")


def compute_template_matches(graph: nx.Graph, extractor: str = "isomorphism") -> dict[str, Pattern | None]:
    return {node: matched_template(graph, node, extractor=extractor) for node in graph.nodes}


def compute_local_carrier(graph: nx.Graph, extractor: str = "isomorphism") -> dict[str, Pattern]:
    return {
        node: ZERO_PATTERN if pattern is None else pattern
        for node, pattern in compute_template_matches(graph, extractor=extractor).items()
    }


def aggregate_moments(carrier: dict[str, Pattern]) -> dict[str, int]:
    values = np.asarray(list(carrier.values()), dtype=int)
    x1, x2, x3 = values[:, 0], values[:, 1], values[:, 2]
    return {
        "sum_x1": int(x1.sum()),
        "sum_x2": int(x2.sum()),
        "sum_x3": int(x3.sum()),
        "sum_x1x2": int((x1 * x2).sum()),
        "sum_x1x3": int((x1 * x3).sum()),
        "sum_x2x3": int((x2 * x3).sum()),
        "sum_x1x2x3": int((x1 * x2 * x3).sum()),
    }


def pattern_histogram(carrier: dict[str, Pattern]) -> dict[str, int]:
    counts = {pattern_to_text(pattern): 0 for pattern in ALL_PATTERNS}
    for pattern in carrier.values():
        counts[pattern_to_text(pattern)] += 1
    return counts


def expected_moments(label_name: str) -> dict[str, int]:
    triple = 1 if label_name == "positive" else 0
    return {
        "sum_x1": 2,
        "sum_x2": 2,
        "sum_x3": 2,
        "sum_x1x2": 1,
        "sum_x1x3": 1,
        "sum_x2x3": 1,
        "sum_x1x2x3": triple,
    }


def basis_names(degree: int) -> list[str]:
    names = ["1", "x1", "x2", "x3"]
    if degree >= 2:
        names.extend(["x1x2", "x1x3", "x2x3"])
    if degree >= 3:
        names.append("x1x2x3")
    return names


def graph_feature_vector(num_vertices: int, moments: dict[str, int], degree: int) -> list[float]:
    features: list[float] = [
        float(num_vertices),
        float(moments["sum_x1"]),
        float(moments["sum_x2"]),
        float(moments["sum_x3"]),
    ]
    if degree >= 2:
        features.extend(
            [
                float(moments["sum_x1x2"]),
                float(moments["sum_x1x3"]),
                float(moments["sum_x2x3"]),
            ]
        )
    if degree >= 3:
        features.append(float(moments["sum_x1x2x3"]))
    return features


def build_record(
    label_name: str,
    spacer_length: int,
    nuisance_intensity: int = 0,
    extractor: str = "isomorphism",
) -> dict:
    label = 1 if label_name == "positive" else 0
    patterns = POSITIVE_PATTERNS if label_name == "positive" else NEGATIVE_PATTERNS
    witness = build_witness_graph(
        patterns,
        spacer_length=spacer_length,
        nuisance_intensity=nuisance_intensity,
    )
    graph = witness["graph"]
    template_matches = compute_template_matches(graph, extractor=extractor)
    carrier = {
        node: ZERO_PATTERN if pattern is None else pattern
        for node, pattern in template_matches.items()
    }
    moments = aggregate_moments(carrier)
    matched_templates = {
        node: pattern_to_text(pattern)
        for node, pattern in template_matches.items()
        if pattern is not None
    }
    expected_root_patterns = {
        root: pattern_to_text(pattern) for root, pattern in witness["root_patterns"].items()
    }
    unexpected_matches = sorted(node for node in matched_templates if node not in witness["roots"])
    missing_roots = sorted(root for root in witness["roots"] if root not in matched_templates)
    root_pattern_mismatches = {
        root: {
            "expected": expected_root_patterns[root],
            "observed": matched_templates.get(root, "missing"),
        }
        for root in witness["roots"]
        if matched_templates.get(root) != expected_root_patterns[root]
    }
    diagnostics_ok = (
        not unexpected_matches
        and not missing_roots
        and not root_pattern_mismatches
        and moments == expected_moments(label_name)
    )

    return {
        "label": label,
        "label_name": label_name,
        "spacer_length": spacer_length,
        "nuisance_intensity": nuisance_intensity,
        "extractor": extractor,
        "num_vertices": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "roots": list(witness["roots"]),
        "expected_root_patterns": expected_root_patterns,
        "matched_templates": matched_templates,
        "pattern_counts": pattern_histogram(carrier),
        "moments": moments,
        "features": {str(degree): graph_feature_vector(graph.number_of_nodes(), moments, degree) for degree in (1, 2, 3)},
        "diagnostics": {
            "ok": diagnostics_ok,
            "unexpected_matches": unexpected_matches,
            "missing_roots": missing_roots,
            "root_pattern_mismatches": root_pattern_mismatches,
        },
    }


def build_dataset(
    splits: dict[str, tuple[int, ...]] | None = None,
    nuisance_intensity: int = 0,
    extractor: str = "isomorphism",
) -> dict[str, list[dict]]:
    active_splits = splits or DEFAULT_SPLITS
    return {
        split: [
            build_record(
                label_name=label_name,
                spacer_length=spacer_length,
                nuisance_intensity=nuisance_intensity,
                extractor=extractor,
            )
            for spacer_length in spacer_lengths
            for label_name in ("positive", "negative")
        ]
        for split, spacer_lengths in active_splits.items()
    }


def flatten_dataset(dataset: dict[str, list[dict]]) -> list[dict]:
    return [record for records in dataset.values() for record in records]


def verify_dataset(dataset: dict[str, list[dict]]) -> dict[str, int | bool]:
    all_records = flatten_dataset(dataset)
    return {
        "all_ok": all(record["diagnostics"]["ok"] for record in all_records),
        "total_graphs": len(all_records),
        "graphs_with_ok_diagnostics": sum(record["diagnostics"]["ok"] for record in all_records),
    }


def split_matrix(records: list[dict], degree: int) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([record["features"][str(degree)] for record in records], dtype=float)
    labels = np.asarray([record["label"] for record in records], dtype=int)
    return features, labels


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean())


def summarize_seed_runs(seed_runs: list[dict]) -> dict:
    metrics = ("train_accuracy", "validation_accuracy", "test_accuracy")
    summary = {
        metric: {
            "mean": statistics.mean(run[metric] for run in seed_runs),
            "min": min(run[metric] for run in seed_runs),
            "max": max(run[metric] for run in seed_runs),
        }
        for metric in metrics
    }
    summary["seeds"] = seed_runs
    return summary


def run_linear_ladder(dataset: dict[str, list[dict]], seeds: Iterable[int] = DEFAULT_SEEDS) -> dict:
    results = {
        "feature_names": {str(degree): basis_names(degree) for degree in (1, 2, 3)},
        "per_degree": {},
    }
    train_x: dict[int, np.ndarray] = {}
    train_y: dict[int, np.ndarray] = {}
    validation_x: dict[int, np.ndarray] = {}
    validation_y: dict[int, np.ndarray] = {}
    test_x: dict[int, np.ndarray] = {}
    test_y: dict[int, np.ndarray] = {}

    for degree in (1, 2, 3):
        train_x[degree], train_y[degree] = split_matrix(dataset["train"], degree)
        validation_x[degree], validation_y[degree] = split_matrix(dataset["validation"], degree)
        test_x[degree], test_y[degree] = split_matrix(dataset["test"], degree)

    for degree in (1, 2, 3):
        seed_runs: list[dict] = []
        for seed in seeds:
            classifier = LogisticRegression(
                fit_intercept=False,
                random_state=seed,
                solver="liblinear",
                penalty="l2",
                C=1_000_000.0,
                max_iter=2_000,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier.fit(train_x[degree], train_y[degree])

            train_predictions = classifier.predict(train_x[degree])
            validation_predictions = classifier.predict(validation_x[degree])
            test_predictions = classifier.predict(test_x[degree])
            seed_runs.append(
                {
                    "seed": seed,
                    "train_accuracy": accuracy(train_predictions, train_y[degree]),
                    "validation_accuracy": accuracy(validation_predictions, validation_y[degree]),
                    "test_accuracy": accuracy(test_predictions, test_y[degree]),
                    "weights": classifier.coef_[0].tolist(),
                }
            )

        results["per_degree"][str(degree)] = summarize_seed_runs(seed_runs)

    return results


def summarize_degree_three_support(dataset: dict[str, list[dict]], linear_results: dict) -> dict:
    feature_names = linear_results["feature_names"]["3"]
    seed_runs = linear_results["per_degree"]["3"]["seeds"]
    coefficient_table: list[dict] = []
    all_records = flatten_dataset(dataset)
    positive_records = [record for record in all_records if record["label"] == 1]
    negative_records = [record for record in all_records if record["label"] == 0]

    positive_mean = np.asarray([record["features"]["3"] for record in positive_records], dtype=float).mean(axis=0)
    negative_mean = np.asarray([record["features"]["3"] for record in negative_records], dtype=float).mean(axis=0)
    feature_delta = positive_mean - negative_mean

    for index, name in enumerate(feature_names):
        weights = [seed["weights"][index] for seed in seed_runs]
        coefficient_table.append(
            {
                "term": name,
                "weight_mean": statistics.mean(weights),
                "weight_min": min(weights),
                "weight_max": max(weights),
                "positive_minus_negative_feature_delta": float(feature_delta[index]),
                "mean_separating_contribution": float(statistics.mean(weights) * feature_delta[index]),
            }
        )

    decisive = max(coefficient_table, key=lambda row: abs(row["mean_separating_contribution"]))
    return {
        "coefficient_table": coefficient_table,
        "decisive_term": decisive["term"],
        "decisive_contribution": decisive["mean_separating_contribution"],
    }


def capacity_from_values(values: dict[str, float]) -> dict[frozenset[int], float]:
    return {
        frozenset(): 0.0,
        frozenset({1}): values["1"],
        frozenset({2}): values["2"],
        frozenset({3}): values["3"],
        frozenset({1, 2}): values["12"],
        frozenset({1, 3}): values["13"],
        frozenset({2, 3}): values["23"],
        frozenset({1, 2, 3}): values["123"],
    }


def serialize_capacity(capacity: dict[frozenset[int], float]) -> dict[str, float]:
    return {subset_key(subset): float(value) for subset, value in capacity.items()}


def mobius_from_capacity(capacity: dict[frozenset[int], float]) -> dict[frozenset[int], float]:
    mobius: dict[frozenset[int], float] = {}
    for subset in sorted(capacity, key=lambda item: (len(item), tuple(sorted(item)))):
        total = 0.0
        for smaller in subsets_of(subset):
            total += ((-1) ** (len(subset) - len(smaller))) * capacity[smaller]
        mobius[subset] = float(total)
    return mobius


def serialize_mobius(mobius: dict[frozenset[int], float]) -> dict[str, float]:
    return {subset_key(subset): float(value) for subset, value in mobius.items()}


def choquet_integral(x: Pattern | tuple[float, float, float], capacity: dict[frozenset[int], float]) -> float:
    ordered = sorted(enumerate(x, start=1), key=lambda item: (item[1], item[0]))
    previous = 0.0
    total = 0.0
    for index, (_, value) in enumerate(ordered):
        active = frozenset(node_index for node_index, _ in ordered[index:])
        total += (float(value) - previous) * capacity[active]
        previous = float(value)
    return total


def additive_capacity(rng: random.Random) -> dict[frozenset[int], float]:
    weights = np.asarray([rng.random(), rng.random(), rng.random()], dtype=float)
    weights /= weights.sum()
    values = {
        "1": float(weights[0]),
        "2": float(weights[1]),
        "3": float(weights[2]),
        "12": float(weights[0] + weights[1]),
        "13": float(weights[0] + weights[2]),
        "23": float(weights[1] + weights[2]),
        "123": 1.0,
    }
    return capacity_from_values(values)


def sample_two_additive_capacity(rng: random.Random) -> dict[frozenset[int], float]:
    for _ in range(10_000):
        mu1, mu2, mu3 = rng.random(), rng.random(), rng.random()
        mu12 = max(mu1, mu2) + rng.random() * (1.0 - max(mu1, mu2))
        mu13 = max(mu1, mu3) + rng.random() * (1.0 - max(mu1, mu3))
        mu23 = 1.0 + mu1 + mu2 + mu3 - mu12 - mu13
        if max(mu2, mu3) <= mu23 <= 1.0:
            return capacity_from_values(
                {
                    "1": mu1,
                    "2": mu2,
                    "3": mu3,
                    "12": mu12,
                    "13": mu13,
                    "23": mu23,
                    "123": 1.0,
                }
            )
    raise RuntimeError("Failed to sample a valid 2-additive capacity.")


def explicit_triple_capacity() -> dict[frozenset[int], float]:
    return capacity_from_values(
        {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
            "12": 0.0,
            "13": 0.0,
            "23": 0.0,
            "123": 1.0,
        }
    )


def score_record_with_capacity(record: dict, capacity: dict[frozenset[int], float]) -> float:
    total = 0.0
    for pattern_text, count in record["pattern_counts"].items():
        if count == 0:
            continue
        total += count * choquet_integral(pattern_from_text(pattern_text), capacity)
    return total


def paired_score_gaps(records: list[dict], capacity: dict[frozenset[int], float]) -> list[float]:
    use_pair_ids = all("pair_id" in record for record in records)
    by_pair: dict[object, dict[str, float]] = {}
    for record in records:
        pair_key: object = (
            int(record["pair_id"])
            if use_pair_ids
            else ("legacy_spacer_length", int(record["spacer_length"]))
        )
        by_pair.setdefault(pair_key, {})[record["label_name"]] = score_record_with_capacity(record, capacity)
    incomplete = [key for key, pair in by_pair.items() if set(pair) != {"positive", "negative"}]
    if incomplete:
        raise RuntimeError(f"Incomplete positive/negative score pairs for keys: {incomplete[:10]}")
    return [
        pair["positive"] - pair["negative"]
        for _, pair in sorted(by_pair.items(), key=lambda item: str(item[0]))
    ]


def best_threshold(scores: list[float], labels: list[int]) -> dict:
    unique_scores = sorted(set(scores))
    if len(unique_scores) == 1:
        candidates = [unique_scores[0] - 1.0, unique_scores[0] + 1.0]
    else:
        candidates = [unique_scores[0] - 1.0]
        candidates.extend((left + right) / 2.0 for left, right in zip(unique_scores, unique_scores[1:]))
        candidates.append(unique_scores[-1] + 1.0)

    best: dict | None = None
    for orientation in ("ge", "le"):
        for threshold in candidates:
            predictions = [
                int(score >= threshold) if orientation == "ge" else int(score <= threshold)
                for score in scores
            ]
            current_accuracy = sum(pred == label for pred, label in zip(predictions, labels)) / len(labels)
            if best is None or current_accuracy > best["accuracy"]:
                best = {
                    "orientation": orientation,
                    "threshold": float(threshold),
                    "accuracy": float(current_accuracy),
                }
    if best is None:
        raise RuntimeError("Threshold search failed.")
    return best


def apply_threshold(scores: list[float], labels: list[int], rule: dict) -> dict:
    predictions = [
        int(score >= rule["threshold"]) if rule["orientation"] == "ge" else int(score <= rule["threshold"])
        for score in scores
    ]
    return {
        "accuracy": float(sum(pred == label for pred, label in zip(predictions, labels)) / len(labels)),
        "scores": [float(score) for score in scores],
    }


def representative_capacity_result(
    dataset: dict[str, list[dict]],
    capacity: dict[frozenset[int], float],
) -> dict:
    train_scores = [score_record_with_capacity(record, capacity) for record in dataset["train"]]
    train_labels = [record["label"] for record in dataset["train"]]
    rule = best_threshold(train_scores, train_labels)
    return {
        "capacity": serialize_capacity(capacity),
        "threshold_rule": rule,
        "train": apply_threshold(train_scores, train_labels, rule),
        "validation": apply_threshold(
            [score_record_with_capacity(record, capacity) for record in dataset["validation"]],
            [record["label"] for record in dataset["validation"]],
            rule,
        ),
        "test": apply_threshold(
            [score_record_with_capacity(record, capacity) for record in dataset["test"]],
            [record["label"] for record in dataset["test"]],
            rule,
        ),
    }


def run_shared_choquet_experiment(
    dataset: dict[str, list[dict]],
    additive_samples: int = 64,
    two_additive_samples: int = 256,
) -> dict:
    all_records = flatten_dataset(dataset)

    one_additive_gaps: list[float] = []
    one_additive_capacity = additive_capacity(random.Random(0))
    for seed in range(additive_samples):
        capacity = additive_capacity(random.Random(seed))
        one_additive_gaps.extend(abs(gap) for gap in paired_score_gaps(all_records, capacity))

    two_additive_gaps: list[float] = []
    two_additive_capacity = sample_two_additive_capacity(random.Random(0))
    for seed in range(two_additive_samples):
        capacity = sample_two_additive_capacity(random.Random(seed))
        two_additive_gaps.extend(abs(gap) for gap in paired_score_gaps(all_records, capacity))

    three_additive_capacity = explicit_triple_capacity()
    three_additive_gaps = [abs(gap) for gap in paired_score_gaps(all_records, three_additive_capacity)]

    return {
        "1_additive": {
            "sample_count": additive_samples,
            "max_abs_positive_negative_gap": float(max(one_additive_gaps, default=0.0)),
            "representative_result": representative_capacity_result(dataset, one_additive_capacity),
        },
        "2_additive": {
            "sample_count": two_additive_samples,
            "max_abs_positive_negative_gap": float(max(two_additive_gaps, default=0.0)),
            "representative_result": representative_capacity_result(dataset, two_additive_capacity),
        },
        "3_additive": {
            "construction": "explicit_full_triple_capacity",
            "max_abs_positive_negative_gap": float(max(three_additive_gaps, default=0.0)),
            "representative_result": representative_capacity_result(dataset, three_additive_capacity),
        },
    }


def stage_summary_markdown(stage_name: str, linear_results: dict) -> list[str]:
    lines = [f"## {stage_name}", ""]
    for degree in (1, 2, 3):
        summary = linear_results["per_degree"][str(degree)]
        lines.append(
            f"- Degree {degree}: "
            f"train {summary['train_accuracy']['mean']:.3f}, "
            f"validation {summary['validation_accuracy']['mean']:.3f}, "
            f"test {summary['test_accuracy']['mean']:.3f}"
        )
    lines.append("")
    return lines


def choquet_summary_markdown(choquet_results: dict, title: str = "Stage E1") -> list[str]:
    lines = [f"## {title}", ""]
    for key in ("1_additive", "2_additive", "3_additive"):
        result = choquet_results[key]
        representative = result["representative_result"]
        lines.append(
            f"- {key.replace('_', '-')} Choquet: "
            f"gap {result['max_abs_positive_negative_gap']:.3f}, "
            f"train {representative['train']['accuracy']:.3f}, "
            f"validation {representative['validation']['accuracy']:.3f}, "
            f"test {representative['test']['accuracy']:.3f}"
        )
    lines.append("")
    return lines


def linear_frontier_holds(linear_results: dict, tolerance: float = 1e-9) -> bool:
    degree_1 = linear_results["per_degree"]["1"]["test_accuracy"]["mean"]
    degree_2 = linear_results["per_degree"]["2"]["test_accuracy"]["mean"]
    degree_3 = linear_results["per_degree"]["3"]["test_accuracy"]["mean"]
    return (
        abs(degree_1 - 0.5) <= tolerance
        and abs(degree_2 - 0.5) <= tolerance
        and abs(degree_3 - 1.0) <= tolerance
    )


def choquet_frontier_holds(choquet_results: dict, tolerance: float = 1e-9) -> bool:
    return (
        abs(choquet_results["1_additive"]["representative_result"]["test"]["accuracy"] - 0.5) <= tolerance
        and abs(choquet_results["2_additive"]["representative_result"]["test"]["accuracy"] - 0.5) <= tolerance
        and abs(choquet_results["3_additive"]["representative_result"]["test"]["accuracy"] - 1.0) <= tolerance
        and abs(choquet_results["1_additive"]["max_abs_positive_negative_gap"]) <= tolerance
        and abs(choquet_results["2_additive"]["max_abs_positive_negative_gap"]) <= tolerance
        and abs(choquet_results["3_additive"]["max_abs_positive_negative_gap"] - 1.0) <= tolerance
    )


def write_summary_markdown(path: Path, payloads: dict[str, dict]) -> None:
    diagnostics_e0 = payloads["E0"]["diagnostics"]
    diagnostics_e2 = payloads["E2"]["diagnostics"]
    diagnostics_e6 = payloads["E6"]["diagnostics"]
    lines = [
        "# Frontier Experiment Results",
        "",
        "## Diagnostics",
        "",
        (
            f"- Stage E0 graphs with clean rooted-template diagnostics: "
            f"{diagnostics_e0['graphs_with_ok_diagnostics']}/{diagnostics_e0['total_graphs']}"
        ),
        (
            f"- Stage E2 graphs with clean rooted-template diagnostics: "
            f"{diagnostics_e2['graphs_with_ok_diagnostics']}/{diagnostics_e2['total_graphs']}"
        ),
        (
            f"- Stage E6 graphs with clean explicit-extractor diagnostics: "
            f"{diagnostics_e6['graphs_with_ok_diagnostics']}/{diagnostics_e6['total_graphs']}"
        ),
        "",
    ]
    lines.extend(stage_summary_markdown("Stage E0", payloads["E0"]["linear_ladder"]))
    lines.extend(choquet_summary_markdown(payloads["E1"]["choquet"], title="Stage E1"))
    lines.extend(stage_summary_markdown("Stage E2", payloads["E2"]["linear_ladder"]))
    lines.extend(choquet_summary_markdown(payloads["E2"]["choquet"], title="Stage E2 Choquet"))
    lines.extend(
        [
            "## Stage E3",
            "",
            (
                f"- Polynomial support audit decisive term: "
                f"{payloads['E3']['polynomial_support']['decisive_term']}"
            ),
            (
                f"- Shared 3-additive Choquet decisive Möbius term: "
                f"{payloads['E3']['choquet_support']['decisive_term']}"
            ),
            "",
            "## Stage E4",
            "",
        ]
    )
    for regime_name, regime_payload in payloads["E4"]["regimes"].items():
        lines.append(
            f"- {regime_name}: "
            f"linear frontier={linear_frontier_holds(regime_payload['linear_ladder'])}, "
            f"Choquet frontier={choquet_frontier_holds(regime_payload['choquet'])}"
        )
    lines.extend(
        [
            "",
            "## Stage E5",
            "",
            (
                f"- Carrier consistency: "
                f"{payloads['E5']['graphs_with_identical_carrier']}/{payloads['E5']['total_graphs']} graphs identical"
            ),
            "",
            "## Stage E6",
            "",
        ]
    )
    lines.extend(stage_summary_markdown("Explicit Extractor E0", payloads["E6"]["linear_ladder"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_e0_stage() -> dict:
    dataset = build_dataset(nuisance_intensity=0, extractor="isomorphism")
    return {
        "stage": "E0",
        "diagnostics": verify_dataset(dataset),
        "splits": DEFAULT_SPLITS,
        "linear_ladder": run_linear_ladder(dataset),
    }


def run_e1_stage() -> dict:
    dataset = build_dataset(nuisance_intensity=0, extractor="isomorphism")
    return {
        "stage": "E1",
        "diagnostics": verify_dataset(dataset),
        "splits": DEFAULT_SPLITS,
        "choquet": run_shared_choquet_experiment(dataset),
    }


def run_e2_stage() -> dict:
    dataset = build_dataset(nuisance_intensity=1, extractor="isomorphism")
    return {
        "stage": "E2",
        "diagnostics": verify_dataset(dataset),
        "splits": DEFAULT_SPLITS,
        "linear_ladder": run_linear_ladder(dataset),
        "choquet": run_shared_choquet_experiment(dataset),
    }


def run_e2_d_stage() -> dict:
    clean_dataset = build_dataset(nuisance_intensity=0, extractor="isomorphism")
    nuisance_dataset = build_dataset(nuisance_intensity=1, extractor="isomorphism")
    return {
        "stage": "E2-D",
        "clean": verify_dataset(clean_dataset),
        "nuisance": verify_dataset(nuisance_dataset),
    }


def run_e3_stage() -> dict:
    dataset = build_dataset(nuisance_intensity=0, extractor="isomorphism")
    linear_results = run_linear_ladder(dataset)
    degree_three_support = summarize_degree_three_support(dataset, linear_results)

    successful_capacity = explicit_triple_capacity()
    mobius = mobius_from_capacity(successful_capacity)
    return {
        "stage": "E3",
        "polynomial_support": degree_three_support,
        "choquet_support": {
            "capacity": serialize_capacity(successful_capacity),
            "mobius": serialize_mobius(mobius),
            "decisive_term": "123",
            "representative_result": representative_capacity_result(dataset, successful_capacity),
        },
        "interpretation": (
            "Only the aggregated triple moment differs between positive and negative graphs. "
            "The successful polynomial model therefore separates through the x1x2x3 term, "
            "and the successful shared Choquet construction has only the full triple Möbius coefficient active."
        ),
    }


def stress_regimes() -> tuple[dict, ...]:
    return (
        {
            "name": "small_clean",
            "splits": {
                "train": tuple(range(4, 11)),
                "validation": tuple(range(11, 15)),
                "test": tuple(range(15, 25)),
            },
            "nuisance_intensity": 0,
        },
        {
            "name": "baseline_clean",
            "splits": DEFAULT_SPLITS,
            "nuisance_intensity": 0,
        },
        {
            "name": "long_clean",
            "splits": {
                "train": tuple(range(40, 61)),
                "validation": tuple(range(61, 71)),
                "test": tuple(range(71, 91)),
            },
            "nuisance_intensity": 0,
        },
        {
            "name": "long_nuisance_light",
            "splits": {
                "train": tuple(range(40, 61)),
                "validation": tuple(range(61, 71)),
                "test": tuple(range(71, 91)),
            },
            "nuisance_intensity": 1,
        },
        {
            "name": "long_nuisance_heavy",
            "splits": {
                "train": tuple(range(40, 61)),
                "validation": tuple(range(61, 71)),
                "test": tuple(range(71, 91)),
            },
            "nuisance_intensity": 3,
        },
    )


def write_e4_plot(path: Path, payload: dict) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    orders = [1, 2, 3]
    for regime_name, regime_payload in payload["regimes"].items():
        linear_curve = [
            regime_payload["linear_ladder"]["per_degree"][str(order)]["test_accuracy"]["mean"]
            for order in orders
        ]
        choquet_curve = [
            regime_payload["choquet"][f"{order}_additive"]["representative_result"]["test"]["accuracy"]
            for order in orders
        ]
        axes[0].plot(orders, linear_curve, marker="o", label=regime_name)
        axes[1].plot(orders, choquet_curve, marker="o", label=regime_name)

    axes[0].set_title("Polynomial Readout Order")
    axes[1].set_title("Choquet Additivity Order")
    for axis in axes:
        axis.set_xlabel("Order")
        axis.set_ylabel("Test Accuracy")
        axis.set_xticks(orders)
        axis.set_ylim(0.45, 1.05)
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_e4_stage() -> dict:
    regimes: dict[str, dict] = {}
    for regime in stress_regimes():
        dataset = build_dataset(
            splits=regime["splits"],
            nuisance_intensity=regime["nuisance_intensity"],
            extractor="isomorphism",
        )
        linear_results = run_linear_ladder(dataset)
        choquet_results = run_shared_choquet_experiment(dataset, additive_samples=32, two_additive_samples=128)
        regimes[regime["name"]] = {
            "splits": regime["splits"],
            "nuisance_intensity": regime["nuisance_intensity"],
            "diagnostics": verify_dataset(dataset),
            "linear_ladder": linear_results,
            "choquet": choquet_results,
            "frontier_holds": {
                "linear": linear_frontier_holds(linear_results),
                "choquet": choquet_frontier_holds(choquet_results),
            },
        }

    return {
        "stage": "E4",
        "regimes": regimes,
    }


def compare_extractors_on_graph(
    label_name: str,
    spacer_length: int,
    nuisance_intensity: int,
) -> dict:
    patterns = POSITIVE_PATTERNS if label_name == "positive" else NEGATIVE_PATTERNS
    witness = build_witness_graph(
        patterns,
        spacer_length=spacer_length,
        nuisance_intensity=nuisance_intensity,
    )
    graph = witness["graph"]
    exact_matches = compute_template_matches(graph, extractor="isomorphism")
    explicit_matches = compute_template_matches(graph, extractor="explicit")
    exact_carrier = {
        node: ZERO_PATTERN if pattern is None else pattern
        for node, pattern in exact_matches.items()
    }
    explicit_carrier = {
        node: ZERO_PATTERN if pattern is None else pattern
        for node, pattern in explicit_matches.items()
    }
    exact_record = build_record(
        label_name=label_name,
        spacer_length=spacer_length,
        nuisance_intensity=nuisance_intensity,
        extractor="isomorphism",
    )
    explicit_record = build_record(
        label_name=label_name,
        spacer_length=spacer_length,
        nuisance_intensity=nuisance_intensity,
        extractor="explicit",
    )
    return {
        "carrier_equal": exact_carrier == explicit_carrier,
        "matches_equal": exact_matches == explicit_matches,
        "moments_equal": exact_record["moments"] == explicit_record["moments"],
        "diagnostics_equal": exact_record["diagnostics"] == explicit_record["diagnostics"],
    }


def run_e5_stage() -> dict:
    regimes = list(stress_regimes()) + [
        {
            "name": "baseline_nuisance",
            "splits": DEFAULT_SPLITS,
            "nuisance_intensity": 1,
        }
    ]
    total_graphs = 0
    graphs_with_identical_carrier = 0
    graphs_with_identical_matches = 0
    graphs_with_identical_moments = 0
    graphs_with_identical_diagnostics = 0
    per_regime: dict[str, dict] = {}

    for regime in regimes:
        regime_total = 0
        regime_carrier = 0
        regime_matches = 0
        regime_moments = 0
        regime_diagnostics = 0
        for split, spacer_lengths in regime["splits"].items():
            for spacer_length in spacer_lengths:
                for label_name in ("positive", "negative"):
                    comparison = compare_extractors_on_graph(
                        label_name=label_name,
                        spacer_length=spacer_length,
                        nuisance_intensity=regime["nuisance_intensity"],
                    )
                    total_graphs += 1
                    regime_total += 1
                    graphs_with_identical_carrier += int(comparison["carrier_equal"])
                    graphs_with_identical_matches += int(comparison["matches_equal"])
                    graphs_with_identical_moments += int(comparison["moments_equal"])
                    graphs_with_identical_diagnostics += int(comparison["diagnostics_equal"])
                    regime_carrier += int(comparison["carrier_equal"])
                    regime_matches += int(comparison["matches_equal"])
                    regime_moments += int(comparison["moments_equal"])
                    regime_diagnostics += int(comparison["diagnostics_equal"])
        per_regime[regime["name"]] = {
            "total_graphs": regime_total,
            "identical_carrier": regime_carrier,
            "identical_matches": regime_matches,
            "identical_moments": regime_moments,
            "identical_diagnostics": regime_diagnostics,
        }

    return {
        "stage": "E5",
        "total_graphs": total_graphs,
        "graphs_with_identical_carrier": graphs_with_identical_carrier,
        "graphs_with_identical_matches": graphs_with_identical_matches,
        "graphs_with_identical_moments": graphs_with_identical_moments,
        "graphs_with_identical_diagnostics": graphs_with_identical_diagnostics,
        "all_identical": (
            total_graphs == graphs_with_identical_carrier
            == graphs_with_identical_matches
            == graphs_with_identical_moments
            == graphs_with_identical_diagnostics
        ),
        "per_regime": per_regime,
    }


def run_e6_stage() -> dict:
    dataset = build_dataset(nuisance_intensity=0, extractor="explicit")
    exact_e0 = run_e0_stage()
    explicit_linear = run_linear_ladder(dataset)
    return {
        "stage": "E6",
        "diagnostics": verify_dataset(dataset),
        "splits": DEFAULT_SPLITS,
        "linear_ladder": explicit_linear,
        "matches_exact_e0": explicit_linear == exact_e0["linear_ladder"],
    }
