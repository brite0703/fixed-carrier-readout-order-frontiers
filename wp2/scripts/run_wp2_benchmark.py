#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from importlib import metadata
from sklearn.metrics import roc_auc_score
from scipy.stats import t as student_t
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, PNAConv, SAGEConv, global_mean_pool
from torch_geometric.utils import to_networkx

LOCAL_SCRIPTS_ROOT = Path(__file__).resolve().parent
_workspace_candidate = Path(__file__).resolve().parents[3]
IN_REPOSITORY_LAYOUT = (_workspace_candidate / "scripts" / "frontier_common.py").exists()
WORKSPACE_ROOT = _workspace_candidate if IN_REPOSITORY_LAYOUT else LOCAL_SCRIPTS_ROOT.parent
SCRIPTS_ROOT = _workspace_candidate / "scripts" if IN_REPOSITORY_LAYOUT else LOCAL_SCRIPTS_ROOT
sys.path.insert(0, str(LOCAL_SCRIPTS_ROOT))
if SCRIPTS_ROOT.exists():
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_is_extension_package import select_host_anchors

from frontier_common import (
    NEGATIVE_PATTERNS,
    POSITIVE_PATTERNS,
    ZERO_PATTERN,
    aggregate_moments,
    pattern_histogram,
    build_witness_graph,
    compute_template_matches,
    graph_feature_vector,
    run_linear_ladder,
    run_shared_choquet_experiment,
)

WP2_ROOT = (
    WORKSPACE_ROOT / "revision_work" / "wp2_benchmark"
    if IN_REPOSITORY_LAYOUT
    else LOCAL_SCRIPTS_ROOT.parent
)
RESULTS_ROOT = WP2_ROOT / "results"
DEFAULT_HISTORICAL_MANIFEST = (
    WP2_ROOT / "historical" / "host_split_manifest_prior.json"
    if (WP2_ROOT / "historical" / "host_split_manifest_prior.json").exists()
    else WP2_ROOT / "results_clean" / "host_split_manifest.json"
)

DEFAULT_DATASET = "NCI1"
DEFAULT_HOSTS_PER_SPLIT = 400
BASE_SPACER_LENGTH = 8
SELECTED_SPLITS = ("train", "validation", "test")
BLOCK_SPLITS = (
    "train",
    "validation",
    "discarded_legacy_test",
    "discarded_viewed_test",
    "test",
)
ROOT_SEED = 20260710
DATASET_SHUFFLE_SEED = 3134173889

MODEL_ARCHITECTURES = ("gin", "graphsage", "pna")
AGGREGATORS = ("mean", "min", "max", "std")
SCALERS = ("identity", "amplification", "attenuation")
SEED_STABILITY_SD_LIMIT = 0.15
SEED_STABILITY_SAMPLES = 10
LOGGER = logging.getLogger("wp2_clean")


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def stable_seed(*parts: object) -> int:
    import hashlib

    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2 ** 32)


class SeedManager:
    """Deterministic, nested seed stream from a single numpy SeedSequence root."""

    def __init__(self, root_seed: int) -> None:
        self._sequence = np.random.SeedSequence(root_seed)

    def next_seed(self) -> int:
        child = self._sequence.spawn(1)[0]
        return int(child.generate_state(1)[0])

    def spawn_ints(self, count: int) -> list[int]:
        return [self.next_seed() for _ in range(count)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        headers = fieldnames or list(rows[0].keys())
    else:
        headers = fieldnames or []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def student_t_seed_summary(values: list[float]) -> dict[str, Any]:
    """Summarize independent seed scores with a two-sided Student-t CI."""
    if not values:
        raise ValueError("At least one seed score is required.")
    numeric = [float(value) for value in values]
    n = len(numeric)
    mean = float(statistics.mean(numeric))
    sample_sd = float(statistics.stdev(numeric)) if n > 1 else 0.0
    if n > 1:
        critical = float(student_t.ppf(0.975, df=n - 1))
        half_width = critical * sample_sd / math.sqrt(n)
    else:
        critical = float("nan")
        half_width = 0.0
    return {
        "scores": numeric,
        "mean": mean,
        "sample_sd": sample_sd,
        "ci_95_low": max(0.0, mean - half_width),
        "ci_95_high": min(1.0, mean + half_width),
        "n": n,
        "method": "two-sided 95% Student-t confidence interval for the seed mean; sample SD uses n-1; interval clamped to [0,1]",
        "degrees_of_freedom": n - 1,
        "t_critical_0.975": critical,
    }


def balanced_accuracy(y_true: list[int], y_prob: list[float]) -> float:
    y_true_a = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_prob) >= 0.5
    y_pred = y_pred_a.astype(int)
    positives = y_true_a == 1
    negatives = y_true_a == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return float((y_pred == y_true_a).mean())
    tpr = float((y_pred[positives] == 1).mean())
    tnr = float((y_pred[negatives] == 0).mean())
    return float((tpr + tnr) / 2.0)


def binary_accuracy(y_true: list[int], y_prob: list[float]) -> float:
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    return float(np.mean(np.asarray(y_true, dtype=int) == np.asarray(y_pred, dtype=int)))


def bounded_node_features(graph: nx.Graph, nodes: list[str]) -> np.ndarray:
    degrees = dict(graph.degree(nodes))
    max_degree = max(degrees.values(), default=1)
    max_degree = max(max_degree, 1)
    clustering = nx.clustering(graph, nodes=nodes)
    return np.asarray(
        [
            [1.0, float(degrees[node]) / float(max_degree), float(clustering.get(node, 0.0))]
            for node in nodes
        ],
        dtype=np.float32,
    )


@dataclass
class HostCandidate:
    dataset_index: int
    host_id: str
    graph: nx.Graph


def graph_fingerprint(graph: nx.Graph) -> dict[str, Any]:
    degrees = sorted(int(degree) for _, degree in graph.degree())
    edge_rows = sorted(
        tuple(sorted((str(left), str(right))))
        for left, right in graph.edges()
    )
    labeled_edge_payload = json.dumps(edge_rows, separators=(",", ":")).encode("utf-8")
    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "degree_sequence": degrees,
        "weisfeiler_lehman_hash": nx.weisfeiler_lehman_graph_hash(graph, iterations=3),
        "labeled_edge_sha256": hashlib.sha256(labeled_edge_payload).hexdigest(),
    }


def structural_bucket_key(fingerprint: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(fingerprint["nodes"]),
        int(fingerprint["edges"]),
        tuple(int(value) for value in fingerprint["degree_sequence"]),
        str(fingerprint["weisfeiler_lehman_hash"]),
    )


def edge_text_set(graph: nx.Graph) -> set[tuple[str, str]]:
    return {
        tuple(sorted((str(left), str(right))))
        for left, right in graph.edges()
    }


def graph_to_torch_data(graph: nx.Graph, y: int, pair_id: int, graph_id: str, condition: str, host_id: str) -> Data:
    ordered_nodes = sorted(graph.nodes(), key=str)
    index = {node: idx for idx, node in enumerate(ordered_nodes)}
    edge_pairs = []
    for left, right in graph.edges():
        edge_pairs.append((index[left], index[right]))
        edge_pairs.append((index[right], index[left]))
    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).T
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    data = Data(
        x=torch.tensor(bounded_node_features(graph, ordered_nodes), dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([int(y)], dtype=torch.long),
        pair_id=torch.tensor([int(pair_id)], dtype=torch.long),
        condition=condition,
        host_id=host_id,
        graph_id=graph_id,
    )
    return data


def build_real_host_candidates(
    dataset_name: str,
    requested_target: int,
    minimum_required: int,
    seed: int,
    dataset_root: Path,
) -> tuple[list[HostCandidate], dict[str, int]]:
    dataset = TUDataset(root=str(dataset_root), name=dataset_name)
    target = min(int(requested_target), len(dataset))
    if len(dataset) < minimum_required:
        raise RuntimeError(
            f"Dataset {dataset_name} has only {len(dataset)} graphs; minimum {minimum_required} repairable hosts required."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    candidates: list[HostCandidate] = []
    graphs_scanned = 0

    for index in indices:
        graphs_scanned += 1
        raw_graph = dataset[int(index)]
        graph = to_networkx(raw_graph, to_undirected=True)
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))
        if graph.number_of_nodes() < 6:
            continue

        if not nx.is_connected(graph):
            components = sorted(nx.connected_components(graph), key=len, reverse=True)
            if not components or len(components[0]) < 6:
                continue
            graph = graph.subgraph(components[0]).copy()

        template_matches = compute_template_matches(graph, extractor="explicit")
        if any(pattern is not None for pattern in template_matches.values()):
            continue

        candidates.append(
            HostCandidate(
                dataset_index=int(index),
                host_id=f"{dataset_name}_{int(index)}",
                graph=graph,
            )
        )
        if len(candidates) >= target:
            break

    if len(candidates) < minimum_required:
        raise RuntimeError(
            f"Found only {len(candidates)} accidental-match-free real-host candidates for {dataset_name}, "
            f"minimum {minimum_required} required (target {target}, dataset size {len(dataset)})."
        )
    return candidates, {
        "dataset_size": int(len(dataset)),
        "requested_target": int(requested_target),
        "capped_target": int(target),
        "minimum_required": int(minimum_required),
        "eligible_candidates": int(len(candidates)),
        "graphs_scanned": int(graphs_scanned),
    }


def add_partial_decoy_fragment(graph: nx.Graph, anchor: str, seed: int) -> nx.Graph:
    out = nx.Graph(graph)
    rng = random.Random(seed)
    witness = build_witness_graph(((1, 1, 0),), BASE_SPACER_LENGTH, 0)
    prefix = f"decoy_{seed}_"
    node_map = {node: f"{prefix}{node}" for node in witness["graph"].nodes()}
    motif = nx.relabel_nodes(witness["graph"], node_map)
    motif_connector = node_map[witness["connectors"][0]]

    chord_options = [
        (node_map["c0_a"], node_map["c0_b"]),
        (node_map["c0_a"], node_map["c0_c"]),
    ]
    chord = rng.choice(chord_options)
    if motif.has_edge(*chord):
        motif.remove_edge(*chord)
    stem_root = node_map["c0_r"]
    stem_connector = node_map["c0_z1"]
    if motif.has_edge(stem_root, stem_connector):
        motif.remove_edge(stem_root, stem_connector)

    out = nx.compose(out, motif)
    if anchor in out:
        out.add_edge(motif_connector, anchor)
    return out


def perturb_host_graph(graph: nx.Graph, seed: int, attempts: int = 12) -> nx.Graph:
    out = nx.Graph(graph)
    rng = random.Random(seed)

    if out.number_of_nodes() < 6 or out.number_of_edges() < 4:
        raise RuntimeError("Host is too small for a certified perturbation.")

    edges = list(out.edges())
    for _ in range(attempts):
        if len(edges) < 2:
            raise RuntimeError("Host has too few edges for a certified perturbation.")
        (left, right), (source, target) = rng.sample(edges, 2)
        if len({left, right, source, target}) < 4:
            continue
        if (
            out.has_edge(left, source)
            or out.has_edge(right, target)
            or out.has_edge(left, target)
            or out.has_edge(right, source)
        ):
            continue
        out.remove_edge(left, right)
        out.remove_edge(source, target)
        out.add_edge(left, source)
        out.add_edge(right, target)
        if nx.is_connected(out):
            edges = list(out.edges())
            return out
        out.remove_edge(left, source)
        out.remove_edge(right, target)
        out.add_edge(left, right)
        out.add_edge(source, target)
    raise RuntimeError(f"No changed connected perturbation found in {attempts} rewiring attempts.")


def apply_condition(host_graph: nx.Graph, condition: str, anchor: str, seed: int) -> nx.Graph:
    if condition == "clean":
        return nx.Graph(host_graph)
    if condition == "decoy":
        return add_partial_decoy_fragment(host_graph, anchor=anchor, seed=seed)
    if condition == "perturb":
        return perturb_host_graph(host_graph, seed=seed)
    raise ValueError(f"Unknown condition '{condition}'.")


def check_preplant_no_accidental(graph: nx.Graph) -> bool:
    matches = compute_template_matches(graph, extractor="explicit")
    return all(pattern is None for pattern in matches.values())


def build_planted_pair_record(
    condition: str,
    host: HostCandidate,
    transformed_host: nx.Graph,
    anchors: tuple[str, ...],
    label_name: str,
    spacer_length: int,
    split: str,
    pair_id: int,
    condition_seed: int,
    transform_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    patterns = POSITIVE_PATTERNS if label_name == "positive" else NEGATIVE_PATTERNS
    witness = build_witness_graph(patterns, spacer_length=spacer_length, nuisance_intensity=0)
    graph = nx.compose(transformed_host, witness["graph"])
    for connector, anchor in zip(witness["connectors"], anchors):
        graph.add_edge(connector, anchor)

    template_matches = compute_template_matches(graph, extractor="explicit")
    designated_matches = sum(
        1 for root in witness["roots"] if template_matches.get(root) is not None
    )
    if designated_matches != len(witness["roots"]):
        raise RuntimeError(
            f"Host {host.host_id} condition {condition} {label_name} missing designated rooted-template matches."
        )

    accidental_matches = sorted(
        node
        for node, pattern in template_matches.items()
        if pattern is not None and node not in witness["roots"]
    )
    if accidental_matches:
        raise RuntimeError(
            f"Host {host.host_id} condition {condition} {label_name} has accidental template matches before selection."
        )

    local_carrier = {
        node: ZERO_PATTERN if pattern is None else pattern
        for node, pattern in template_matches.items()
    }
    moments = aggregate_moments(local_carrier)
    label = 1 if label_name == "positive" else 0
    host_node_count = int(transformed_host.number_of_nodes())

    record = {
        "label": label,
        "label_name": label_name,
        "spacer_length": spacer_length,
        "split": split,
        "condition": condition,
        "host_id": host.host_id,
        "dataset_index": int(host.dataset_index),
        "graph_id": f"{condition}_{split}_{host.host_id}_{label_name}",
        "num_vertices": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "pair_id": pair_id,
        "pair_key": f"{host.host_id}_{condition}",
        "num_matches_designated_roots": int(designated_matches),
        "accidental_match_count": int(len(accidental_matches)),
        "features": {
            str(degree): graph_feature_vector(graph.number_of_nodes(), moments, degree)
            for degree in (1, 2, 3)
        },
        "pattern_counts": pattern_histogram(local_carrier),
        "pattern_match_counts": {pattern_text: int(count) for pattern_text, count in pattern_histogram(local_carrier).items()},
        "graph": graph,
        "host_nodes": host_node_count,
        "condition_seed": condition_seed,
        "anchor_nodes": [str(anchor) for anchor in anchors],
        "witness_roots": list(witness["roots"]),
        "witness_connectors": list(witness["connectors"]),
        "transform_metadata": transform_metadata,
    }

    diagnostics = {
        "host_id": host.host_id,
        "dataset_index": int(host.dataset_index),
        "condition": condition,
        "split": split,
        "label_name": label_name,
        "pair_id": pair_id,
        "host_nodes": host_node_count,
        "host_edges": int(transformed_host.number_of_edges()),
        "accidental_match_count": int(len(accidental_matches)),
        "accidental_matches": accidental_matches,
        "designated_root_matches": int(designated_matches),
        "spacer_length": spacer_length,
        "condition_seed": int(condition_seed),
        "anchor_nodes": [str(anchor) for anchor in anchors],
        "transform_metadata": transform_metadata,
    }

    return record, diagnostics


def build_condition_datasets(
    candidates: list[HostCandidate],
    conditions: list[str],
    split_sizes: dict[str, int],
    seed_manager: SeedManager,
    max_condition_attempts: int = 12,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, int],
    dict[str, nx.Graph],
]:
    needed = sum(split_sizes.values())
    accepted_bundles: list[
        tuple[HostCandidate, dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]
    ] = []
    host_diagnostics: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
    accepted_isomorphism_buckets: dict[tuple[Any, ...], list[nx.Graph]] = {}
    construction_audit = {
        "candidates_considered": 0,
        "duplicate_topology_candidates_rejected": 0,
        "condition_bundle_failures": 0,
    }

    for host in candidates:
        construction_audit["candidates_considered"] += 1
        if len(accepted_bundles) >= needed:
            break

        if not check_preplant_no_accidental(host.graph):
            continue

        host_fingerprint = graph_fingerprint(host.graph)
        bucket_key = structural_bucket_key(host_fingerprint)
        if any(nx.is_isomorphic(host.graph, prior) for prior in accepted_isomorphism_buckets.get(bucket_key, [])):
            construction_audit["duplicate_topology_candidates_rejected"] += 1
            continue

        anchors = select_host_anchors(host.graph, count=4)
        pair_seed = stable_seed(host.host_id, tuple(anchors), split_sizes["train"])
        bundle: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
        bundle_diagnostics: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}

        try:
            for condition in conditions:
                transformed_host = None
                transform_metadata: dict[str, Any] | None = None
                pre_edges = edge_text_set(host.graph)
                for attempt in range(max_condition_attempts):
                    condition_seed = seed_manager.next_seed()
                    try:
                        candidate_host = apply_condition(
                            host.graph, condition=condition, anchor=anchors[0], seed=condition_seed
                        )
                    except RuntimeError:
                        continue
                    post_edges = edge_text_set(candidate_host)
                    changed = pre_edges != post_edges or host.graph.number_of_nodes() != candidate_host.number_of_nodes()
                    isomorphic_to_pre = nx.is_isomorphic(host.graph, candidate_host)
                    valid_change = (
                        True if condition == "clean"
                        else changed if condition == "decoy"
                        else changed and not isomorphic_to_pre
                    )
                    if valid_change and check_preplant_no_accidental(candidate_host):
                        transformed_host = candidate_host
                        transform_metadata = {
                            "attempts_used": int(attempt + 1),
                            "seed": int(condition_seed),
                            "changed": bool(changed),
                            "isomorphic_to_pre": bool(isomorphic_to_pre),
                            "pre_fingerprint": graph_fingerprint(host.graph),
                            "post_fingerprint": graph_fingerprint(candidate_host),
                            "edges_added": [list(edge) for edge in sorted(post_edges - pre_edges)],
                            "edges_removed": [list(edge) for edge in sorted(pre_edges - post_edges)],
                            "edge_symmetric_difference_count": int(len(pre_edges ^ post_edges)),
                        }
                        break
                if transformed_host is None or transform_metadata is None:
                    raise RuntimeError(f"{host.host_id} condition {condition}: unable to repair accidental matches.")

                pair_id = stable_seed(host.host_id, condition, pair_seed)
                for label_name in ("positive", "negative"):
                    record, diagnostics = build_planted_pair_record(
                        condition=condition,
                        host=host,
                        transformed_host=transformed_host,
                        anchors=anchors,
                        label_name=label_name,
                        spacer_length=BASE_SPACER_LENGTH,
                        split="",
                        pair_id=pair_id,
                        condition_seed=condition_seed,
                        transform_metadata=transform_metadata,
                    )
                    bundle[condition].append(record)
                    bundle_diagnostics[condition].append(diagnostics)
        except RuntimeError:
            construction_audit["condition_bundle_failures"] += 1
            continue

        # Provenance travels atomically with its accepted host and graph bundle.
        accepted_bundles.append((host, bundle, bundle_diagnostics))
        accepted_isomorphism_buckets.setdefault(bucket_key, []).append(host.graph)

    if len(accepted_bundles) < needed:
        raise RuntimeError(
            f"Could not collect {needed} repairable hosts under all conditions; "
            f"collected only {len(accepted_bundles)}."
        )

    split_slices = {}
    cursor = 0
    for split_name in BLOCK_SPLITS:
        split_slices[split_name] = (cursor, cursor + split_sizes[split_name])
        cursor += split_sizes[split_name]

    dataset: dict[str, dict[str, list[dict[str, Any]]]] = {
        condition: {split: [] for split in BLOCK_SPLITS} for condition in conditions
    }
    host_manifest: dict[str, list[dict[str, Any]]] = {split: [] for split in BLOCK_SPLITS}

    for split_name in BLOCK_SPLITS:
        start, end = split_slices[split_name]
        for position, (host, bundle, diagnostics_bundle) in enumerate(accepted_bundles[start:end], start=start):
            host_manifest[split_name].append({
                "split": split_name,
                "position": int(position),
                "host_id": host.host_id,
                "dataset_index": int(host.dataset_index),
                "structural_fingerprint": graph_fingerprint(host.graph),
            })
            for condition in conditions:
                records = bundle[condition]
                for record in records:
                    record["split"] = split_name
                    record["position"] = int(position)
                    record["graph_id"] = (
                        f"{condition}_{split_name}_{host.host_id}_{record['label_name']}"
                    )
                diagnostics = diagnostics_bundle[condition]
                for diagnostic in diagnostics:
                    diagnostic["split"] = split_name
                    diagnostic["position"] = int(position)
                dataset[condition][split_name].extend(records)
                host_diagnostics[condition].extend(diagnostics)

    for condition in conditions:
        counts = [len(dataset[condition][split]) for split in BLOCK_SPLITS]
        if any(count != 2 * split_sizes[split] for split, count in zip(BLOCK_SPLITS, counts)):
            raise RuntimeError(f"{condition} split imbalance: {counts}.")

    construction_audit["accepted_unique_repairable_hosts"] = len(accepted_bundles)
    accepted_host_graphs = {host.host_id: host.graph for host, _, _ in accepted_bundles}
    return dataset, host_diagnostics, host_manifest, construction_audit, accepted_host_graphs


def ordered_unique_host_ids(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in records:
        host_id = str(record["host_id"])
        if host_id not in seen:
            seen.add(host_id)
            ordered.append(host_id)
    return ordered


def validate_clean_protocol(
    condition_datasets: dict[str, dict[str, list[dict[str, Any]]]],
    host_diagnostics: dict[str, list[dict[str, Any]]],
    host_manifest: dict[str, list[dict[str, Any]]],
    conditions: list[str],
    hosts_per_split: int,
    accepted_host_graphs: dict[str, nx.Graph],
    historical_viewed_host_ids: set[str] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    block_ids = {
        split: [str(row["host_id"]) for row in host_manifest[split]]
        for split in BLOCK_SPLITS
    }
    block_indices = {
        split: [int(row["dataset_index"]) for row in host_manifest[split]]
        for split in BLOCK_SPLITS
    }
    for split in BLOCK_SPLITS:
        if len(block_ids[split]) != hosts_per_split:
            failures.append(f"{split}: expected {hosts_per_split} hosts, got {len(block_ids[split])}")
        if len(set(block_ids[split])) != hosts_per_split:
            failures.append(f"{split}: host IDs are not unique")
        if len(set(block_indices[split])) != hosts_per_split:
            failures.append(f"{split}: dataset indices are not unique")

    intersections: dict[str, int] = {}
    for left_index, left in enumerate(BLOCK_SPLITS):
        for right in BLOCK_SPLITS[left_index + 1:]:
            count = len(set(block_ids[left]) & set(block_ids[right]))
            intersections[f"{left}__{right}"] = count
            if count:
                failures.append(f"{left}/{right}: {count} overlapping host IDs")

    topology_buckets: dict[tuple[Any, ...], list[tuple[str, str]]] = {}
    for split in BLOCK_SPLITS:
        for row in host_manifest[split]:
            if "structural_fingerprint" not in row:
                failures.append(f"{split}/{row['host_id']}: missing structural fingerprint")
                continue
            topology_buckets.setdefault(
                structural_bucket_key(row["structural_fingerprint"]), []
            ).append((split, str(row["host_id"])))
    exact_isomorphism_comparisons = 0
    duplicate_topologies: list[dict[str, str]] = []
    for bucket in topology_buckets.values():
        for left_index, (left_split, left_id) in enumerate(bucket):
            for right_split, right_id in bucket[left_index + 1:]:
                exact_isomorphism_comparisons += 1
                if nx.is_isomorphic(accepted_host_graphs[left_id], accepted_host_graphs[right_id]):
                    duplicate_topologies.append({
                        "left_split": left_split,
                        "left_host_id": left_id,
                        "right_split": right_split,
                        "right_host_id": right_id,
                    })
    if duplicate_topologies:
        failures.append(f"Found {len(duplicate_topologies)} duplicate host topologies")

    historical_intersection = 0
    if historical_viewed_host_ids is not None:
        historical_intersection = len(set(block_ids["test"]) & historical_viewed_host_ids)
        if historical_intersection:
            failures.append(
                f"New final test overlaps {historical_intersection} historically viewed test hosts"
            )

    ordered_condition_alignment: dict[str, dict[str, bool]] = {}
    for split in SELECTED_SPLITS:
        expected = block_ids[split]
        ordered_condition_alignment[split] = {}
        for condition in conditions:
            observed = ordered_unique_host_ids(condition_datasets[condition][split])
            aligned = observed == expected
            ordered_condition_alignment[split][condition] = aligned
            if not aligned:
                failures.append(f"{condition}/{split}: ordered host IDs do not match manifest")

    pair_checks = 0
    for condition in conditions:
        for split in BLOCK_SPLITS:
            by_host: dict[str, list[dict[str, Any]]] = {}
            for record in condition_datasets[condition][split]:
                by_host.setdefault(str(record["host_id"]), []).append(record)
                if "dataset_index" not in record:
                    failures.append(f"{condition}/{split}: record missing dataset_index")
                if int(record["accidental_match_count"]) != 0:
                    failures.append(f"{record['graph_id']}: accidental template match")
                if int(record["num_matches_designated_roots"]) != 4:
                    failures.append(f"{record['graph_id']}: designated root count is not four")
            for host_id, records in by_host.items():
                pair_checks += 1
                labels = sorted(int(record["label"]) for record in records)
                if labels != [0, 1]:
                    failures.append(f"{condition}/{split}/{host_id}: labels are {labels}, expected [0, 1]")

    diagnostic_count = 0
    for condition, diagnostics in host_diagnostics.items():
        diagnostic_count += len(diagnostics)
        for diagnostic in diagnostics:
            if "dataset_index" not in diagnostic:
                failures.append(f"{condition}: diagnostic missing dataset_index")
            if int(diagnostic["accidental_match_count"]) != 0:
                failures.append(f"{condition}/{diagnostic['host_id']}: diagnostic accidental match")
            if int(diagnostic["designated_root_matches"]) != 4:
                failures.append(f"{condition}/{diagnostic['host_id']}: diagnostic root count is not four")
            transform = diagnostic.get("transform_metadata", {})
            if not transform:
                failures.append(f"{condition}/{diagnostic['host_id']}: missing transform metadata")
            elif condition == "perturb":
                if not transform.get("changed", False):
                    failures.append(f"perturb/{diagnostic['host_id']}: unchanged perturbation")
                if transform.get("isomorphic_to_pre", True):
                    failures.append(f"perturb/{diagnostic['host_id']}: perturbation is isomorphic to original")
                if int(transform.get("edge_symmetric_difference_count", 0)) <= 0:
                    failures.append(f"perturb/{diagnostic['host_id']}: no edge difference recorded")

    report = {
        "passed": not failures,
        "hosts_per_block": {split: len(block_ids[split]) for split in BLOCK_SPLITS},
        "unique_host_ids_per_block": {split: len(set(block_ids[split])) for split in BLOCK_SPLITS},
        "unique_dataset_indices_per_block": {split: len(set(block_indices[split])) for split in BLOCK_SPLITS},
        "pairwise_host_id_intersection_counts": intersections,
        "historical_viewed_vs_new_test_intersection_count": int(historical_intersection),
        "structural_fingerprint_count": int(sum(len(rows) for rows in topology_buckets.values())),
        "exact_isomorphism_comparisons": int(exact_isomorphism_comparisons),
        "duplicate_topology_count": int(len(duplicate_topologies)),
        "duplicate_topologies": duplicate_topologies,
        "ordered_condition_host_alignment": ordered_condition_alignment,
        "condition_host_pair_checks": int(pair_checks),
        "diagnostic_record_count": int(diagnostic_count),
        "accidental_template_matches": 0 if not any("accidental" in failure for failure in failures) else None,
        "designated_roots_per_graph": 4,
        "failures": failures,
    }
    if failures:
        raise RuntimeError("Clean protocol validation failed: " + "; ".join(failures[:20]))
    return report


def summarize_linear_readout(condition_datasets: dict[str, dict[str, list[dict[str, Any]]]], readout_seeds: list[int]) -> dict[str, Any]:
    linear_rows: list[dict[str, Any]] = []
    choquet_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for condition, dataset in condition_datasets.items():
        selected_dataset = {split: dataset[split] for split in SELECTED_SPLITS}
        linear_results = run_linear_ladder(selected_dataset, seeds=readout_seeds)
        linear_summary: dict[str, Any] = {}
        for degree in ("1", "2", "3"):
            runs = linear_results["per_degree"][degree]["seeds"]
            train_values = [float(run["train_accuracy"]) for run in runs]
            val_values = [float(run["validation_accuracy"]) for run in runs]
            test_values = [float(run["test_accuracy"]) for run in runs]
            test_uncertainty = student_t_seed_summary(test_values)
            linear_summary[degree] = {
                "train_mean": float(statistics.mean(train_values)),
                "validation_mean": float(statistics.mean(val_values)),
                "test_mean": float(statistics.mean(test_values)),
                "test_sd": test_uncertainty["sample_sd"],
                "test_ci_low": test_uncertainty["ci_95_low"],
                "test_ci_high": test_uncertainty["ci_95_high"],
                "uncertainty": test_uncertainty,
            }
            for run in runs:
                linear_rows.append(
                    {
                        "condition": condition,
                        "model": "readout_linear",
                        "variant": degree,
                        "seed": int(run["seed"]),
                        "split": "overall",
                        "train_accuracy": float(run["train_accuracy"]),
                        "validation_accuracy": float(run["validation_accuracy"]),
                        "test_accuracy": float(run["test_accuracy"]),
                    }
                )
                weights = np.asarray(run["weights"], dtype=float)
                for record in selected_dataset["test"]:
                    score = float(np.dot(weights, np.asarray(record["features"][degree], dtype=float)))
                    probability = float(1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, score)))))
                    prediction_rows.append({
                        "condition": condition,
                        "model": "readout_linear",
                        "variant": degree,
                        "seed": int(run["seed"]),
                        "graph_id": record["graph_id"],
                        "host_id": record["host_id"],
                        "pair_id": int(record["pair_id"]),
                        "label": int(record["label"]),
                        "score": score,
                        "probability": probability,
                        "prediction": int(score >= 0.0),
                    })

        choquet_results = run_shared_choquet_experiment(selected_dataset)
        choquet_summary: dict[str, Any] = {}
        for order in ("1_additive", "2_additive", "3_additive"):
            rep = choquet_results[order]["representative_result"]
            choquet_summary[order] = {
                "train_accuracy": float(rep["train"]["accuracy"]),
                "validation_accuracy": float(rep["validation"]["accuracy"]),
                "test_accuracy": float(rep["test"]["accuracy"]),
                "max_abs_gap": float(choquet_results[order]["max_abs_positive_negative_gap"]),
            }
            choquet_rows.append(
                {
                    "condition": condition,
                    "model": "readout_choquet",
                    "order": order,
                    "train_accuracy": float(rep["train"]["accuracy"]),
                    "validation_accuracy": float(rep["validation"]["accuracy"]),
                    "test_accuracy": float(rep["test"]["accuracy"]),
                    "max_abs_gap": float(choquet_results[order]["max_abs_positive_negative_gap"]),
                }
            )
            rule = rep["threshold_rule"]
            for record, score in zip(selected_dataset["test"], rep["test"]["scores"]):
                prediction = (
                    int(score >= rule["threshold"])
                    if rule["orientation"] == "ge"
                    else int(score <= rule["threshold"])
                )
                prediction_rows.append({
                    "condition": condition,
                    "model": "readout_choquet",
                    "variant": order,
                    "seed": "",
                    "graph_id": record["graph_id"],
                    "host_id": record["host_id"],
                    "pair_id": int(record["pair_id"]),
                    "label": int(record["label"]),
                    "score": float(score),
                    "probability": "",
                    "prediction": prediction,
                })

        summaries[condition] = {
            "linear": linear_summary,
            "choquet": choquet_summary,
            "counts": {
                "overall": sum(len(split_records) for split_records in dataset.values()),
                "train": len(dataset["train"]),
                "validation": len(dataset["validation"]),
                "test": len(dataset["test"]),
            },
        }

    write_csv(RESULTS_ROOT / "readout_linear_seed_runs.csv", linear_rows, [
        "condition",
        "model",
        "variant",
        "seed",
        "split",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
    ])
    write_csv(RESULTS_ROOT / "readout_choquet_summary.csv", choquet_rows, [
        "condition",
        "model",
        "order",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "max_abs_gap",
    ])
    write_json(RESULTS_ROOT / "readout_summary.json", summaries)
    write_csv(RESULTS_ROOT / "readout_test_predictions.csv", prediction_rows, [
        "condition", "model", "variant", "seed", "graph_id", "host_id", "pair_id",
        "label", "score", "probability", "prediction",
    ])
    return summaries


def pair_summary(pair_ids: list[int], y_prob: list[float], y_true: list[int]) -> dict[str, float]:
    by_pair: dict[int, dict[str, float]] = {}
    for pair_id, probability, label in zip(pair_ids, y_prob, y_true):
        state = by_pair.setdefault(int(pair_id), {})
        if label == 1:
            state["positive"] = float(probability)
        else:
            state["negative"] = float(probability)

    margins: list[float] = []
    for values in by_pair.values():
        if "positive" in values and "negative" in values:
            margins.append(values["positive"] - values["negative"])

    if not margins:
        return {
            "paired_ranking_accuracy": 0.0,
            "paired_margin_mean": 0.0,
            "paired_margin_median": 0.0,
            "paired_margin_minimum": 0.0,
            "paired_margin_maximum": 0.0,
        }
    return {
        "paired_ranking_accuracy": float(sum(1 for margin in margins if margin > 0.0) / len(margins)),
        "paired_margin_mean": float(statistics.mean(margins)),
        "paired_margin_median": float(statistics.median(margins)),
        "paired_margin_minimum": float(min(margins)),
        "paired_margin_maximum": float(max(margins)),
    }


class MessagePassingClassifier(nn.Module):
    def __init__(
        self,
        architecture: str,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        pna_deg: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if architecture not in MODEL_ARCHITECTURES:
            raise ValueError(f"Unknown architecture {architecture}.")
        self.architecture = architecture
        self.depth = int(depth)
        self.input_dim = in_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        current = in_dim
        for _ in range(self.depth):
            if architecture == "gin":
                mlp = nn.Sequential(
                    nn.Linear(current, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                layer = GINConv(mlp)
            elif architecture == "graphsage":
                layer = SAGEConv(current, hidden_dim, aggr="mean")
            else:
                if pna_deg is None:
                    raise ValueError("PNA requires a degree histogram.")
                layer = PNAConv(
                    current,
                    hidden_dim,
                    aggregators=AGGREGATORS,
                    scalers=SCALERS,
                    deg=pna_deg,
                    towers=1,
                    pre_layers=1,
                    post_layers=1,
                )
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(layer)
            current = hidden_dim

        self.readout = nn.Sequential(
            nn.Linear(self.input_dim + self.depth * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Data) -> torch.Tensor:
        device = next(self.parameters()).device
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        batch_index = batch.batch.to(device)

        states = [x]
        for layer in self.convs:
            if self.architecture == "graphsage":
                x = layer(x, edge_index)
            elif self.architecture == "gin":
                x = layer(x, edge_index)
            else:
                x = layer(x, edge_index)
            x = self.bns[len(states) - 1](x)
            x = torch.relu(x)
            states.append(x)
        pooled = [global_mean_pool(state, batch_index) for state in states]
        logits = self.readout(torch.cat(pooled, dim=1)).squeeze(-1)
        return logits


def compute_pna_degree_histogram(training_graphs: list[Data]) -> torch.Tensor:
    """Count actual node degrees using training graphs and no other split."""
    histogram = torch.zeros(1, dtype=torch.long)
    for data in training_graphs:
        node_count = int(data.num_nodes)
        if node_count == 0:
            continue
        degrees = torch.bincount(data.edge_index[0], minlength=node_count)
        graph_histogram = torch.bincount(degrees, minlength=1)
        if graph_histogram.numel() > histogram.numel():
            histogram = F.pad(histogram, (0, graph_histogram.numel() - histogram.numel()))
        histogram[: graph_histogram.numel()] += graph_histogram.to(dtype=torch.long)
    return histogram


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses = []
    probabilities: list[float] = []
    labels: list[int] = []
    pair_ids: list[int] = []
    graph_ids: list[str] = []
    host_ids: list[str] = []
    conditions: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y.float())
            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            probabilities.extend([float(probability) for probability in probs])
            labels.extend(int(label) for label in batch.y.detach().cpu().numpy())
            pair_ids.extend(int(value) for value in batch.pair_id.detach().cpu().numpy())
            graph_ids.extend(str(value) for value in batch.graph_id)
            host_ids.extend(str(value) for value in batch.host_id)
            conditions.extend(str(value) for value in batch.condition)
    if not probabilities:
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "auc": float("nan"),
            "paired_ranking_accuracy": 0.0,
            "paired_margin_mean": 0.0,
            "paired_margin_median": 0.0,
            "paired_margin_minimum": 0.0,
            "paired_margin_maximum": 0.0,
            "predictions": [],
        }
    accuracy = binary_accuracy(labels, probabilities)
    bacc = balanced_accuracy(labels, probabilities)
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    pair_stats = pair_summary(pair_ids, probabilities, labels)
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "accuracy": accuracy,
        "balanced_accuracy": bacc,
        "auc": auc,
        "paired_ranking_accuracy": pair_stats["paired_ranking_accuracy"],
        "paired_margin_mean": pair_stats["paired_margin_mean"],
        "paired_margin_median": pair_stats["paired_margin_median"],
        "paired_margin_minimum": pair_stats["paired_margin_minimum"],
        "paired_margin_maximum": pair_stats["paired_margin_maximum"],
        "predictions": [
            {
                "graph_id": graph_id,
                "host_id": host_id,
                "condition": condition,
                "pair_id": int(pair_id),
                "label": int(label),
                "probability": float(probability),
                "prediction": int(probability >= 0.5),
            }
            for graph_id, host_id, condition, pair_id, label, probability in zip(
                graph_ids, host_ids, conditions, pair_ids, labels, probabilities
            )
        ],
    }


def train_gnn(
    architecture: str,
    config: dict[str, Any],
    dataset: dict[str, list[Data]],
    device: torch.device,
    batch_size: int,
    seed: int,
    evaluate_test: bool = False,
    pna_on_cpu: bool = False,
    pna_degree_histogram: torch.Tensor | None = None,
) -> tuple[dict[str, Any], float]:
    seed_everything(seed)
    effective_device = torch.device("cpu") if architecture == "pna" and pna_on_cpu else device
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False)
    test_loader = None
    if evaluate_test:
        if "test" not in dataset:
            raise ValueError("evaluate_test=True requires a test dataset.")
        test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)

    labels = torch.tensor([int(data.y.item()) for data in dataset["train"]], dtype=torch.float32)
    positives = float((labels == 1).sum().item())
    negatives = float((labels == 0).sum().item())
    if positives <= 0:
        pos_weight = torch.tensor(1.0, device=effective_device)
    else:
        pos_weight = torch.tensor(negatives / positives, device=effective_device)

    pna_deg = None
    if architecture == "pna":
        pna_deg = (
            pna_degree_histogram
            if pna_degree_histogram is not None
            else compute_pna_degree_histogram(dataset["train"])
        )

    model = MessagePassingClassifier(
        architecture=architecture,
        in_dim=int(dataset["train"][0].x.shape[1]),
        hidden_dim=int(config["hidden_dim"]),
        depth=int(config["depth"]),
        dropout=float(config["dropout"]),
        pna_deg=pna_deg,
    ).to(effective_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stop_counter = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(effective_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y.float())
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss in epoch {epoch}")
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_metrics = evaluate_classifier(model, val_loader, criterion, effective_device)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": float(val_metrics["loss"]),
                "validation_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "validation_accuracy": float(val_metrics["accuracy"]),
                "validation_auc": float(val_metrics["auc"]),
            }
        )

        if val_metrics["balanced_accuracy"] > best_val + 1e-8:
            best_val = float(val_metrics["balanced_accuracy"])
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= int(config["patience"]):
                break

    if best_state is None:
        raise RuntimeError(f"No best epoch found for {architecture}")

    runtime = time.perf_counter() - start
    model.load_state_dict(best_state)
    payload: dict[str, Any] = {
        "architecture": architecture,
        "seed": seed,
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stopped": len(history) < int(config["max_epochs"]),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "validation_loss_curve": history,
        "validation_balanced_accuracy": float(best_val),
        "runtime_seconds": float(runtime),
        "config": config,
        "evaluate_test": bool(evaluate_test),
        "execution_device": str(effective_device),
    }
    score = float(best_val)
    if evaluate_test:
        assert test_loader is not None
        test_metrics = evaluate_classifier(model, test_loader, criterion, effective_device)
        payload.update({
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "test_auc": float(test_metrics["auc"]),
            "test_paired_ranking_accuracy": float(test_metrics["paired_ranking_accuracy"]),
            "test_paired_margin_mean": float(test_metrics["paired_margin_mean"]),
            "test_paired_margin_median": float(test_metrics["paired_margin_median"]),
            "test_predictions": test_metrics["predictions"],
        })
        score = float(test_metrics["balanced_accuracy"])
    return payload, score


def build_condition_gnn_payload(
    condition_datasets: dict[str, dict[str, list[dict[str, Any]]]],
    split_sizes: dict[str, int],
    tune_repeats: int,
    eval_repeats: int,
    seed_manager: SeedManager,
    batch_size: int,
    device: torch.device,
    pna_on_cpu: bool = False,
) -> dict[str, Any]:
    dataset = condition_datasets["clean"]
    tuning_records = {
        split: [
            graph_to_torch_data(
                record["graph"],
                int(record["label"]),
                int(record["pair_id"]),
                str(record["graph_id"]),
                str(record["condition"]),
                str(record["host_id"]),
            )
            for record in dataset[split]
        ]
        for split in ("train", "validation")
    }
    for split in ("train", "validation"):
        if len(tuning_records[split]) != 2 * split_sizes[split]:
            raise RuntimeError(f"GNN split {split} has {len(tuning_records[split])} graphs, expected {2*split_sizes[split]}.")

    pna_degree_histogram = compute_pna_degree_histogram(tuning_records["train"])
    training_graph_ids = [str(data.graph_id) for data in tuning_records["train"]]
    training_graph_id_sha256 = hashlib.sha256(
        json.dumps(training_graph_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(RESULTS_ROOT / "pna_degree_histogram.json", {
        "source_split": "train",
        "validation_or_test_graphs_used": False,
        "histogram": [int(value) for value in pna_degree_histogram.tolist()],
        "degree_bins": list(range(int(pna_degree_histogram.numel()))),
        "training_graph_count": len(tuning_records["train"]),
        "training_node_count": int(sum(int(data.num_nodes) for data in tuning_records["train"])),
        "maximum_degree": int(pna_degree_histogram.numel() - 1),
        "ordered_training_graph_ids": training_graph_ids,
        "ordered_training_graph_ids_sha256": training_graph_id_sha256,
        "histogram_node_count_check": int(pna_degree_histogram.sum().item()),
    })

    search_space = [
        {"depth": 2, "hidden_dim": 32, "dropout": 0.0, "learning_rate": 1e-3, "weight_decay": 0.0, "max_epochs": 160, "patience": 18},
        {"depth": 3, "hidden_dim": 32, "dropout": 0.1, "learning_rate": 1e-3, "weight_decay": 1e-4, "max_epochs": 180, "patience": 20},
        {"depth": 4, "hidden_dim": 48, "dropout": 0.2, "learning_rate": 3e-4, "weight_decay": 1e-4, "max_epochs": 200, "patience": 25},
        {"depth": 5, "hidden_dim": 64, "dropout": 0.3, "learning_rate": 3e-4, "weight_decay": 2e-4, "max_epochs": 220, "patience": 25},
        {"depth": 6, "hidden_dim": 64, "dropout": 0.35, "learning_rate": 2e-4, "weight_decay": 2e-4, "max_epochs": 240, "patience": 28},
    ]

    tuning_rows: list[dict[str, Any]] = []
    final_seed_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    selected_configurations: dict[str, Any] = {}
    tuning_seed_schedules = {
        architecture: seed_manager.spawn_ints(tune_repeats)
        for architecture in MODEL_ARCHITECTURES
    }

    LOGGER.info("GNN tuning started: only train and validation graphs have been converted")
    for architecture in MODEL_ARCHITECTURES:
        best_config: dict[str, Any] | None = None
        best_score = -1.0
        configuration_scores: list[dict[str, Any]] = []
        for config in search_space:
            tune_scores: list[float] = []
            for seed in tuning_seed_schedules[architecture]:
                payload, score = train_gnn(
                    architecture=architecture,
                    config=config,
                    dataset=tuning_records,
                    device=device,
                    batch_size=batch_size,
                    seed=seed,
                    evaluate_test=False,
                    pna_on_cpu=pna_on_cpu,
                    pna_degree_histogram=pna_degree_histogram,
                )
                tune_scores.append(score)
                tuning_rows.append({
                    "architecture": architecture,
                    "seed": int(seed),
                    **{k: float(v) for k, v in config.items()},
                    "validation_balanced_accuracy": float(payload["validation_balanced_accuracy"]),
                    "runtime_seconds": float(payload["runtime_seconds"]),
                    "selected_epoch": int(payload["selected_epoch"]),
                    "epochs_run": int(payload["epochs_run"]),
                    "early_stopped": bool(payload["early_stopped"]),
                    "trainable_parameters": int(payload["trainable_parameters"]),
                    "execution_device": str(payload["execution_device"]),
                })
            mean_score = float(statistics.mean(tune_scores)) if tune_scores else 0.0
            configuration_scores.append({
                "config": dict(config),
                "seed_scores": tune_scores,
                "mean_validation_balanced_accuracy": mean_score,
            })
            if mean_score > best_score:
                best_score = mean_score
                best_config = dict(config)

        if best_config is None:
            raise RuntimeError(f"No best config discovered for {architecture}.")
        tied = [
            entry for entry in configuration_scores
            if abs(float(entry["mean_validation_balanced_accuracy"]) - best_score) <= 1e-12
        ]
        best_config = dict(tied[0]["config"])
        selected_configurations[architecture] = {
            "config": best_config,
            "selection_metric": "mean validation balanced accuracy",
            "mean_validation_balanced_accuracy": float(best_score),
            "tuning_seed_count_per_configuration": int(tune_repeats),
            "execution_device": "cpu" if architecture == "pna" and pna_on_cpu else str(device),
            "matched_tuning_seeds": tuning_seed_schedules[architecture],
            "tie_break_rule": "first maximum in prespecified search-space order",
            "tied_maximum_count": len(tied),
            "tied_maximum_configurations": [entry["config"] for entry in tied],
            "all_configuration_scores": configuration_scores,
        }
        LOGGER.info("Frozen %s configuration at mean validation BA %.6f", architecture, best_score)

    write_json(RESULTS_ROOT / "selected_gnn_configurations.json", {
        "status": "frozen_before_fresh_test_conversion",
        "architectures": selected_configurations,
    })
    LOGGER.info("All GNN configurations frozen; converting fresh-test graphs now")
    fresh_test_records = [
        graph_to_torch_data(
            record["graph"],
            int(record["label"]),
            int(record["pair_id"]),
            str(record["graph_id"]),
            str(record["condition"]),
            str(record["host_id"]),
        )
        for record in dataset["test"]
    ]
    if len(fresh_test_records) != 2 * split_sizes["test"]:
        raise RuntimeError(
            f"GNN fresh test has {len(fresh_test_records)} graphs; expected {2 * split_sizes['test']}."
        )
    evaluation_records = {**tuning_records, "test": fresh_test_records}
    evaluation_seeds = seed_manager.spawn_ints(eval_repeats)
    final_prediction_rows: list[dict[str, Any]] = []

    for architecture in MODEL_ARCHITECTURES:
        best_config = dict(selected_configurations[architecture]["config"])
        final_scores: list[float] = []
        for seed in evaluation_seeds:
            payload, score = train_gnn(
                architecture=architecture,
                config=best_config,
                dataset=evaluation_records,
                device=device,
                batch_size=batch_size,
                seed=seed,
                evaluate_test=True,
                pna_on_cpu=pna_on_cpu,
                pna_degree_histogram=pna_degree_histogram,
            )
            final_scores.append(score)
            final_seed_rows.append({
                "architecture": architecture,
                "depth": int(best_config["depth"]),
                "hidden_dim": int(best_config["hidden_dim"]),
                "dropout": float(best_config["dropout"]),
                "learning_rate": float(best_config["learning_rate"]),
                "weight_decay": float(best_config["weight_decay"]),
                "seed": int(seed),
                "selected_epoch": int(payload["selected_epoch"]),
                "epochs_run": int(payload["epochs_run"]),
                "early_stopped": bool(payload["early_stopped"]),
                "runtime_seconds": float(payload["runtime_seconds"]),
                "trainable_parameters": int(payload["trainable_parameters"]),
                "execution_device": str(payload["execution_device"]),
                "test_accuracy": float(payload["test_accuracy"]),
                "test_balanced_accuracy": float(payload["test_balanced_accuracy"]),
                "test_auc": float(payload["test_auc"]),
                "test_paired_ranking_accuracy": float(payload["test_paired_ranking_accuracy"]),
                "test_paired_margin_mean": float(payload["test_paired_margin_mean"]),
                "test_paired_margin_median": float(payload["test_paired_margin_median"]),
            })
            for prediction in payload["test_predictions"]:
                final_prediction_rows.append({
                    "architecture": architecture,
                    "seed": int(seed),
                    "execution_device": str(payload["execution_device"]),
                    **prediction,
                })

        uncertainty = student_t_seed_summary(final_scores)
        summary[architecture] = {
            "best_config": best_config,
            "tune_repeats": tune_repeats,
            "eval_repeats": eval_repeats,
            "test_seed_balanced_accuracies": final_scores,
            "test_mean": uncertainty["mean"],
            "test_sd": uncertainty["sample_sd"],
            "test_ci_low": uncertainty["ci_95_low"],
            "test_ci_high": uncertainty["ci_95_high"],
            "uncertainty": uncertainty,
            "matched_evaluation_seeds": evaluation_seeds,
            "seed_stability_gate": {
                "sd": uncertainty["sample_sd"],
                "limit": float(SEED_STABILITY_SD_LIMIT),
                "passes": uncertainty["sample_sd"] <= SEED_STABILITY_SD_LIMIT,
            },
        }

    write_csv(RESULTS_ROOT / "gnn_tuning_grid.csv", tuning_rows, [
        "architecture",
        "seed",
        "depth",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "patience",
        "validation_balanced_accuracy",
        "runtime_seconds",
        "selected_epoch",
        "epochs_run",
        "early_stopped",
        "trainable_parameters",
        "execution_device",
    ])
    write_csv(RESULTS_ROOT / "gnn_seed_runs.csv", final_seed_rows, [
        "architecture",
        "depth",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "seed",
        "selected_epoch",
        "epochs_run",
        "early_stopped",
        "runtime_seconds",
        "trainable_parameters",
        "execution_device",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_auc",
        "test_paired_ranking_accuracy",
        "test_paired_margin_mean",
        "test_paired_margin_median",
    ])
    write_csv(RESULTS_ROOT / "gnn_test_predictions.csv", final_prediction_rows, [
        "architecture",
        "seed",
        "execution_device",
        "graph_id",
        "host_id",
        "condition",
        "pair_id",
        "label",
        "probability",
        "prediction",
    ])
    write_json(RESULTS_ROOT / "gnn_summary.json", summary)
    return summary


def write_order_depth_table(readout_summary: dict[str, Any], gnn_summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for condition, payload in readout_summary.items():
        for degree in ("1", "2", "3"):
            entry = payload["linear"][degree]
            rows.append(
                {
                    "regime": condition,
                    "component": "readout_linear",
                    "variant": f"degree_{degree}",
                    "train_accuracy": entry["train_mean"],
                    "validation_accuracy": entry["validation_mean"],
                    "test_accuracy": entry["test_mean"],
                    "test_sd": entry["test_sd"],
                    "test_ci_low": entry["test_ci_low"],
                    "test_ci_high": entry["test_ci_high"],
                }
            )
        for order in ("1_additive", "2_additive", "3_additive"):
            entry = payload["choquet"][order]
            rows.append(
                {
                    "regime": condition,
                    "component": "readout_choquet",
                    "variant": order,
                    "train_accuracy": entry["train_accuracy"],
                    "validation_accuracy": entry["validation_accuracy"],
                    "test_accuracy": entry["test_accuracy"],
                    "test_sd": 0.0,
                    "test_ci_low": entry["test_accuracy"],
                    "test_ci_high": entry["test_accuracy"],
                }
            )

    for architecture, payload in gnn_summary.items():
        rows.append(
            {
                "regime": "clean",
                "component": architecture,
                "variant": f"depth_{payload['best_config']['depth']}",
                "train_accuracy": float("nan"),
                "validation_accuracy": float("nan"),
                "test_balanced_accuracy": payload["test_mean"],
                "test_sd": payload["test_sd"],
                "test_ci_low": payload["test_ci_low"],
                "test_ci_high": payload["test_ci_high"],
            }
        )

    write_csv(RESULTS_ROOT / "ablation_table.csv", rows, [
        "regime",
        "component",
        "variant",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_sd",
        "test_ci_low",
        "test_ci_high",
    ])


def write_environment_manifest(
    dataset_name: str,
    dataset_root: Path,
    host_count: int,
    split_sizes: dict[str, int],
    dataset_info: dict[str, Any],
) -> None:
    try:
        driver_version = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        driver_version = "unavailable"
    gpu_devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_devices.append({
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
        })
    manifest = {
        "dataset": {
            "name": dataset_name,
            "root": str(dataset_root),
            "host_count": int(host_count),
            "split_sizes": split_sizes,
            "dataset_info": dataset_info,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_geometric": package_version("torch_geometric"),
            "numpy": package_version("numpy"),
            "networkx": package_version("networkx"),
            "scikit_learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
            "torch_scatter": package_version("torch-scatter"),
        },
        "hardware": {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "hostname": platform.node(),
            "cuda_runtime_reported_by_torch": torch.version.cuda,
            "nvidia_driver": driver_version,
            "gpu_devices": gpu_devices,
            "architecture_devices": {
                "gin": "cuda" if torch.cuda.is_available() else "cpu",
                "graphsage": "cuda" if torch.cuda.is_available() else "cpu",
                "pna": "cpu" if dataset_info.get("pna_on_cpu") else ("cuda" if torch.cuda.is_available() else "cpu"),
            },
            "pna_cpu_reason": (
                "Installed torch_scatter lacks CUDA support; explicit user-authorized CPU assignment."
                if dataset_info.get("pna_on_cpu") else None
            ),
        },
    }
    write_json(RESULTS_ROOT / "environment_manifest.json", manifest)


def write_findings(
    readout_summary: dict[str, Any],
    gnn_summary: dict[str, Any],
    output: Path,
) -> None:
    lines = [
        "# WP2 findings",
        "",
        "Primary question: whether fixed-carrier readout-order ablation shows the theory's collapse-then-jump "
        "prediction on natural real-host graphs.",
        "",
        f"Host protocol: {DEFAULT_DATASET} real hosts, paired positive/negative planting, fixed across "
        f"train/validation/test (400/400/400).",
        "",
        "## Readout ablation",
        "",
    ]

    for condition, payload in readout_summary.items():
        deg1 = payload["linear"]["1"]["test_mean"]
        deg2 = payload["linear"]["2"]["test_mean"]
        deg3 = payload["linear"]["3"]["test_mean"]
        choquet1 = payload["choquet"]["1_additive"]["test_accuracy"]
        choquet2 = payload["choquet"]["2_additive"]["test_accuracy"]
        choquet3 = payload["choquet"]["3_additive"]["test_accuracy"]
        jump = (deg3 > deg2 and deg3 >= deg2)
        lines.append(
            f"- {condition}: degree-1={deg1:.3f}, degree-2={deg2:.3f}, degree-3={deg3:.3f}; "
            f"Choquet(≤1/≤2/3)={choquet1:.3f}/{choquet2:.3f}/{choquet3:.3f}; "
            f"collapse-then-jump indicator={str(jump).lower()}."
        )

    lines.extend(
        [
            "",
            "## Tuned GNN baselines (clean condition only)",
            "",
        ]
    )
    for architecture, payload in gnn_summary.items():
        lines.append(
            f"- {architecture}: mean test BA={payload['test_mean']:.3f}, SD={payload['test_sd']:.3f}, "
            f"95% CI=[{payload['test_ci_low']:.3f}, {payload['test_ci_high']:.3f}], "
            f"best config={payload['best_config']}, seed-stability gate "
            f"{'passed' if payload['seed_stability_gate']['passes'] else 'failed'}."
        )

    lines.extend(
        [
            "",
            "## Gates and limitations",
            "",
            "- Seed-stability gate uses balanced-accuracy SD over the ≥10 evaluation seeds.",
            "- Model tuning was done on validation only; test metrics are post-selection reports.",
            "- If any readout condition fails the collapse-then-jump check, that is reported directly as a practical bound.",
            "- Results are conditional on the prespecified protocol and this real-host distribution.",
            "",
        ]
    )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")




def write_summary(
    dataset_name: str,
    split_sizes: dict[str, int],
    dataset_info: dict[str, Any],
    readout_summary: dict[str, Any],
    gnn_summary: dict[str, Any],
    host_manifest: dict[str, Any],
) -> None:
    payload = {
        "dataset": {
            "name": dataset_name,
            "split_sizes": split_sizes,
            "dataset_info": dataset_info,
        },
        "splits": split_sizes,
        "readout": readout_summary,
        "gnn": gnn_summary,
        "host_manifest": host_manifest,
        "seed_stability_gate": {
            architecture: payload["seed_stability_gate"]
            for architecture, payload in gnn_summary.items()
        },
    }
    write_json(RESULTS_ROOT / "summary.json", payload)


def prepare_results_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Refusing to mix clean output with nonempty results directory: {path}. "
                "Use --overwrite-results only for an intentional clean replacement."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def configure_logging(results_root: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(results_root / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def write_dataset_manifests(
    args: argparse.Namespace,
    dataset_root: Path,
    device: torch.device,
    split_sizes: dict[str, int],
    condition_datasets: dict[str, dict[str, list[dict[str, Any]]]],
    host_diagnostics: dict[str, list[dict[str, Any]]],
    host_manifest: dict[str, list[dict[str, Any]]],
    candidate_info: dict[str, int],
    validation_report: dict[str, Any],
    conditions: list[str],
    construction_audit: dict[str, int],
) -> dict[str, Any]:
    total_required = sum(split_sizes.values())
    manifest_payload = {
        "protocol": "fresh_holdout",
        "block_order": list(BLOCK_SPLITS),
        "position_ranges": {
            split: [
                int(host_manifest[split][0]["position"]),
                int(host_manifest[split][-1]["position"] + 1),
            ]
            for split in BLOCK_SPLITS
        },
        "blocks": host_manifest,
    }
    write_json(RESULTS_ROOT / "host_split_manifest.json", manifest_payload)
    host_rows = []
    for split in BLOCK_SPLITS:
        for row in host_manifest[split]:
            fingerprint = row["structural_fingerprint"]
            host_rows.append({
                "split": row["split"],
                "position": row["position"],
                "host_id": row["host_id"],
                "dataset_index": row["dataset_index"],
                "nodes": fingerprint["nodes"],
                "edges": fingerprint["edges"],
                "weisfeiler_lehman_hash": fingerprint["weisfeiler_lehman_hash"],
                "labeled_edge_sha256": fingerprint["labeled_edge_sha256"],
            })
    write_csv(
        RESULTS_ROOT / "host_split_manifest.csv",
        host_rows,
        [
            "split", "position", "host_id", "dataset_index", "nodes", "edges",
            "weisfeiler_lehman_hash", "labeled_edge_sha256",
        ],
    )
    write_json(RESULTS_ROOT / "host_diagnostics.json", host_diagnostics)

    split_manifest = {
        "block_order": list(BLOCK_SPLITS),
        "blocks": {
            split: {
                "position_start": int(host_manifest[split][0]["position"]),
                "position_stop": int(host_manifest[split][-1]["position"] + 1),
                "host_count": len(host_manifest[split]),
                "condition_graph_counts": {
                    condition: len(condition_datasets[condition][split]) for condition in conditions
                },
                "evaluation_status": "never_evaluate" if split.startswith("discarded_") else "selected",
            }
            for split in BLOCK_SPLITS
        },
        "validation": validation_report,
        "construction_audit": construction_audit,
    }
    write_json(RESULTS_ROOT / "split_manifest.json", split_manifest)
    write_json(RESULTS_ROOT / "seed_manifest.json", {
        "root_seed": int(args.root_seed),
        "dataset_shuffle_child_seed": DATASET_SHUFFLE_SEED,
        "pna_on_cpu": bool(args.pna_on_cpu),
        "dataset_shuffle_seed_verified_from_root": bool(
            args.root_seed == ROOT_SEED and DATASET_SHUFFLE_SEED == 3134173889
        ),
        "gnn_tune_seeds_per_configuration": int(args.gnn_tune_seeds),
        "gnn_evaluation_seeds_per_architecture": int(args.gnn_eval_seeds),
        "dataset": args.dataset,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "max_transform_attempts": int(args.max_transform_attempts),
        "pna_on_cpu": bool(args.pna_on_cpu),
        "exact_invocation": list(args.exact_invocation),
    })
    dataset_info = {
        "requested_total_hosts": int(total_required),
        "selected_hosts": {split: len(host_manifest[split]) for split in BLOCK_SPLITS},
        "conditions": conditions,
        "spacer_length": BASE_SPACER_LENGTH,
        "condition_attempts": int(args.max_transform_attempts),
        "candidate_collection": candidate_info,
        "construction_audit": construction_audit,
        "seed_root": int(args.root_seed),
        "dataset_shuffle_child_seed": DATASET_SHUFFLE_SEED,
        "pna_on_cpu": bool(args.pna_on_cpu),
    }
    write_environment_manifest(args.dataset, dataset_root, total_required, split_sizes, dataset_info)
    dataset_files = []
    dataset_path = dataset_root / args.dataset
    for path in sorted(dataset_path.rglob("*")):
        if path.is_file():
            dataset_files.append({
                "path": str(path.relative_to(dataset_root)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            })
    write_json(RESULTS_ROOT / "dataset_cache_manifest.json", {
        "dataset_root": str(dataset_root),
        "dataset": args.dataset,
        "files": dataset_files,
    })
    write_json(RESULTS_ROOT / "invocation.json", {
        "argv": list(args.exact_invocation),
        "working_directory": str(Path.cwd()),
        "environment_fallbacks": {
            "WP2_DATASET_ROOT": os.environ.get("WP2_DATASET_ROOT"),
            "WP2_RESULTS_ROOT": os.environ.get("WP2_RESULTS_ROOT"),
        },
    })
    return {"split_manifest": split_manifest, "dataset_info": dataset_info}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WP2: real-host readout-order ablation and tuned GNN baselines."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="TU dataset name for real hosts.")
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    parser.add_argument("--hosts-per-split", type=int, default=DEFAULT_HOSTS_PER_SPLIT)
    parser.add_argument("--gnn-tune-seeds", type=int, default=3)
    parser.add_argument("--gnn-eval-seeds", type=int, default=SEED_STABILITY_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-transform-attempts", type=int, default=12)
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("WP2_DATASET_ROOT", str(WORKSPACE_ROOT / ".cache" / "wp2_dataset")),
    )
    parser.add_argument(
        "--results-root",
        default=os.environ.get("WP2_RESULTS_ROOT", str(WP2_ROOT / "results_clean")),
    )
    parser.add_argument("--mode", choices=["full", "dataset-only"], default="full")
    parser.add_argument("--fresh-holdout", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument(
        "--pna-on-cpu",
        action="store_true",
        help="Run PNA on CPU while other architectures use --device.",
    )
    parser.add_argument(
        "--historical-manifest",
        default=str(DEFAULT_HISTORICAL_MANIFEST),
        help="Prior viewed-host manifest whose test IDs must be excluded from the revised final test.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global RESULTS_ROOT
    args = parse_args(argv)
    effective_arguments = list(argv) if argv is not None else list(sys.argv[1:])
    args.exact_invocation = [sys.executable, str(Path(__file__).resolve()), *effective_arguments]
    if args.hosts_per_split != 400:
        raise ValueError("The clean fresh-holdout protocol requires exactly 400 hosts per block.")
    if args.gnn_eval_seeds < 10:
        raise ValueError("WP2 seed-stability gate requires ≥10 evaluation seeds.")
    if not args.fresh_holdout:
        raise ValueError("The clean protocol requires --fresh-holdout.")

    split_sizes = {split: args.hosts_per_split for split in BLOCK_SPLITS}
    total_required = sum(split_sizes.values())
    cushion = max(200, int(total_required * 0.5))
    candidate_request = total_required * 2 + cushion
    dataset_root = Path(args.dataset_root)
    historical_manifest_path = Path(args.historical_manifest).resolve()
    if not historical_manifest_path.exists():
        raise RuntimeError(f"Historical viewed-host manifest not found: {historical_manifest_path}")
    historical_manifest = json.loads(historical_manifest_path.read_text(encoding="utf-8"))
    historical_viewed_host_ids = {
        str(row["host_id"])
        for split in ("discarded_legacy_test", "test")
        for row in historical_manifest["blocks"][split]
    }
    RESULTS_ROOT = Path(args.results_root).resolve()
    prepare_results_root(RESULTS_ROOT, overwrite=bool(args.overwrite_results))
    configure_logging(RESULTS_ROOT)
    seed_manager = SeedManager(args.root_seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if args.mode == "full" and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Full run requested CUDA, but torch.cuda.is_available() is false.")
    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except (AttributeError, RuntimeError):
        pass

    LOGGER.info("Phase 1: loading and filtering real hosts")
    dataset_shuffle_seed = seed_manager.next_seed()
    if args.root_seed == ROOT_SEED and dataset_shuffle_seed != DATASET_SHUFFLE_SEED:
        raise RuntimeError(
            f"Dataset shuffle child seed mismatch: got {dataset_shuffle_seed}, expected {DATASET_SHUFFLE_SEED}."
        )
    host_candidates, candidate_info = build_real_host_candidates(
        dataset_name=args.dataset,
        requested_target=candidate_request,
        minimum_required=total_required,
        seed=dataset_shuffle_seed,
        dataset_root=dataset_root,
    )

    conditions = ["clean", "decoy", "perturb"]
    LOGGER.info("Phase 2: building all %d accepted host bundles", total_required)
    (
        condition_datasets,
        host_diagnostics,
        host_manifest,
        construction_audit,
        accepted_host_graphs,
    ) = build_condition_datasets(
        candidates=host_candidates,
        conditions=conditions,
        split_sizes=split_sizes,
        seed_manager=seed_manager,
        max_condition_attempts=args.max_transform_attempts,
    )
    LOGGER.info("Phase 3: validating provenance, split isolation, and diagnostics")
    validation_report = validate_clean_protocol(
        condition_datasets=condition_datasets,
        host_diagnostics=host_diagnostics,
        host_manifest=host_manifest,
        conditions=conditions,
        hosts_per_split=args.hosts_per_split,
        accepted_host_graphs=accepted_host_graphs,
        historical_viewed_host_ids=historical_viewed_host_ids,
    )
    manifest_state = write_dataset_manifests(
        args=args,
        dataset_root=dataset_root,
        device=device,
        split_sizes=split_sizes,
        condition_datasets=condition_datasets,
        host_diagnostics=host_diagnostics,
        host_manifest=host_manifest,
        candidate_info=candidate_info,
        validation_report=validation_report,
        conditions=conditions,
        construction_audit=construction_audit,
    )
    LOGGER.info("Dataset validation passed with block counts %s", validation_report["hosts_per_block"])

    if args.mode == "dataset-only":
        write_json(RESULTS_ROOT / "completion.json", {
            "status": "dataset-only-completed",
            "dataset_only_passed": True,
            "validation": validation_report,
        })
        LOGGER.info("Dataset-only mode complete; stopped before model fitting")
        return

    if device.type != "cuda":
        raise RuntimeError(f"Full GPU experiment requires CUDA; resolved device is {device}.")

    readout_seeds = seed_manager.spawn_ints(10)
    LOGGER.info("Phase 4: tuning and freezing all GNN architectures on train/validation only")
    gnn_summary = build_condition_gnn_payload(
        condition_datasets=condition_datasets,
        split_sizes=split_sizes,
        tune_repeats=args.gnn_tune_seeds,
        eval_repeats=args.gnn_eval_seeds,
        seed_manager=seed_manager,
        batch_size=args.batch_size,
        device=device,
        pna_on_cpu=bool(args.pna_on_cpu),
    )

    LOGGER.info("Phase 5: evaluating the fixed-carrier readout ladder on fresh test")
    readout_summary = summarize_linear_readout(condition_datasets, readout_seeds)
    write_order_depth_table(readout_summary, gnn_summary)

    write_findings(readout_summary, gnn_summary, WP2_ROOT / "WP2_FINDINGS.md")

    write_summary(
        dataset_name=args.dataset,
        split_sizes=split_sizes,
        dataset_info=manifest_state["dataset_info"],
        readout_summary=readout_summary,
        gnn_summary=gnn_summary,
        host_manifest=manifest_state["split_manifest"],
    )

    write_json(RESULTS_ROOT / "completion.json", {
        "status": "completed",
        "dataset": args.dataset,
        "readout_conditions": list(readout_summary.keys()),
        "gnn_models": list(gnn_summary.keys()),
        "dataset_only_passed": True,
        "validation": validation_report,
        "historical_manifest": str(historical_manifest_path),
        "historical_viewed_host_count": len(historical_viewed_host_ids),
    })
    LOGGER.info("All computational and validation phases complete; outputs remain in the selected local results directory")
    for handler in LOGGER.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
