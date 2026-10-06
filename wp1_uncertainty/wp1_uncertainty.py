#!/usr/bin/env python3
"""WP1 uncertainty for a fixed balanced synthetic carrier design."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

K = 3
DEGREES = (1, 2, 3)
NOISE = (0.00, 0.02, 0.05, 0.10)
R = 120
N_PAIRS = 24
N_TRAIN = 16
N_TEST = 8
SPLIT_SEED = 20260709
DATA_SEED = 20260710
BOOT_SEED = 20260711
BOOTSTRAP_RESAMPLES = 20_000
L2 = 1e-3
MAX_ITERATIONS = 100
TOLERANCE = 1e-8

ODD = tuple(c for c in itertools.product((0, 1), repeat=K) if sum(c) % 2 == 1)
EVEN = tuple(c for c in itertools.product((0, 1), repeat=K) if sum(c) % 2 == 0)
MONOMIALS = tuple(
    subset
    for degree in DEGREES
    for subset in itertools.combinations(range(K), degree)
)
MONOMIAL_DEGREES = np.array([len(subset) for subset in MONOMIALS])

_pair_permutation = np.random.default_rng(SPLIT_SEED).permutation(N_PAIRS)
TRAIN_PAIR_IDS = tuple(sorted(map(int, _pair_permutation[:N_TRAIN])))
TEST_PAIR_IDS = tuple(sorted(map(int, _pair_permutation[N_TRAIN:])))


def monomial_features(codes):
    return np.array(
        [
            sum(int(all(code[i] for i in subset)) for code in codes)
            for subset in MONOMIALS
        ],
        dtype=float,
    )


def apply_noise(codes, probability, rng):
    values = np.array(codes, dtype=np.int8)
    flips = (rng.random(values.shape) < probability).astype(np.int8)
    return tuple(map(tuple, np.bitwise_xor(values, flips).tolist()))


def make_dataset(probability, rng):
    features, labels, pair_ids = [], [], []
    for pair_id in range(N_PAIRS):
        features.append(monomial_features(apply_noise(ODD, probability, rng)))
        labels.append(1)
        pair_ids.append(pair_id)
        features.append(monomial_features(apply_noise(EVEN, probability, rng)))
        labels.append(0)
        pair_ids.append(pair_id)
    return (
        np.vstack(features),
        np.asarray(labels, dtype=int),
        np.asarray(pair_ids, dtype=int),
    )


def expit(values):
    output = np.empty_like(values, dtype=float)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def objective(design, labels, weights):
    logits = design @ weights
    penalty = 0.5 * L2 * (weights[:-1] @ weights[:-1])
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits) + penalty)


def fit_logistic(features, labels):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack(((features - mean) / scale, np.ones(len(features))))
    weights = np.zeros(design.shape[1])
    penalty = np.r_[np.full(design.shape[1] - 1, L2), 0.0]
    converged = False
    gradient_norm = float("inf")

    for iteration in range(MAX_ITERATIONS + 1):
        probabilities = expit(design @ weights)
        gradient = design.T @ (probabilities - labels) / len(labels) + penalty * weights
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= TOLERANCE:
            converged = True
            break
        if iteration == MAX_ITERATIONS:
            break

        curvature = probabilities * (1.0 - probabilities)
        hessian = (
            design.T @ (design * curvature[:, None]) / len(labels)
            + np.diag(penalty)
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        old_objective = objective(design, labels, weights)
        descent = float(gradient @ step)
        alpha = 1.0
        while alpha > 2 ** -20:
            candidate = weights - alpha * step
            if objective(design, labels, candidate) <= (
                old_objective - 1e-4 * alpha * descent
            ):
                weights = candidate
                break
            alpha *= 0.5
        else:
            break

    return mean, scale, weights, converged, iteration, gradient_norm


def predict(model, features):
    mean, scale, weights, *_ = model
    design = np.column_stack(((features - mean) / scale, np.ones(len(features))))
    return (expit(design @ weights) >= 0.5).astype(int)


def data_seed(noise_index, replicate_id):
    return int(
        np.random.SeedSequence(
            [DATA_SEED, noise_index, replicate_id]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def bootstrap_seed(noise_index):
    return int(
        np.random.SeedSequence(
            [BOOT_SEED, noise_index]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def summarize(values, bootstrap_indices):
    values = np.asarray(values, dtype=float)
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    )
    sample_sd = float(values.std(ddof=1))
    return {
        "mean_test_accuracy": round(float(values.mean()), 6),
        "replicate_sd": round(sample_sd, 6),
        "monte_carlo_se_of_mean": round(sample_sd / np.sqrt(len(values)), 6),
        "bootstrap_ci95_for_mean": [
            round(float(lower), 6),
            round(float(upper), 6),
        ],
        "n_independent_noise_replicates": int(len(values)),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    replicate_rows = []
    results = {}

    for noise_index, probability in enumerate(NOISE):
        values = {degree: [] for degree in DEGREES}
        converged_fits = {degree: 0 for degree in DEGREES}

        for replicate_id in range(R):
            seed = data_seed(noise_index, replicate_id)
            features, labels, pair_ids = make_dataset(
                probability, np.random.default_rng(seed)
            )
            train = np.isin(pair_ids, TRAIN_PAIR_IDS)
            test = np.isin(pair_ids, TEST_PAIR_IDS)
            assert (
                int(labels[train].sum()),
                int((1 - labels[train]).sum()),
                int(labels[test].sum()),
                int((1 - labels[test]).sum()),
            ) == (N_TRAIN, N_TRAIN, N_TEST, N_TEST)

            row = {
                "noise_p": f"{probability:.2f}",
                "noise_index": noise_index,
                "replicate_id": replicate_id,
                "data_seed_uint64": seed,
            }

            for degree in DEGREES:
                columns = MONOMIAL_DEGREES <= degree
                model = fit_logistic(features[train][:, columns], labels[train])
                predictions = predict(model, features[test][:, columns])
                correct = int(np.sum(predictions == labels[test]))
                accuracy = correct / len(labels[test])
                values[degree].append(accuracy)
                converged_fits[degree] += int(model[3])

                row[f"degree_{degree}_correct_of_16"] = correct
                row[f"degree_{degree}_test_accuracy"] = f"{accuracy:.6f}"
                row[f"degree_{degree}_fit_converged"] = int(model[3])
                row[f"degree_{degree}_newton_iterations"] = model[4]
                row[f"degree_{degree}_gradient_inf_norm"] = f"{model[5]:.12e}"

            replicate_rows.append(row)

        if probability == 0.0:
            assert set(values[1]) == {0.5}
            assert set(values[2]) == {0.5}
            assert set(values[3]) == {1.0}
        assert all(converged_fits[degree] == R for degree in DEGREES)

        seed = bootstrap_seed(noise_index)
        bootstrap_indices = np.random.default_rng(seed).integers(
            0, R, size=(BOOTSTRAP_RESAMPLES, R)
        )
        results[f"p={probability:.2f}"] = {
            "shared_bootstrap_seed_uint64": seed,
            **{
                f"degree_{degree}": {
                    **summarize(values[degree], bootstrap_indices),
                    "converged_fits": converged_fits[degree],
                }
                for degree in DEGREES
            },
        }

    csv_path = RESULTS / "wp1_replicates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(replicate_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(replicate_rows)

    report = {
        "schema_version": 2,
        "stage": "wp1_frontier_uncertainty",
        "estimand": {
            "quantity": (
                "mean balanced test accuracy of the fixed training rule over "
                "independent carrier-bit-noise realizations"
            ),
            "stochastic_unit": (
                "one complete independent Bernoulli bit-flip realization on "
                "all 48 fixed synthetic graph slots at a specified probability"
            ),
            "fixed_components": (
                "odd/even parity carrier multisets, 24 paired graph slots, "
                "the fixed 16/8 pairwise train/test split, and classifier specification"
            ),
            "scope_exclusion": (
                "no natural host graph, corpus split, or learned-carrier resampling "
                "is represented by this Monte Carlo estimand"
            ),
        },
        "design": {
            "k": K,
            "positive_class": "odd-parity multiset of the 3-bit cube",
            "negative_class": "even-parity multiset of the 3-bit cube",
            "noise_levels": list(NOISE),
            "independent_noise_replicates_per_level": R,
            "paired_graph_slots_per_class_per_replicate": N_PAIRS,
            "train_pair_ids": list(TRAIN_PAIR_IDS),
            "test_pair_ids": list(TEST_PAIR_IDS),
            "train_graphs_per_class": N_TRAIN,
            "test_graphs_per_class": N_TEST,
            "split_is_fixed_across_noise_levels_replicates_and_degrees": True,
            "same_noisy_dataset_is_used_for_all_degrees_within_each_replicate": True,
            "bit_flips_are_independent_across_graphs_states_and_coordinates": True,
            "accuracy_equals_balanced_accuracy": True,
        },
        "features_and_classifier": {
            "degree_rows": list(DEGREES),
            "feature_definition": (
                "aggregated counts of every nonconstant squarefree monomial "
                "of degree at most the stated row"
            ),
            "classifier": "L2-regularized logistic regression with unpenalized intercept",
            "optimizer": "damped Newton method",
            "l2": L2,
            "maximum_newton_iterations": MAX_ITERATIONS,
            "gradient_infinity_norm_tolerance": TOLERANCE,
            "decision_rule": "predicted probability >= 0.5 is class 1",
            "hyperparameters_selected_before_simulation": True,
        },
        "uncertainty_reporting": {
            "replicate_sd": "sample SD of 120 replicate-level test accuracies",
            "confidence_interval": (
                "two-sided 95% nonparametric percentile-bootstrap interval "
                "for the mean replicate-level test accuracy"
            ),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_unit": "complete replicate-level accuracy",
            "common_bootstrap_indices_across_degrees_within_noise_level": True,
        },
        "seeds": {
            "fixed_pair_split_seed": SPLIT_SEED,
            "data_root_seed": DATA_SEED,
            "bootstrap_root_seed": BOOT_SEED,
            "data_seed_derivation": "SeedSequence([data_root_seed, noise_index, replicate_id])",
            "bootstrap_seed_derivation": "SeedSequence([bootstrap_root_seed, noise_index])",
        },
        "results": results,
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": sha256(Path(__file__).resolve()),
            "replicate_file": csv_path.name,
            "replicate_file_sha256": sha256(csv_path),
        },
    }
    (RESULTS / "wp1_uncertainty.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        "# WP1 - fixed-carrier accuracy under independent bit-flip noise",
        "",
        (
            "Each cell is mean +/- replicate SD [95% percentile-bootstrap CI "
            f"for the mean] across {R} independent carrier-noise realizations."
        ),
        (
            f"The balanced design has {N_TRAIN} training and {N_TEST} test "
            "graphs per class; the pairwise split is fixed, and all degrees "
            "use the same noisy data within each replicate."
        ),
        "",
        "| flip probability p | degree 1 | degree 2 | degree 3 |",
        "|---:|---:|---:|---:|",
    ]
    for probability in NOISE:
        payload = results[f"p={probability:.2f}"]

        def cell(degree):
            summary = payload[f"degree_{degree}"]
            lower, upper = summary["bootstrap_ci95_for_mean"]
            return (
                f"{summary['mean_test_accuracy']:.3f} +/- "
                f"{summary['replicate_sd']:.3f} "
                f"[{lower:.3f}, {upper:.3f}]"
            )

        lines.append(
            f"| {probability:.2f} | {cell(1)} | {cell(2)} | {cell(3)} |"
        )
    lines.extend(
        [
            "",
            (
                "Estimand: mean balanced test accuracy over carrier-noise "
                "realizations for this fixed synthetic design and fixed training rule. "
                "The interval does not quantify host-corpus, learned-carrier, "
                "or natural-data uncertainty."
            ),
        ]
    )
    (RESULTS / "wp1_uncertainty_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()