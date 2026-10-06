# Fixed-carrier graph aggregation: source code

This repository contains author-owned Python source, operation instructions,
environment requirements and the MIT license. It does not contain research data,
recorded results, historical host/seed manifests, predictions, checkpoints, plots,
audit logs, manuscript/proof files or submission materials. Dependencies and the
NCI1 dataset must be obtained separately under their applicable terms.

## Environment

The recorded Python version was 3.13.5. `requirements.txt` records the observed
library versions where available; the Matplotlib version was not recorded.
The observed Torch build was 2.9.0+cu128. Match Torch, PyTorch Geometric and
torch-scatter builds to the selected CPU/CUDA environment using their own
installation instructions. This requirements file is not a validated lockfile.
No third-party package or dataset is vendored or relicensed by this repository.

## Programs

| Path | Purpose |
|---|---|
| `wp1_uncertainty/wp1_uncertainty.py` | Synthetic carrier-noise program |
| `wp4_theorem/verify_kbit_realization.py` | Finite graph-construction verifier for k = 3, 4, 5, 6 |
| `wp2/scripts/frontier_common.py` | Fixed-carrier graph, moment and readout helpers |
| `wp2/scripts/run_is_extension_package.py` | Original host-anchor selection helper |
| `wp2/scripts/run_wp2_benchmark.py` | Planted-host benchmark engine |
| `wp2/scripts/reproduce_wp2.py` | Launcher requiring external data and historical manifest |

The WP1 and finite-verification programs can be invoked from the repository root:

```sh
python wp1_uncertainty/wp1_uncertainty.py
python wp4_theorem/verify_kbit_realization.py
```

They generate new local outputs in their respective `results/` directories.
No recorded output is distributed here. Finite verification does not establish
a general theorem independently of its proof.

## NCI1 benchmark inputs

Obtain NCI1 separately from its provider:
https://chrsmrrs.github.io/datasets/docs/datasets/
The benchmark also requires an external historical viewed-host exclusion manifest.
That research record is intentionally absent from this source-only release.
The repository alone therefore cannot reconstruct the recorded NCI1 protocol.
The benchmark retains the original manifest requirement; no empty replacement
or unaudited fresh-test substitution is supplied.

```sh
python wp2/scripts/reproduce_wp2.py --help
python wp2/scripts/reproduce_wp2.py --dataset-root /path/to/nci1-cache --historical-manifest /path/to/prior-host-manifest.json --results-root /path/to/local-results --mode dataset-only
```

`dataset-only` constructs and validates supplied data and stops before fitting.
`--mode full` performs model fitting and requires the benchmark's CUDA-capable
environment. Generated outputs remain local; they are excluded from this release.
The task that prepared this release did not rerun experiments or fit models.

## Source scope

Computational routines and protocol checks retain the reviewed source. The
public version removes an embedded reviewer-response generator and automatic
archive packaging. The anchor module contains the original selection function;
unrelated document/table export helpers are excluded. The launcher accepts an
external manifest explicitly. No recorded evaluation values or data payloads
are embedded as substitutes for the excluded files.

## License

The authors authorized the MIT license for their software and associated release
documentation/configuration; see `LICENSE`. The named authors are Jih-Jeng Huang
and Chin-Yi Chen. Imported libraries, external datasets and third-party materials
retain their own terms. No research-output license is granted by this repository.
