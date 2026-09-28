# SpecForge

SpecForge evaluates whether a verified specification is faithful to independently fixed intent witnesses and uses accepted negative witnesses to guide append-only Dafny postcondition repair. The released implementation follows the paper's V1--V3 pipeline:

1. establish Dafny verification (V1);
2. check positive and negative witnesses with admission-aware verifier probes (V2);
3. append witness-guided `ensures` clauses, re-verify, and enforce clause/body preservation (V3).

The repository contains the SpecForge implementation, its Dafny and temporal-logic evaluation code, and the experiment input datasets used by the paper.

## Setup

Python 3.10+ and Dafny 4.11.0 are recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[anthropic,test]"
```

Install Dafny from the [official Dafny repository](https://github.com/dafny-lang/dafny) and ensure `dafny` is on `PATH`. For model calls, configure one official provider:

```bash
export ANTHROPIC_API_KEY=...
# or
export OPENAI_API_KEY=...
```

Use `config/anthropic.yaml` or `config/openai.yaml`. Both clients use the provider's official SDK endpoint.

## Run SpecForge

Run a small controlled experiment first:

```bash
python -m experiments.run_specforge \
  --data data/dafny/controlled.jsonl \
  --arm controlled \
  --config config/anthropic.yaml \
  --limit 2 \
  --out outputs/smoke.jsonl
```

Remove `--limit` for the complete arm. The controlled dataset produces two instances per task (faithful and weakened). For translated contracts, select `--arm nl` and use `data/dafny/nl_primary.jsonl` or `data/dafny/nl_secondary.jsonl`.

Translated contracts that initially fail V1 require a proof-recovery tool. Clone the selected upstream implementation and expose it through a wrapper that reads `{input}` and writes `{output}`; then enable `proof_recovery.command` in the YAML configuration. The adapter validates the returned source with Dafny and the V3 preservation guard.

## Cross-language criterion

The language-agnostic TFVR implementation is under `specforge/xlang/`. For example:

```bash
python -m experiments.run_xlang \
  --data data/xlang/ltl.jsonl \
  --out outputs/ltl.jsonl \
  --smt
```

The same entry point supports `mtl.jsonl`, `stl.jsonl`, `ptltl.jsonl`, and `sva.jsonl`.

## Comparison systems

Comparison methods are run from their upstream repositories and connected at the experiment boundary. See [BASELINES.md](BASELINES.md) for repository links, method descriptions, and the adapter contract.

## Datasets

See [data/README.md](data/README.md) for task counts, schemas, provenance, and SHA-256 checksums. The Dafny controlled set contains 191 reference tasks, 576 positive witnesses, and 1,017 negative witnesses. The two translated-contract sets contain 191 tasks each. The temporal datasets contain 951 tasks across five languages.

## Validation

```bash
pytest
python tools/validate_release.py
```

The test suite includes offline algorithm tests, dataset-integrity checks, and a real-Dafny smoke test when Dafny is available.

## Repository layout

```text
specforge/              SpecForge algorithm, V1--V3 checks, Dafny adapter, temporal semantics
experiments/            SpecForge and cross-language experiment entry points
experiments/adapters/   integration boundary for independently installed tools
data/                   released experiment input datasets
config/                 official-provider example configurations
tests/                  offline and Dafny smoke tests
tools/                  release validation
```

