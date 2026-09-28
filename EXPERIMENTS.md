# Reproducing the research questions

Run these commands from the repository root after installing the dependencies in
`README.md`. The input files are documented in `data/README.md`. Commands write
new records and summaries under `outputs/`, which is ignored by Git. `--limit`
selects a small number of reference tasks per dataset for a smoke run; omit it
for the full population. The reported percentages in the paper also depend on
the stated model versions, proof-recovery configuration, and complete runs.

| Question | Entry point | Input | Main output |
|---|---|---|---|
| RQ1: check--faithfulness gap | `experiments.rq1` | five temporal datasets and Dafny controlled | `outputs/rq1/summary.json` |
| RQ2: silent errors | `experiments.rq2` | RQ1 temporal candidate records | `outputs/rq2/summary.json` |
| RQ3: verdict validity | `experiments.rq3` | RQ1 records and frozen witnesses | `outputs/rq3/summary.json` |
| RQ4: untouched translations | `experiments.rq4` | natural-language requirements and model outputs | `outputs/rq4/translations.summary.json` |
| RQ5: guarded repair | `experiments.rq5` | three Dafny arms | `outputs/rq5/records.summary.json` |

## RQ1--RQ3: controlled candidates

```bash
python -m experiments.rq1 --dafny-path dafny
python -m experiments.rq2
python -m experiments.rq3
```

RQ1 uses 14 mutation candidates per temporal task, seed 0, and the bounded
symbolic relation check. Its temporal raw records are shared by RQ2 and RQ3.
The Dafny row uses only the weakened source from each of 191 controlled tasks;
both the verifier and the fixed positive/negative witnesses are checked.
RQ2 excludes equivalent candidates from its error denominator. RQ3 reports
recall on candidates with conclusive symbolic and witness verdicts, false
alarms on equivalent candidates, paired discordances, and nested witness caps.

A fast local check is:

```bash
python -m experiments.rq1 --limit 1 --k 2 --dafny-path dafny
python -m experiments.rq2
python -m experiments.rq3
```

Choose a different `--out-dir` for a full RQ1 run after a limited run; RQ2 and
RQ3 accept `--records` to read that run. RQ1 rewrites its own output files when
rerun. Keep different configurations in different output directories.

## RQ4: unmodified translator outputs

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in the environment and choose the
matching official-provider config. For example:

```bash
python -m experiments.rq4 --config config/rq4_anthropic.yaml
```

The prompt includes the natural-language requirement and variable signature,
but not the reference formula or witnesses. Candidate formulas are parsed and
scored against the frozen witnesses and bounded symbolic oracle. API failures
and parse failures are counted separately; `--resume` appends unfinished tasks.
Use a separate output file for each model configuration. The RQ4 example
configs cap output at 400 tokens; `config/rq4_openai.yaml` uses the same cap.

For predictions made by another translator, convert the upstream output to
JSONL with `task_id` and `spec_cand_src` and score it with the same evaluator:

```bash
python -m experiments.rq4 --candidates path/to/predictions.jsonl \
  --label upstream-model --out outputs/rq4/upstream.jsonl
```

The summary reports conditional rates on outputs with complete witness
verdicts, and equivalence accuracy on conclusive symbolic verdicts. These
denominators are deliberately separate from the attempted-translation count.

## RQ5: Dafny specification repair

The controlled arm is directly runnable with an official provider:

```bash
python -m experiments.rq5 --arm controlled --method both \
  --config config/anthropic.yaml --repeats 3 --dafny-path dafny
```

For a no-model smoke run of the fixed GivenSpec condition:

```bash
python -m experiments.rq5 --arm controlled --method given \
  --limit 1 --dafny-path dafny --out outputs/rq5/given-smoke.jsonl
```

The NL-A and NL-B paper protocol uses upstream proof recovery in Phase 1. Add
`proof_recovery.command` to a local YAML config with an executable wrapper that
accepts `--input {input}` and `--output {output}` as described in `BASELINES.md`.
Then run:

```bash
python -m experiments.rq5 --arm controlled --method both --repeats 3 \
  --config path/to/local-config.yaml --dafny-path dafny \
  --out outputs/rq5/controlled.jsonl
python -m experiments.rq5 --arm nl-a --method both \
  --config path/to/local-config.yaml --dafny-path dafny \
  --out outputs/rq5/nl-a.jsonl
python -m experiments.rq5 --arm nl-b --method both \
  --config path/to/local-config.yaml --dafny-path dafny \
  --out outputs/rq5/nl-b.jsonl
python -m experiments.rq5_compare \
  --records outputs/rq5/controlled.jsonl \
  --records outputs/rq5/nl-a.jsonl \
  --records outputs/rq5/nl-b.jsonl
```

`rq5_compare` accepts external method records through repeatable
`--external METHOD=FILE` options. Each record must contain `arm`, `run`, `task`,
and Boolean `tfvr` (or `null` for an unscored provider failure). Pairing uses
the exact arm, run and task. The report gives exact paired McNemar tests and
Holm-adjusted values over the available prior systems within each arm and run.
Use `--weak-only --out outputs/rq5/weak-comparison.json` for the controlled
sensitivity with one weakened variant per reference task. The comparison
systems themselves are obtained from the upstream projects in `BASELINES.md`.

The initial paper run and later repeated runs are distinct experiments.
Reproducing a paper number requires the matching model and upstream tool
versions; a newer hosted model or unavailable upstream proof-recovery release
can change the observed rate. A failed provider call receives no outcome score.
As checked on 2026-09-28, the AxDafny code URL cited by its authors returns
"repository not found". Until the upstream project becomes accessible, the
paper's exact NL Phase 1 and AxDafny comparison cannot be rerun from that URL.
