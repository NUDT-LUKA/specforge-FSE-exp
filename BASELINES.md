# Comparison systems

SpecForge uses a small process boundary for comparison systems. Clone and configure each project according to its own documentation, then write a wrapper that accepts:

```text
--input  PATH_TO_INPUT_DFY
--output PATH_TO_OUTPUT_DFY
```

The wrapper writes one complete Dafny source file to the output path. `ExternalProofRecoverer` executes the wrapper without a shell and SpecForge independently checks verification and preservation.

## Upstream projects

| System | Role in the paper | Upstream source |
|---|---|---|
| AxDafny | Primary verifier-guided proof-recovery baseline and Phase-1 prover | [Axiomatic-AI/ax-dafny](https://github.com/Axiomatic-AI/ax-dafny) |
| dafny-annotator | LLM-guided annotation search | [metareflection/dafny-annotator](https://github.com/metareflection/dafny-annotator) |
| Laurel | Placeholder localization and example retrieval for Dafny assertions | [emugnier/dafny_repair](https://github.com/emugnier/dafny_repair) |
| DafnyPro | Diff checking, pruning, and proof-hint augmentation | [paper and implementation description](https://arxiv.org/abs/2601.05385) |
| Dafny | Verifier used by all Dafny arms | [dafny-lang/dafny](https://github.com/dafny-lang/dafny) |

`GivenSpec` is the unchanged supplied source condition. `DirectDafny` is the paper's direct prompting protocol rather than a separately released system.

## Adapter example

```yaml
proof_recovery:
  command:
    - python
    - ../ax-dafny-wrapper/run.py
    - --input
    - "{input}"
    - --output
    - "{output}"
  timeout: 900
```

Keep the upstream checkout outside this repository so its version, environment, and license remain explicit.

