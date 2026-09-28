# Experiment datasets

All files are JSON Lines, UTF-8 encoded, with one task per line. They contain experiment inputs, source variants, reference specifications, and frozen witnesses.

## Dafny

| File | Tasks | Description | SHA-256 |
|---|---:|---|---|
| `dafny/controlled.jsonl` | 191 | Paired faithful/weakened Dafny sources and frozen witnesses | `29ab973359e8038da37ac4c6c49efaa7e00cff497967f17ada25123a96b2b8bf` |
| `dafny/nl_primary.jsonl` | 191 | Primary translated-contract arm | `adff804d927c40817d4ae1aa3982146498e23d93872fb8a5d302abaf71d0fa3a` |
| `dafny/nl_secondary.jsonl` | 191 | Secondary translated-contract arm | `4b66bb9c15a908ed416a39a209eee0744dc24a5f02b256f4f02114950f94837b` |

The controlled set contains 576 positive and 1,017 negative witnesses. It is a selected first-order subset derived from [DafnyBench](https://huggingface.co/datasets/wendy-sun/DafnyBench): 85 MBPP-derived tasks, 34 Clover tasks, and 72 public-repository tasks. Each record stores the faithful source, weakened source, mutation operator, positive witnesses, and negative witnesses. The translated arms retain the same task IDs and witness sets.

## Temporal and assertion languages

| File | Tasks | Public-corpus basis | SHA-256 |
|---|---:|---|---|
| `xlang/ltl.jsonl` | 200 | nl2spec expert set and Efficient-Eng-2-LTL | `016d9ed5e5d92054cb17e389081eafb586eb0bba731d5a7d340acd51a1106e5e` |
| `xlang/mtl.jsonl` | 200 | NL2TL lifted timed subset | `9f5927efc86cded2a7f583a0cca41e7cd39f6693a48d641a5e1839275143f1b0` |
| `xlang/ptltl.jsonl` | 151 | NASA FRET industrial requirements | `09187d11d3a97bc077a9b46553ec58b39dd2ebd1ef810a516e67502aee6ee038` |
| `xlang/stl.jsonl` | 200 | DeepSTL | `3ba11e2a4d89bf27d1b11af406f8e70c9ef02ce7c0723691bdffceb44018b8ad` |
| `xlang/sva.jsonl` | 200 | NVIDIA FVEval nl2sva | `65638ee11f55851fe567b38a2c66d179459e5d0db4c4b795882a0c6e6632f665` |

Each temporal record includes the source corpus label, natural-language requirement, reference specification, normalized IR, typed variable domains, horizon, and independently constructed positive/negative traces.

## Core schema

Dafny controlled records use:

```text
name, faithful_src, unfaithful_src, weaken_op, positives, negatives, tests
```

Translated Dafny records use:

```text
name, nl_requirement, nl_src, translated_requires, translated_ensures,
positives, negatives, tests
```

Temporal records use:

```text
task_id, lang, source, nl, spec_ref, spec_ir, spec_json, vars, dt,
horizon, pos, neg, witness_stats
```

