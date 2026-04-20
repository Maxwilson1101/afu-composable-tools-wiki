---
date-generated: 2026-04-17
generated-from:
  - "[[split-as-router]]"
  - "[[literal-type-alias]]"
uses:
  - "[[split-as-router]]"
  - "[[literal-type-alias]]"
pattern: domain-transfer
status: proposed
---

# CLI mode router

You are building `train.py`, a command-line script with a `--mode` argument. Valid values are `"train"`, `"eval"`, and `"export"`. Each mode runs a different top-level function: 

```
python train.py --mode train   → runs training loop
python train.py --mode eval    → runs evaluation on a checkpoint
python train.py --mode export  → exports the model to ONNX
```

Implement the routing such that:
1. An invalid `--mode` value is caught **before** any mode logic runs, with a clear error message listing the valid options.
2. Adding a new mode (e.g. `"profile"`) requires changing exactly one place in the code.
3. Pyright flags `--mode blah` at any call site where the mode is passed as a string literal in Python code (e.g. in tests that invoke `main(mode="blah")`).

You may use `argparse` or any argument-parsing library.

## Stretches

[[split-as-router]] in the source problem maps a string to data keys inside a class. Here the router is the entry point of an entire program and the mapped values are functions, not tuples. [[literal-type-alias]] in the source catches invalid values at static-analysis time inside Python; here the primary invalid-value path is at runtime via CLI input, which `Literal` alone cannot validate — a second enforcement layer is required.

## Difficulty vs source

harder-because: The source problem has one enforcement layer (the `Literal` type caught by Pyright). Here there are two distinct inputs — Python call sites (where `Literal` works) and CLI strings (where it does not) — and both must be handled, using different mechanisms, without duplicating the list of valid modes.

## Hints

The valid modes live in exactly one place: the `Literal` type alias (or equivalently, the dispatch dict keys). The argparse `choices=` parameter can be derived from that same source — `list(MODES.keys())` — so the CLI validation and the static type stay in sync automatically.
