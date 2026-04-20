---
title: Data Loading with Validation Pipeline
type: code-snippets
domain:
- ml/data-loading
- ml/data-validation
---

## Source Code

The code loads subject data from a `.npz` file, validates its structure and content, and converts it into a dataset object (`grain.MapDataset`) suitable for training or testing.

```python
from pathlib import Path
from typing import Literal

import grain
from jax import numpy as jnp

REQUIRES_KEYS = ("train_data", "train_label", "test_data", "test_label")
VALID_LABELS = (0, 1, 2, 3, 4)

N_ELECTRODES = 62
N_BANDS = 5

Split = Literal["train", "test"]
SPLITS = {
    "train": ("train_data", "train_label"),
    "test": ("test_data", "test_label"),
}


def _check_required_keys(npz):
    missing = [k for k in REQUIRES_KEYS if k not in npz]
    if missing:
        raise ValueError(f"npz is missing required keys: {missing}")


def _check_split_lengths(split: Split):
    def _check(npz):
        data_key, label_key = SPLITS[split]
        n_data = npz[data_key].shape[0]
        n_label = npz[label_key].shape[0]
        if n_data != n_label:
            raise ValueError(
                f"{data_key} and {label_key} mismatch: "
                f"data: {n_data}, label: {n_label}"
            )

    return _check


def _check_label_values(split: Split, valid=VALID_LABELS):
    def _check(npz):
        _, label_key = SPLITS[split]
        labels = npz[label_key]
        unique = jnp.unique(labels).tolist()
        bad = sorted(set(unique) - set(valid))
        if bad:
            raise ValueError(
                f"{label_key} has unexpected values: {bad}; expected: {valid}"
            )

    return _check


def _check_data_shape(split: Split, expected_tail=(N_ELECTRODES, N_BANDS)):
    def _check(npz):
        data_key, _ = SPLITS[split]
        shape = npz[data_key].shape
        if shape[1:] != expected_tail:
            raise ValueError(
                f"{data_key} has shape {shape}; expected (N, *{expected_tail})"
            )

    return _check


VALIDATORS = (
    _check_required_keys,
    _check_split_lengths(split="train"),
    _check_split_lengths(split="test"),
    _check_label_values(split="train"),
    _check_label_values(split="test"),
    _check_data_shape(split="train"),
    _check_data_shape(split="test"),
)


def load_subject(path: Path, split: Split = "train") -> grain.MapDataset:
    npz = jnp.load(path)
    for check in VALIDATORS:
        check(npz)

    data_key, label_key = SPLITS[split]
    data = npz[data_key]
    labels = npz[label_key].astype("int64")

    samples = list(zip(data, labels))
    return grain.MapDataset.source(samples)
```

## Units

1. **Declare domain constants as module-level tuples** — name the invariants of the data format (required keys, valid label set, expected dimensions) as immutable top-level constants, not literals buried in function bodies. ♻️ reusable
    
2. **Type-alias a small closed string set** — give a `Literal[...]` alias to a finite choice like `"train" | "test"` so function signatures document the legal values. ♻️ reusable
    
3. **Build a lookup table from alias to underlying keys** — map a user-facing tag (`"train"`) to the low-level storage keys it resolves to (`"train_data"`, `"train_label"`), so callers pass the tag and the function resolves the pair. ♻️ reusable
    
4. **Write each invariant as its own one-purpose checker** — one validator per property (keys present, lengths match, labels in set, shape tail matches), so failures name exactly which invariant broke. ♻️ reusable
    
5. **Parameterize a validator via a closure factory** — outer function takes the varying parameter (`split`), returns an inner `_check(npz)` that captures it; lets you emit per-variant validators from one template. ♻️ reusable
    
6. **Compute set difference for membership violations** — subtract the allowed set from the observed set and report the remainder, so the error message shows _which_ offending values were found, not just that some were. ♻️ reusable
    
7. **Assert shape tails, not full shapes** — check `shape[1:] == expected_tail` so the leading batch dimension stays free while the structural dimensions are pinned. ♻️ reusable
    
8. **Freeze a validation pipeline as an ordered tuple** — collect all checker callables into one top-level `VALIDATORS` tuple in the order they should run, so the pipeline is declarative and inspectable. ♻️ reusable
    
9. **Run validators in a loop before any real work** — iterate the pipeline and call each checker on the raw input before touching the data path; fail fast, fail before transformation. ♻️ reusable
    
10. **Raise with the offending value embedded in the message** — every `ValueError` includes the actual bad value (missing keys, mismatched counts, unexpected labels, wrong shape), not a generic failure string. ♻️ reusable
    
11. **Cast label dtype explicitly after load** — coerce labels to `int64` right after extraction so downstream code never has to guess the integer width. ♻️ reusable
    
12. **Zip parallel arrays into per-sample tuples** — pair `data[i]` with `label[i]` via `zip` to produce a list of `(x, y)` samples, decoupling the storage layout (two arrays) from the iteration layout (one stream of pairs). ♻️ reusable
    
13. **Wrap the sample list in a dataset source adapter** — hand the list to `grain.MapDataset.source(...)` so the caller receives a dataset object with a standard interface instead of a raw list. 📍 context-bound
    
14. **Load EEG subject file into split-selected dataset** — end-to-end: open an `.npz`, validate, select the requested split, emit a dataset — with the split choice defaulted to `"train"`. 📍 context-bound