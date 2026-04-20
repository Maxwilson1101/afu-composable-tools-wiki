---
date-generated: 2026-04-17
generated-from:
  - "[[validation-as-pipeline-stage]]"
uses:
  - "[[validation-as-pipeline-stage]]"
pattern: adversarial-twist
status: proposed
---

# Validate a streaming sensor feed

Your EEG pipeline now receives data from a live sensor instead of a file. Samples arrive as a generator of dicts, one per time window:

```python
def sensor_stream() -> Generator[dict, None, None]:
    while True:
        yield acquire_sample()  # {"eeg": np.ndarray, "label": int, "timestamp": float}
```

The stream is infinite and you cannot buffer it entirely before validating — memory is bounded. Design a validation stage for this pipeline.

Requirements:
- Validate each sample as it arrives (required keys, expected shape, sane value ranges).
- Decide and implement a policy for malformed samples: does one bad sample abort the whole stream, or does the pipeline skip it and continue?
- The validation logic must be testable in isolation without running real sensor hardware.

## Stretches

[[validation-as-pipeline-stage]] currently validates a fully-loaded file — all keys are present or the whole load fails. Here the data is unbounded and arrives incrementally, so "validate at load time" has no meaning. The stage must move inside the loop, and the raise-on-failure contract must be reconsidered: aborting the stream on one bad sample may be wrong.

## Difficulty vs source

harder-because: The source problem has one validation decision (valid or not) with a clear raise. Here you must choose a policy (abort vs skip vs quarantine), implement it as part of the stage, and keep the stage testable despite the infinite input.

## Hints

A generator wrapper is the natural shape: `def validated_stream(raw: Generator) -> Generator`. It pulls from `raw`, validates each sample, and either yields it or handles the error. The validation function itself stays pure and unchanged — only the policy wrapper is new.
