# TODO Lists

[[situations/dataloader-pil-image-type-error]]

- [ ] seperate `collate_fn` error from error of `ToImage`
  - [ ] make reproductive `collate_fn` error
  - [ ] make reproductive `ToImage` error

<https://mlops.community/smoke-testing-for-ml-pipelines/>
<https://arxiv.org/pdf/2009.01521>
<https://openreview.net/pdf?id=eC4ygDs02R>

[[carla-od-training]]

https://github.com/kelvins/awesome-mlops
https://medium.com/online-inference/top-mlops-tools-in-2026-858fd479acac

https://docs.jaxstack.ai/en/latest/JAX_basic_text_classification.html

- train
- eval
- smoke test
- log
- checkpoint

https://github.com/ziyujia/Signal-feature-extraction_DE-and-PSD
https://github.com/eerstone/eeg_signal_feature_extraction_de_psd
https://github.com/ynulonger/DE_CNN
https://github.com/KeiraLalala/DGCNN_EEG_EmotionRecognition
https://github.com/ziyujia/SST-EmotionNet
https://github.com/shivam-199/Python-Emotion-using-EEG-Signal

<https://plotly.com/python/knn-classification/>
<https://plotly.com/python/pca-visualization/>
<https://plotly.com/python/plotly-express/>
<https://dash.plotly.com/>

Baseline

tool: sklearn + plotly
features: DE + Band Ratio
model: svm

---


## 5. System Operations

The system supports three agent operations. Each is triggered by an `ai-prompt` file with the matching `operation` field.

### Operation 1 — Review Units (`review-units`)

#### Goal

Evaluate whether individual Units are correct, efficient, minimal, and reusable.

#### Scope

One or more unit files (specified by `scope:` in the prompt frontmatter).

#### Agent Actions

For each Unit in scope:

1. Verify correctness of reasoning in each `## Attempt` block
2. Detect logical gaps — label each as `[GAP]` in the output
3. Identify redundant Attempts or steps
4. Suggest alternative methods where applicable
5. Evaluate reusability: **High** / **Medium** / **Low** with a one-line reason

#### Output File

`AI/CRITIQUE.md` (or scoped: `AI/CRITIQUE-<module-name>.md` for module-scoped reviews)

#### Output Structure

```markdown
---
type: ai-output
source: <prompt-name>.md
operation: review-units
scope: <module-name>/
created: <YYYY-MM-DD>
---

# Unit Critique — <scope>

## Critique — <unit-name>

### Correctness

<one paragraph assessment>

### Gaps

- [GAP] <description of logical gap or missing justification>

### Redundancy

<assessment — or "None detected.">

### Reusability

**Rating:** High | Medium | Low

**Reason:** <one line>

### Suggested Improvements

- <actionable suggestion>
```

### Operation 2 — Review Module Decomposition (`review-modules`)

#### Goal

Evaluate whether a Trace has been decomposed into Modules effectively.

#### Scope

The Trace index plus all module indexes (specified by `scope:` pointing to the trace folder).

#### Agent Actions

1. Evaluate structural validity of the module ordering
2. Detect missing phases (logical work with no module)
3. Detect redundant modules (two modules with overlapping scope)
4. Suggest improved ordering using dependency analysis
5. Propose alternative decompositions where warranted

#### Evaluation Criteria

- Logical coherence (do modules follow from each other?)
- Dependency correctness (does `deps:` match actual data flow?)
- Granularity balance (no module too large or too trivial)
- Scalability (could a new module be added without restructuring?)

#### Output File

`CRITIQUE.md`

#### Output Structure

```markdown
## Critique — <trace-name> (module decomposition)

### Structure

<assessment of overall module ordering and completeness>

### Missing Phases

- <phase-name> — <reason it should exist>

### Redundant Modules

- <module-name> — <reason for overlap>

### Dependency Issues

- <issue description referencing specific `deps:` fields>

### Suggested Reordering

| Current Order | Suggested Order | Reason |
|---|---|---|
| <module> | <module> | <reason> |

### Alternative Decompositions

<one or more alternative structures with rationale>
```

### Operation 3 — Generate Transfer Problems (`generate-problems`)

#### Goal

Create new problems that preserve the reasoning structure and dependency relationships of the Trace, but change the domain entirely.

#### Scope

Full trace or specific modules (specified by `scope:`).

#### Agent Actions

1. Extract core reasoning patterns from Units — strip domain-specific surface features
2. Identify structural isomorphisms across modules (where the same pattern recurs)
3. Generate new problems in entirely different domains that:
   - Are structurally isomorphic to the original Units
   - Require the same reasoning patterns
   - Do not reuse the original domain
   - Reward abstraction over domain knowledge

#### Output File

`DERIVED.md`

#### Output Structure

```markdown
---
type: ai-output
source: <prompt-name>.md
operation: generate-problems
scope: <trace-name>/
created: <YYYY-MM-DD>
---

# Transfer Problems — <trace-name>

## Core Patterns

- **<pattern-name>**: <one-line domain-agnostic description>
- **<pattern-name>**: <one-line domain-agnostic description>

## Transfer — <new-domain>

### Easy

1. <problem — concept transfer, single pattern>

### Medium

1. <problem — applied reasoning, two or more patterns combined>

### Hard

1. <problem — cross-domain system design, full structural isomorphism>
```

---

## 6. Log System

**Location:** `AI/.log/agent.ndjson`

**Format:** NDJSON — Newline-Delimited JSON. One JSON object per line.

**Purpose:** Agent verification and operation tracing. **Not for human reading.** Agents use this log to confirm prior operations, detect partial runs, and avoid redundant work.

### 6.1 Log Entry Schema

Each line in `agent.ndjson` is a valid JSON object with the following fields:

```json
{
  "ts":            "<ISO-8601 datetime>",
  "op":            "review-units | review-modules | generate-problems",
  "scope":         "<relative path to targeted file or folder>",
  "agent":         "<model-id>",
  "status":        "start | complete | error",
  "output":        "<relative path to output file, or null>",
  "units_visited": ["<relative-path>", "..."],
  "issues": [
    {
      "type": "gap | redundancy | structural | naming | dependency",
      "ref":  "<relative path to file containing the issue>",
      "msg":  "<short machine-readable description>"
    }
  ],
  "meta": {}
}
```

### 6.2 Log Rules

1. Every agent operation MUST append a `"status": "start"` entry before any file reads begin.
2. Every agent operation MUST append a `"status": "complete"` or `"status": "error"` entry before the response ends.
3. `units_visited` MUST list every file the agent read during the operation, in order.
4. `issues` MUST be an array (empty `[]` if nothing detected — never omitted).
5. The log file is **append-only** — existing entries are never modified or deleted.
6. All string values are machine-parseable identifiers or short phrases. No prose sentences except `msg`.
7. `meta` is reserved for operation-specific key-value pairs; use `{}` if unused.

### 6.3 Example Entries

```jsonl
{"ts":"2026-04-16T10:00:00Z","op":"review-units","scope":"visualize-eeg-dataset/","agent":"claude-sonnet-4-6","status":"start","output":null,"units_visited":[],"issues":[],"meta":{}}
{"ts":"2026-04-16T10:01:34Z","op":"review-units","scope":"visualize-eeg-dataset/","agent":"claude-sonnet-4-6","status":"complete","output":"AI/CRITIQUE.md","units_visited":["visualize-eeg-dataset/construct-channel-band-heatmap.md","visualize-eeg-dataset/generate-band-radar-summary.md","shared/load-npz-file.md"],"issues":[{"type":"structural","ref":"visualize-eeg-dataset/construct-channel-band-heatmap.md","msg":"missing Attempt and Decision headings"},{"type":"redundancy","ref":"visualize-eeg-dataset/generate-band-radar-summary.md","msg":"two statistics combined in one Attempt block"}],"meta":{}}
```

### 6.4 Log Extension Points

The `meta` field accepts any additional key-value pairs for future operations. Reserved keys:

| Key | Type | Meaning |
|---|---|---|
| `pattern_count` | integer | Number of core patterns extracted (generate-problems only) |
| `transfer_domains` | string[] | New domains generated (generate-problems only) |
| `critique_depth` | `unit \| module \| trace` | Scope level reviewed |

---

## 7. Constraints

### Trace index must have

- [ ] `problem`, `created`, `modules` frontmatter
- [ ] H1 matching the trace folder name
- [ ] All modules linked with `[[module-name]]`
- [ ] `@TODO` TODO markers on unimplemented modules

### Module index must have

- [ ] `problem`, `created` frontmatter
- [ ] H1 matching the module folder name
- [ ] All steps linked with `[[wikilink]]`
- [ ] `@TODO` TODO markers on unimplemented units

### Unit file must have

- [ ] `problem`, `created` frontmatter
- [ ] H1 matching the unit's subject
- [ ] At least one `## Attempt [label]`

### Unit file must not have

- [ ] Unlabeled `## Attempt` headings
- [ ] `deps` entries without a `reason` field
- [ ] `## Module N` headings (Module headings belong in the Trace index only)

### Log entries must have

- [ ] All nine required fields present on every line
- [ ] A `start` entry written before any file reads begin
- [ ] A `complete` or `error` entry before the agent response ends
- [ ] No prose in any field except `msg` (short phrase, ≤ 120 characters)

---

## 8. Naming Conventions

Inherits all conventions from `template/SPEC.md` Section 6, extended with Trace and Module terms:

| Item | Convention | Examples |
|---|---|---|
| Trace folder | kebab-case noun-noun | `eeg-feature-extraction`, `distributed-kv-store` |
| Trace index file | same as folder | `eeg-feature-extraction.md` |
| Module folder | kebab-case noun-noun | `build-eeg-dataset`, `visualize-eeg-dataset` |
| Module index file | same as folder | `visualize-eeg-dataset.md` |
| Unit file | kebab-case verb-noun | `load-npz-file.md`, `group-eeg-samples-by-label.md` |
| Shared unit file | kebab-case verb-noun | `extract-npz-data.md` |
| AI prompt file | `prompt-to-<verb>-<noun>.md` | `prompt-to-review-units.md` |
| AI output file | SCREAMING-KEBAB | `CRITIQUE.md`, `PROBLEMS.md`, `CRITIQUE-visualize.md` |
| Log file | fixed | `AI/.log/agent.ndjson` |

---

- [ ] Allow modules to use `@impl` annotations to indicate specific implementation methods.
- [ ] Add module alias syntax examples to `SPEC.md`.
- [ ] Add specification support for reusable traces weakly related to the `modules` directory.
- [ ] Migrate tools in archive into newly devised modules