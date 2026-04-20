---
title: A Training loop example from equinox's document
type: code-snippets
domain:
- ml/training
---

ref: <https://flax.readthedocs.io/en/stable/mnist_tutorial.html>

#todo
- [ ] read source code of PyTorch Lightning <https://github.com/Lightning-AI/pytorch-lightning>
- [ ] review the conversation <https://claude.ai/share/eddf48f8-f7aa-47ae-bb0d-4a101f71cf0f>
- [ ] design a Pytorch Lightning like Trainer API using Jax and Flax

## Source Code

helper functions

```python
def loss_fn(model: CNN, batch, rngs: nnx.Rngs | None = None):
  logits = model(batch['image'], rngs)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch, rngs)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(model, grads)  # In-place updates.

@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
```

main loop

```python
from IPython.display import clear_output
import matplotlib.pyplot as plt

metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}

rngs = nnx.Rngs(0)
train_model = nnx.view(model, deterministic=False, use_running_average=False)
eval_model = nnx.view(model, deterministic=True, use_running_average=True)

for step, batch in enumerate(train_ds.as_numpy_iterator()):
  # Run the optimization for one step and make a stateful update to the following:
  # - The train state's model parameters
  # - The optimizer state
  # - The training loss and accuracy batch metrics
  train_step(train_model, optimizer, metrics, rngs, batch)

  if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
	# One training epoch has passed.
    # Log the training metrics.
    for metric, value in metrics.compute().items():  # Compute the metrics.
      metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
    metrics.reset()  # Reset the metrics for the test set.

    # Compute the metrics on the test set after each training epoch.
    for test_batch in test_ds.as_numpy_iterator():
      eval_step(eval_model, metrics, test_batch)

    # Log the test metrics.
    for metric, value in metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)
    metrics.reset()  # Reset the metrics for the next training epoch.

    clear_output(wait=True)
    # Plot loss and accuracy in subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    for dataset in ('train', 'test'):
      ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
      ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
    ax1.legend()
    ax2.legend()
    plt.show()
```

## Units

1. **Define `loss_fn` to return `(loss, aux)` for `has_aux=True` differentiation** — return the scalar loss alongside non-differentiated auxiliary outputs (logits) so downstream metrics can consume them without a second forward pass. ♻️ reusable
2. **Thread an `nnx.Rngs` object through the loss signature as an optional argument** — pass RNG state as a first-class parameter (defaulting to `None` for eval) so stochastic layers like dropout can draw keys without module-level globals. ♻️ reusable
3. **Use `optax.softmax_cross_entropy_with_integer_labels` to skip one-hot encoding** — feed raw integer class labels to the combined softmax+cross-entropy op so you avoid the allocation and numerical cost of explicit one-hot conversion. ♻️ reusable
4. **JIT the whole train step with `@nnx.jit` over stateful module arguments** — decorate a function that takes live NNX modules (`model`, `optimizer`, `metrics`) so the transform traces through the module graph rather than requiring a pure-function rewrite. ♻️ reusable
5. **Differentiate with `nnx.value_and_grad(..., has_aux=True)` over the module argument** — unpack the nested return as `(loss, aux), grads` so loss, auxiliary outputs, and gradients all come from a single backward pass. ♻️ reusable
6. **Mutate metrics with `metrics.update(...)` inside the JITted step** — call the stateful metric accumulator in-place inside the traced region; NNX tracks the mutation through its reference system rather than requiring returned state. 📍 context-bound
7. **Mutate optimizer and params with `optimizer.update(model, grads)` in-place** — let the NNX optimizer wrapper apply gradients directly to the module's parameter references instead of receiving and returning a functional `opt_state`. 📍 context-bound
8. **Separate train and eval step functions that share the same `loss_fn`** — define `train_step` (with grads + optimizer) and `eval_step` (forward + metrics only) as two JITted entry points over one loss definition, so the forward path stays DRY. ♻️ reusable
9. **Derive mode-specific module views with `nnx.view(model, deterministic=..., use_running_average=...)`** — produce a `train_model` and `eval_model` from the same underlying module, flipping dropout and batch-norm flags via view rather than by passing a `training` kwarg everywhere. 📍 context-bound
10. **Initialize a single `nnx.Rngs(seed)` at the loop boundary** — construct the RNG container once outside the step, then hand it in each call so key splitting is managed by the Rngs object rather than by manual `jax.random.split`. 📍 context-bound
11. **Pre-declare a metrics history dict keyed by `{split}_{metric}`** — stand up an empty dict with all `train_*` / `test_*` keys before the loop so appending inside the logging branch is a plain `.append`, never a missing-key check. ♻️ reusable
12. **Iterate the dataset directly with `.as_numpy_iterator()`** — convert the TF/grain dataset to a NumPy-yielding iterator once, then drive the training loop with plain `enumerate`, so batches arrive as NumPy arrays at the JAX boundary. ♻️ reusable
13. **Gate eval with `step > 0 and (step % eval_every == 0 or step == train_steps - 1)`** — combine a non-zero-step guard, a modulo cadence, and an explicit final-step check so eval runs periodically but never at step 0 and always on the last step. ♻️ reusable
14. **Fold an accumulator with `compute → record → reset` at each log boundary** — call the metric's `.compute()` to snapshot, append to history, then `.reset()` before the next phase, so train and test accumulations never bleed into each other. ♻️ reusable
15. **Use one metrics accumulator across both splits by reset-between-phases** — share a single `nnx.MultiMetric` for train and test rather than keeping two, relying on explicit `reset()` to delimit which split is being measured. 📍 context-bound
16. **Loop the test set inside the training step's logging branch** — run a full pass over `test_ds` only on eval steps, nesting the eval loop inside the `if` so test-set cost is amortized across the eval interval. ♻️ reusable
17. **Record metrics under programmatic keys via f-strings** — write `metrics_history[f'{dataset}_{metric}'].append(value)` so a single loop body handles both splits without duplicated code. ♻️ reusable
18. **Live-plot training curves with `clear_output(wait=True)` + `plt.show()`** — clear the notebook output cell before each re-plot so the figure updates in place instead of accumulating a new plot per eval step. 📍 context-bound
19. **Plot paired metrics in side-by-side subplots** — use `plt.subplots(1, 2)` with one axis per metric family (loss, accuracy) and overlay train/test on each, so the comparison is two panels rather than four. ♻️ reusable
20. **Drive both train and test curves from one dataset loop** — iterate `for dataset in ('train', 'test')` and plot onto both axes inside that single loop, so adding a new split is one literal change, not four plot calls. ♻️ reusable

## The central design commitment: state lives in mutable reference objects

In Equinox, `model`, `opt_state`, and metrics were **values** — you passed them in, you got new ones back, you rebound names in the outer loop. In NNX, they're **reference objects** — you pass them in, the function mutates them in place, nothing is returned. Look at `train_step`:

```python
def train_step(model, optimizer, metrics, rngs, batch):
    (loss, logits), grads = grad_fn(model, batch, rngs)
    metrics.update(...)       # no return
    optimizer.update(...)     # no return
# the function itself returns None
```

This is JAX code that _looks like PyTorch_. That's the whole pitch of NNX. The underlying machinery is still pure (NNX unwraps modules into PyTrees, runs the JIT on a functional version, writes results back to the refs), but the surface syntax hides the pure-function plumbing. Whether this is good or bad is a taste call — what matters is recognizing that NNX is making a deliberate bet that **ergonomic familiarity beats explicit purity** for most users.

Compare where the state lives in each framework:

|Concern|Equinox|NNX|
|---|---|---|
|Model params|Leaves of a PyTree you thread through|Mutable refs inside the module|
|Optimizer state|`opt_state` you thread through|Owned by `nnx.Optimizer`, mutated in place|
|Metrics|Your responsibility to accumulate|`nnx.MultiMetric` object, mutated in place|
|RNG|Explicit `jax.random.split`|`nnx.Rngs` container that splits internally|
|Mode flags (train/eval)|Passed as bool argument|Flipped by `nnx.view` producing a new view|

Every row is the same tradeoff: NNX gives you an object to hold the state so you don't have to thread it; you pay in a less transparent relationship between what you wrote and what JAX actually sees.

### `nnx.view` as the replacement for `training=True`

This is the most interesting idiom in the example and worth lingering on. In PyTorch or in Flax Linen you'd do `model.train()` / `model.eval()` or pass `training=True`. In NNX the equivalent is:

```python
train_model = nnx.view(model, deterministic=False, use_running_average=False)
eval_model  = nnx.view(model, deterministic=True,  use_running_average=True)
```

Both views point at the same underlying parameters — they're not copies. What differs is the flag values that dropout and batch-norm layers will see when called. So you can train the model via `train_model` and the updates land on the shared params; you then evaluate via `eval_model` and batch-norm uses its running averages instead of batch stats.

The generalizable lesson: **when behavior varies by context (train vs eval, deterministic vs stochastic), prefer a view over a kwarg.** The view localizes the mode-switching to one place (the view creation), not N places (every layer call site). The same pattern shows up in databases (read-only views), graphics (camera views of a scene), and ML generally (teacher/student sharing weights).

### The `Rngs` container abstracts away key splitting

Equinox and raw JAX make you manage RNG keys by hand: `key, subkey = jax.random.split(key)`, thread both through, pass `subkey` to the call site that needs randomness. NNX hides this behind `nnx.Rngs(0)`, which you construct once and pass to any layer that needs a key — it manages the splits internally.

The cost: you now have a stateful object with implicit key advancement, which is harder to reason about when you want reproducibility across a specific call or want to branch a computation on two independent key streams. The benefit: 90% of training loops never care about that, and removing key bookkeeping from every function signature is real ergonomic relief.

The broader pattern: **when a resource has a mechanical advancement rule that's almost always the same, wrap it in an object whose `.next()` encapsulates the rule.** You see the same shape in database sequence generators, snowflake ID generators, and pagination cursors.

### Metrics as a first-class reusable accumulator

`nnx.MultiMetric` is a small but excellent piece of design. It holds multiple named metrics, you call `.update(loss=..., logits=..., labels=...)` each batch, `.compute()` at eval boundaries, `.reset()` to clear. Notice what falls out of this:

The example uses _one_ MultiMetric for both train and test splits, delimited by `.reset()` calls between phases. That's a small cleverness — the accumulator doesn't care which split fed it, and reset is cheap. The alternative would be two parallel metric objects and you'd pass the right one to each step. Both work; the single-accumulator-with-reset version requires less ceremony and encodes the invariant "we measure one split at a time" in the code structure.

The generalizable lesson: **a well-designed accumulator exposes three operations — update, compute, reset — and nothing else.** That's the shape of running means, running quantiles, Prometheus counters, sufficient-statistics online algorithms. If your "metrics" code grows a fourth operation, you probably have two concerns tangled together.

### The eval-gate idiom refined

The condition `step > 0 and (step % eval_every == 0 or step == train_steps - 1)` does three things:

- `step > 0` — don't eval on step 0, where the model is at initialization and metrics are meaningless (also: training metrics haven't accumulated yet, so `.compute()` would be undefined).
- `step % eval_every == 0` — periodic cadence.
- `step == train_steps - 1` — always eval on the last step regardless of whether the cadence hits.

The Equinox example had the second and third guards but not the first. The first matters here because of the metrics reset logic — eval-ing at step 0 would wipe the empty train metrics before any training happened, which is fine but wasteful. It's worth noticing that **each guard in a compound condition usually exists because of something that went wrong without it**; reading them is archaeology on previous bugs.

### Live-plotting in the notebook is a tiny but real discipline

`from IPython.display import clear_output` + `clear_output(wait=True)` + `plt.show()` is the canonical notebook live-dashboard pattern. The `wait=True` is load-bearing — without it you get a flicker where the cell empties, then the new plot appears; with it, the clear happens only when the new output is ready to replace it.

Two things worth internalizing beyond the mechanics:

The plotting loop `for dataset in ('train', 'test'): ax1.plot(history[f'{dataset}_loss'], ...)` is the payoff for keying the metrics dict as `{split}_{metric}` earlier. A single f-string iteration handles both splits. If you'd keyed it as nested dicts (`history['train']['loss']`), you'd need a different plot expression. Flat string-keyed history trades a bit of type safety for iteration ergonomics, and in quick-iteration research code that's usually the right trade.

The side-by-side subplot pattern (`plt.subplots(1, 2, figsize=(15, 5))`) with one axis per metric family and both splits overlaid on each axis is the right default for this kind of dashboard. The _wrong_ default is four panels (train-loss, test-loss, train-acc, test-acc) — you lose the direct train-vs-test comparison that's the whole point of looking.

## What this example doesn't handle — and what its absence teaches

Like the Equinox example, this one is minimal. It has no:

- Gradient accumulation (one batch per step)
- LR scheduling shown explicitly (hidden in the optimizer construction elsewhere)
- Checkpointing
- Multi-device parallelism
- Early stopping based on metrics

What's instructive is _where_ those additions would slot in. Checkpointing in NNX is easier than in Equinox for a specific reason: the mutable-reference model means you checkpoint by serializing one object, and resume by loading into that same object. There's no need to split-and-recombine filtered PyTrees. Conversely, multi-device work in NNX is trickier than in Equinox for the symmetric reason: `pmap` and `shard_map` want pure functions, and NNX has to do more work under the hood to reconstruct a pure view of your mutable modules.

So the tradeoff has a shape: **NNX optimizes for the single-device, single-process, research-script case and pays complexity costs at the boundaries where you leave that regime.** Equinox optimizes for the pure-functional case and pays ergonomic costs at the object-lifecycle boundaries (metrics, optimizers, RNGs).

## Concrete takeaways

The things I'd actually internalize from this example, in priority order:

Mutable-reference frameworks exist as a deliberate alternative to pure-functional ones, not as a mistake — they're a bet that ergonomics beats explicitness for most users. Views are a clean way to handle mode-flag variation without threading a bool through every call site. RNG containers encapsulate the split-advance rule and are worth stealing even into otherwise-functional code. Metric accumulators want three operations, no more. Compound eval-gates usually encode bug-history; read them with that lens. Keying a history dict as flat `{split}_{metric}` strings pays off at plot time.

The overarching meta-lesson is the comparison itself: **when you see two mature frameworks in the same language solve the same problem with opposite design commitments, the right question is never "which is correct?" but "which constraint does each take as non-negotiable?"** For Equinox, it's JAX purity. For NNX, it's PyTorch-like ergonomics. Both then work backward from there to make the rest of the system tolerable.