---
title: A Training loop example from equinox's document
type: code-snippets
domain:
- ml/training
---

ref: <https://docs.kidger.site/equinox/examples/mnist/>

## Source Code

```python
def train(
    model: CNN,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> CNN:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: CNN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model
```

## Units

1. **Initialize optimizer state on the array-only subset of a mixed PyTree** — pass the model through a structural filter (`eqx.filter(model, eqx.is_array)`) before handing it to `optim.init`, so only trainable leaves get corresponding state entries. ♻️ reusable
2. **Wrap the full step — loss, grad, optimizer update, apply — inside one JIT region** — define a single `make_step` closure and decorate it with `@eqx.filter_jit`, so the whole optimization step compiles and fuses as one traced graph rather than across many small calls. ♻️ reusable
3. **Define the step function as a closure over the optimizer** — let `make_step` capture `optim` from the enclosing scope instead of threading it as an argument, keeping the JITted signature to just the data that actually changes per step. ♻️ reusable
4. **Shape-annotate array arguments with jaxtyping** — declare input tensors as `Float[Array, "batch 1 28 28"]` / `Int[Array, " batch"]` in the step function signature so batch and structural dimensions are documented at the call site. ♻️ reusable
5. **Compute loss and gradient in one filtered pass with `eqx.filter_value_and_grad`** — use the filter-aware value-and-grad wrapper so differentiation only happens with respect to array leaves of the PyTree, not the static pieces. ♻️ reusable
6. **Re-filter the model when passing it to the optimizer update** — call `eqx.filter(model, eqx.is_array)` again inside `make_step` so `optim.update` sees a params PyTree whose structure matches the gradient PyTree. ♻️ reusable
7. **Apply optimizer updates via `eqx.apply_updates`** — use the Equinox-aware update applier so non-array static fields pass through untouched while array leaves get the update added. ♻️ reusable
8. **Return the updated triple `(model, opt_state, loss)` from the step** — make the JITted step pure by returning every piece of mutated state plus the reported metric, so the outer loop rebinds names rather than mutating in place. ♻️ reusable
9. **Turn a finite dataloader into an infinite stream with a generator that re-yields** — define `while True: yield from trainloader` so the training loop can be driven by step count rather than epoch count. ♻️ reusable
10. **Drive the training loop by step count via `zip(range(steps), stream)`** — bound iteration with `range(steps)` zipped against an infinite iterator so the loop terminates at a fixed step budget regardless of dataset size. ♻️ reusable
11. **Convert PyTorch tensors to NumPy at the JAX boundary** — call `.numpy()` on each batch tensor before handing it to the JIT-compiled step, so JAX sees array inputs instead of torch tensors. ♻️ reusable
12. **Gate periodic logging with `step % print_every == 0 or step == steps - 1`** — combine modulo cadence with an explicit final-step check so the last step always reports even when `steps` is not a multiple of the interval. ♻️ reusable
13. **Unwrap scalar JAX arrays with `.item()` only at print time** — keep metrics as device arrays through the hot loop and materialize Python scalars only when formatting log output. ♻️ reusable
14. **Call a separate `evaluate` function inside the logging branch** — run test-set evaluation only on logging steps rather than every step, amortizing eval cost across the print interval. 📍 context-bound
15. **Return the trained model from the training function** — have `train` return the final `model` rather than mutating a caller-held reference, matching the functional update style of the step. ♻️ reusable

## The central design tension this example resolves

Equinox models are PyTrees that mix **array leaves** (weights, biases — the things you train) with **non-array leaves** (activation functions, shape ints, boolean flags — static config baked into the module). JAX transformations like `grad`, `jit`, and optimizers from `optax` were designed assuming you hand them pure arrays. So the core question every Equinox training loop must answer is: _how do you keep the ergonomics of "the model is one object" while still giving JAX/optax the pure-array view they need?_

The answer this example commits to is **filter at the boundary, not at the core**. The model stays a single `CNN` object everywhere in user code. Whenever it crosses into JAX/optax territory, it passes through `eqx.filter(..., eqx.is_array)` or an `eqx.filter_*` wrapper. This is worth internalizing because it's the same pattern you'll use in every Equinox codebase — the filter calls are not ceremony, they are the seam where two type systems meet.

### The JIT-boundary decision

The single most consequential choice here is wrapping **the entire step** — forward, loss, backward, optimizer update, parameter update — in one `@eqx.filter_jit` region. A natural-looking alternative is to JIT just the forward or just the loss and leave the optimizer step in Python. That alternative is slower, sometimes dramatically, because each un-JITted boundary forces a host-device sync and prevents XLA from fusing across the boundary.

The lesson generalizes: **in JAX, your JIT boundary should enclose the largest pure-function unit of work that runs every step.** For training, that unit is the step, not the forward. The data loader and the logging stay outside because they're impure (I/O, Python printing); everything in between goes inside.

### Purity discipline at the step level

`make_step` takes `(model, opt_state, x, y)` and returns `(model, opt_state, loss_value)`. Nothing is mutated. The outer loop rebinds names: `model, opt_state, train_loss = make_step(...)`. This looks like a stylistic choice but it's load-bearing — JIT requires functional purity to trace correctly, and returning the new state is what makes the function JIT-able in the first place. If you instead tried to mutate `model` in place, you'd either get a trace-time error or, worse, silent staleness where the "updated" model is a traced abstract value that never makes it back to the host.

The broader practice: **treat state as immutable inside transformed regions, and make the outer loop the only place rebinding happens.** Everything inside the transform is a pure function from old state to new state.

### The closure-over-optimizer move

`make_step` is defined _inside_ `train`, closing over `optim`. This isn't aesthetic — `optax.GradientTransformation` objects contain Python-level config (learning rate schedules, hyperparameters) that you want frozen into the trace rather than passed as a dynamic argument. Closing over it means XLA sees `optim.update` as a fixed computation; passing it as an argument would either fail (it's not a PyTree of arrays) or force you to mark it static.

The generalization: **in JAX, the natural way to distinguish "this is config, bake it in" from "this is data, trace it" is lexical scope.** Closure = static. Argument = dynamic. This is a much lighter-weight mechanism than `static_argnums` and reads more naturally at the call site.

### The step-budget iteration pattern

Instead of the canonical `for epoch in range(epochs): for batch in loader:` double loop, this example uses `zip(range(steps), infinite_trainloader())`. Three things fall out of this choice:

First, the training budget is expressed in the unit that actually matters for compute and comparison — **steps**, not epochs. Epochs are a property of your dataset size; steps are a property of your training run. Papers report steps; wall-clock scales with steps; learning rate schedules are defined over steps.

Second, the `while True: yield from loader` generator cleanly decouples dataset length from run length. You can train for fewer steps than one epoch, or for many epochs' worth, with no branching.

Third, `zip` with a finite range gives you a natural termination condition without a manual counter or `break`.

This pattern is worth stealing wholesale; it's one of those small idioms that quietly improves every training script.

### Logging that respects the hot path

`.item()` is called only inside the print branch. This matters because `.item()` forces a device→host sync — it blocks until the computation actually finishes on the accelerator. If you call `.item()` every step (even just to store a loss history), you serialize your training loop against the device and lose much of the async-dispatch benefit JAX gives you.

The practice: **keep metrics as device arrays through the hot loop; materialize to Python scalars only at log boundaries.** If you need a running loss history, append the device array and `.item()` them in a batch at the end, or accumulate on-device.

The `step % print_every == 0 or step == steps - 1` guard is a smaller but worth-noticing detail: the `or` clause ensures the final step always logs, which you want for end-of-run metrics regardless of whether `steps` divides evenly by the interval.

### The PyTorch-dataloader-to-JAX bridge

Using `torch.utils.data.DataLoader` with JAX is pragmatic rather than ideological. PyTorch's dataloader is battle-tested (workers, prefetching, shuffling, collation) and JAX historically hasn't had a first-class equivalent — grain exists now but the torch loader is still everywhere. The `.numpy()` call at the boundary is the whole bridge.

The lesson: **don't feel obligated to live inside one framework's universe.** Use the best tool for each stage and convert at the seam. The conversion here is cheap (shared-memory numpy view of the torch tensor) so there's no performance argument against it.

## What's not here — and why that's instructive too

Notice what the example doesn't do: it doesn't manually `pmap` or `jit` sub-functions, doesn't thread RNG keys, doesn't handle checkpointing, doesn't track per-parameter stats. The function is minimal on purpose — it's the smallest complete training loop. When you compare this to a production loop, the additions you'd make are mostly _at the same boundaries this skeleton already identifies_: RNG threads through `make_step`'s signature, checkpointing hooks the outer loop at log intervals, multi-device splits replace `filter_jit` with `filter_pmap` or a `shard_map`. The skeleton scales; the primitives are the same.

## Concrete takeaways to internalize

The biggest transferable lessons, in order of how often they'll come up:

Filter at the JAX/optax boundary, never in between. One JIT region per step, as large as you can make it while staying pure. Closure for static config, argument for dynamic data. Steps as the budget unit, with an infinite loader generator. Device arrays through the hot loop, `.item()` only at log time. Return-the-new-state purity, with the outer loop as the only rebinding site.

Those six ideas carry across essentially every JAX training codebase you'll write or read.
