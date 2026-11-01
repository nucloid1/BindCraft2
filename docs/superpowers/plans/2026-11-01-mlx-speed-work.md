# MLX backend speed work (compiled block closures, explicit reverse loop, forward tape) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use checkbox syntax.

**Goal:** make the MLX design step faster by (A) compiling the Evoformer block's forward and a per-block vjp closure and (B) keeping the design forward call's block inputs so the vjp does not repeat the final pass, with identical numerics and the same refusals and failure modes as before.

**Architecture:** the two block stacks (extra-MSA, main) become a stack-level `mx.custom_function` whose vjp is an explicit reverse loop over per-block vjp closures (compiled on the GPU, eager on the CPU); the forward records each block's input in a tape that the worker keeps per call id under a byte budget, a miss recomputes (correctness never depends on a hit).

**Spec:** `docs/mlx_backend/checkpoints.md`, section "Speed headroom: what the investigation found" (items A and B; the gain is an upper bound); Ethan said "build the speed work" and "and merge to main" on 2026-11-01.

## Global Constraints

- Numerics unchanged: gradients and outputs of the new path equal the old path (fp32 on the CPU tightly; bf16 on the GPU within one bf16 ulp; the three existing dropout tests pass on the new path).
- Every differentiable array is an ARGUMENT of the custom function, never captured by its closure (a captured one silently becomes a constant).
- Compile is GPU-only by default, with one kill-switch (`BINDCRAFT_MLX_COMPILE=0`) and nothing more; dropout uses explicit keys only.
- Nothing pushed; never modify `~/RFD4-Proteina`; one GPU run at a time via the shared lock; disk opt-in `BINDCRAFT_MLX_MIN_FREE_GB=20`; commit trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Merge = fast-forward of local `main` (an ancestor of `mlx-backend`); no `git pull`, no push; only after BOTH suites are green (default CPU run, ~55 min; `BINDCRAFT_MLX_TEST_DEVICE=gpu`, ~13 min) and a fresh review of the speed diff.

## Review Focus

- A captured-by-closure differentiable array (silent zero gradient): pinned by gradient comparison against autodiff through plain blocks.
- A tape from another call, key, model, or shape: refused or ignored (miss), never used.
- Tape memory: byte budget, oldest evicted first, worker peak stays under the cap at 400 residues with several states.
- Compile per new shape: first-call cost measured and reported against the per-step saving.
- The monomer validation forward shares `evoformer_stack`: its parity and speed measured, not assumed.

### Task S0: go/no-go probe (GPU, scratch)

- [ ] Time compiled block forward and block-vjp closures with the layer's parameters passed in as a list and a dropout key as an input: traces per shape (layer 0 traces, layer 1 does not), first-call compile time per shape for forward and vjp, main and extra stacks, warm stack sweeps against the current checkpointed stack, at 250 and 400 residues. Stop and report if compile time eats the gain.

### Task S1: explicit reverse loop, uncompiled (CPU tests)

- [ ] Tests first (`tests/mlx_af2/test_block_stack.py`): gradient equals autodiff through plain blocks (fp32, dropout off and on); extra stack returns only `pair` or its unused `msa` cotangent is handled; tape records one input per block and the outputs; vjp from a tape equals vjp without one; a tape skips the stack's forward (call count); a tape of the wrong shape is refused.
- [ ] `evoformer_stack` (callers: embeddings.py x2, monomer.py forward only, tests) gains the custom-vjp reverse loop for `remat=True`; `remat=False` stays a plain loop and can record a tape.

### Task S2: compile behind a GPU-only default

- [ ] `ArrayParams` serves a block's parameters from a list; compiled forward and vjp closures keyed by stack, config, subbatch and dropout flag; gpu-marked tests: compiled equals uncompiled (bf16 within one ulp, fp32 tight), kill-switch and CPU default.

### Task S3: the tape in the model, bridge and worker

- [ ] `MultimerModel.forward_with_state` can return the tapes, `vjp(..., tapes=)` consumes them; `forward` request carries `retain` (only the bridge's forward rule sets it); worker keeps tapes in `calls` under a byte budget (oldest first), validated like `before` plus a shape check; stats report tape hits, misses and bytes. Tests: hit equals miss gradients, eviction under a tiny budget still correct, hit skips the forward, primal and validation forwards do not retain.

### Task S4: measure in one quiet session

- [ ] Pre-change HEAD (`ff897cc`) and the new code back to back: `bench_grad` and `bench_forward` (multimer and monomer) at 250 and 400, worker peak against the cap, first-call cost at a new length; repeat the allocator-cache experiment after A and B and change the default only if repeats confirm it; report realized against projected.

### Task S5: both suites green, fresh review, docs, merge

- [ ] Default CPU suite and GPU-mode suite green; fresh review of the speed diff on the most capable model; fix pass; HANDOFF, checkpoints.md, e2e/README numbers and memory updated; fast-forward `main`.
