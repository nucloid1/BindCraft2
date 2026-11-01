# The MLX AlphaFold backend

AlphaFold runs on the Apple GPU through [MLX](https://github.com/ml-explore/mlx) instead of through the metaljax JAX plugin. Everything else (the front, the loss, ProteinMPNN, the filters, the optimizer) is the
repo's own JAX code, on the CPU. It is opt-in and sits beside the other backends; with `BINDCRAFT_BACKEND` unset nothing changes, and no MLX module is imported (checked with `mlx` made unimportable).

Status: tested on one machine (an M3 Max with 48 GB), on whole campaigns of one trajectory per run and short budgets. Read `HANDOFF.md` for what is and is not established.

## Use

```bash
pip install 'bindcraft[mlx]'              # mlx 0.32; the weights are the AlphaFold ones the JAX path already uses (bindcraft fetch-weights)
BINDCRAFT_BACKEND=mlx bindcraft design examples/pdl1_denovo.json
```

- **Design** runs the AlphaFold-Multimer models (`model_N_multimer_v3`), forward and gradient. **Validation** runs the pTM monomer models (`model_1_ptm`, `model_2_ptm`, the default pool) or multimer models, forward only.
  A monomer model has no gradient on this backend (a request for one fails, naming the model), and the monomer models without a predicted-aligned-error head (`model_1` and so on) are refused by name.
  Seeding from given coordinates (`hard_target`, `large_complex_initialization`, and the desperation ladder's first rung, which switches `hard_target` on by itself) works for both families.
- The backend needs an Apple silicon Mac and the `mlx` package; it is refused up front, with the install command, otherwise.
- The parent's JAX runs on the CPU (`import bindcraft` sets `JAX_PLATFORMS=cpu` when the backend is mlx and you have not set it). Measured equal to metal for the whole pipeline (`e2e.md`); `JAX_PLATFORMS=metal` overrides.
- Precision is bfloat16 as on every backend. The MLX model rounds to bfloat16 at every operation, like bfloat16 hardware; the JAX CPU model's bfloat16 does not (XLA keeps float32 between fused operations), so
  the JAX-CPU numbers are somewhat tighter than any real bfloat16 implementation can be (`checkpoints.md`). In float32 the port agrees with the JAX modules to 1e-7 to 1e-5 per stage.

## How it works

The one call that runs AlphaFold inside the jitted loss is replaced by a `jax.custom_vjp` whose forward and backward each call a persistent MLX worker process through `jax.pure_callback`; JAX's own autodiff then yields
forward, loss, backward with the rest untouched. The worker never imports JAX or torch (an earlier port measured MLX about a third slower in a process that had done other CPU-side work). Arrays cross by shared memory,
unlinked after every call. Which model runs is an integer handle passed as a runtime argument (the compile caches are keyed by model family, so a name baked in at trace time would run the wrong model). The worker
keeps the state before the final recycle pass of the last eight forward calls under parent-assigned call ids, so a backward pass does not repeat the recycle passes. The design forward (the custom_vjp forward rule)
also keeps each Evoformer block's input, a tape, so the vjp that follows skips the final pass's Evoformer forward: it runs the backward as a chain (the heads and structure module, the main stack's blocks in reverse,
the extra-MSA stack's, the embeddings), each block recomputed from its input and differentiated. On the GPU the blocks run compiled (`mx.compile`, one trace per shape for all 48 layers; the same arithmetic, fused);
`BINDCRAFT_MLX_COMPILE=0` turns that off. A design model and a validation model in one process share one worker.

## Memory, the GPU lock, the disk, and what happens when something dies

- **Cap.** The worker is stopped by a memory guard at `BINDCRAFT_MLX_CAP_GIB` (default 75% of the machine's memory, 36 GiB of 48), counting MLX's live arrays and its allocator cache. Weights are read lazily, at most five
  models stay resident (least recently used goes first). The allocator cache may hold a quarter of the cap; the design forwards' tapes at most a fifth (`BINDCRAFT_MLX_TAPE_GIB` overrides it, in GiB; the oldest are
  dropped first, and a vjp whose tapes are gone recomputes them).
  The repo's retry ladder (smaller shards, chunked attention) does not apply: the worker's ladder is one rung, and a failure is worded so that `af2.is_out_of_memory` does not match it.
- **One GPU run at a time.** A GPU worker holds an exclusive lock on `~/.cache/esmfold2_mlx/gpu.lock` (`ESMFOLD2_MLX_LOCK` moves it) for the process tree, shared with the other MLX projects; a second run fails at once, naming
  the holder. A CPU worker (`BINDCRAFT_MLX_DEVICE=cpu`, for tests) takes none.
- **Disk.** A run starts only if the disk is below `BINDCRAFT_MLX_MAX_DISK_PCT` percent full (default 90); how much memory other processes hold is not checked. `BINDCRAFT_MLX_MIN_FREE_GB=<gb>` is an explicit, announced
  opt-in for running above the limit with at least that much free; the default is never loosened.
- **Failure is loud.** A worker that dies (its memory guard, the OS) raises an error naming the exit code and the cap; the call that failed is not retried, the next call starts a new worker and registers the models
  again, and there is no fallback to JAX. A campaign whose worker is killed mid-trajectory exits nonzero with `MLX worker` and `exit code -9` in its output (tested). A worker that stops answering without dying is killed
  after `BINDCRAFT_MLX_CALL_TIMEOUT_S` seconds without a reply (default 1800, a net for a wedge rather than a speed limit; `0` turns it off) and the error says it stalled; a frozen worker would otherwise hold the run, and the GPU lock, forever.

## Measured against metaljax (M3 Max, bf16, warm; ratio = metaljax time / MLX time, above 1 means MLX is faster)

| what | residues | MLX | metaljax | ratio | MLX peak memory |
|---|---|---|---|---|---|
| one Evoformer block, forward | 150 / 250 / 400 | 10.0 / 31.1 / 100.8 ms | 12.2 / 31.6 / 96.9 ms | 1.22 / 1.02 / 0.96 | |
| one block, forward + backward | 150 / 250 / 400 | 27.4 / 85.9 / 278.5 ms | 27.1 / 81.9 / 497.5 ms | 0.99 / 0.95 / 1.79 | 1.11 / 2.51 / 7.25 GiB |
| multimer forward, 1 recycle | 250 / 400 | 2.55 / 7.84 s | 5.67 / 23.86 s | 2.22 / 3.04 | |
| design step: forward + vjp | 250 / 400 | 6.18 / 18.84 s | 12.45 / 54.1 s | 2.01 / 2.87 | 6.8 / 16.6 GiB (worker, allocator cache included) |
| trajectory, per gradient round (real campaigns, five seeds) | 195 | 4.48 ± 0.09 s | 8.87 ± 0.18 s | 1.98 | |
| validation forward, monomer, 3 recycles | 80 / 195 / 259 | 0.67 / 3.20 / 6.27 s | 1.22 / 7.03 / 13.15 s | 1.82 / 2.20 / 2.10 | |
| validation refold set, per validation model: 10 candidates x (complex of 195 + binder alone of 80) | | 38.7 s | 82.5 s | 2.13 | |

The two block rows are the block alone, uncompiled, before the speed work (2026-11-01): it ties metaljax, and the larger gains below come from metaljax's cost for the same blocks inside its whole-model graph and from the
speed work (compiled blocks, the vjp skipping the final pass's Evoformer forward, a larger allocator cache; `checkpoints.md`, "Speed work"), which left every trajectory identical: the five seeds' runs
match the earlier ones round for round, ipTM and loss to four decimals. The rows from the multimer forward down were measured after it; the metaljax side is unchanged. Peak memory is MLX's own for the worker (the GPU's
buffers and the allocator cache included); metaljax's peak is not reported because its process RSS (4 to 5 GiB here) does not include them. The refold-set row is built from the
two forward timings (the full pipeline's measured post-trajectory phase, validation and ProteinMPNN and filters together, was 73 s). Sources: `checkpoints.md`, `e2e.md`, `e2e_results/`; scripts `bench_block.py`,
`bench_grad.py`, `bench_forward.py`, `e2e_compare.py` in `scripts/mlx_af2/`, `experiments/before_after.py` for the speed work.

## Testing

`BINDCRAFT_MLX_INTEGRATION=1 pytest tests/mlx_af2` runs the whole suite: the unit tests on the MLX CPU device (they never touch a GPU another run may hold), the gpu-marked tests (end-to-end campaigns) on the GPU under
the lock. `BINDCRAFT_MLX_TEST_DEVICE=gpu` runs every test on the GPU, an accuracy run on the real device (the lock-ownership tests and one chaotic-gradient case are skipped or expected to fail there, each with its
reason in the test). The references the tests are judged against (real inputs, outputs, cotangents and gradients of the JAX path at the model-call seam) are captured by `scripts/mlx_af2/capture_refs.py` into
`~/.cache/bindcraft_mlx_refs`; without them the integration tests fail with the command to run.
