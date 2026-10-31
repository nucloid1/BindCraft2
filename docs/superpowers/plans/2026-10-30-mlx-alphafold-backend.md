# MLX AlphaFold Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run BindCraft2's AlphaFold forward pass and its sequence gradient on the Apple GPU through MLX in a worker process, as an opt-in backend beside the existing JAX ones.

**Architecture:** `recycled_alphafold_outputs`, the one call that runs AlphaFold inside the jitted loss, becomes a JAX `custom_vjp` whose forward and backward call a persistent MLX worker through `jax.pure_callback`; the front, loss, optimizer, ProteinMPNN and filters stay JAX (on the CPU for this backend). The MLX model is a faithful functional port of the vendored haiku model, gated stage by stage against the repo's own JAX path on captured real inputs.

**Tech Stack:** Python, MLX 0.32.x, JAX 0.11 (CPU), numpy, shared memory between processes, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-mlx-alphafold-backend-design.md` (branch `mlx-backend`, off `apple-silicon-metal`; commit locally, push nothing).

## Global Constraints

- Base: `mlx-backend` was merged with upstream `b6fb3ae` on 2026-10-30 (see the design's "Base" note); all names and anchors are against that code (`large_complex_initialization`, not `bigbang_initialization`).
- Memory ladder: `executable_fits` measures a JAX executable that does not contain the MLX model, so under this backend memory fitting is the worker's job: the subclass overrides `subbatch_ladder` to one rung and the worker chunks by residue count. A worker memory failure's message must not match `bindcraft.af2.is_out_of_memory` (`resource_exhausted`, `out of memory`), or the repo's JAX-specific fallback would retry a computation that cannot succeed; it must still name the cause and the cap.
- Tolerances (fp32, MLX model run in fp32 against JAX fp32 on the CPU): per-stage relative error 1e-5 where a stage is a plain composition of matmuls, norms and gates; composite stages (full forward outputs, positions) 1e-3; sequence-gradient cosine at least 0.999.
- bf16 (the default, as today) is judged against JAX's own spread: the MLX bf16 result must lie within the relative error JAX bf16 has against JAX fp32 on identical inputs. Trajectory-level checks compare distributions over several seeds, never a single run.
- No silent fallback: an unimplemented model family is refused by name; a dead worker or a cap hit stops the run with a message naming the cause and the cap.
- The MLX code is unreachable unless `BINDCRAFT_BACKEND=mlx`; the CUDA, ROCm, oneAPI and metaljax paths stay as they are and are the parity reference.
- The worker never imports `jax` or `torch`. VERIFIED 2026-10-30: `bindcraft`, `bindcraft.af`, `bindcraft.af.alphafold`, `...common.residue_constants` and `...model.config` import without jax or haiku, so the worker lives inside the package and imports them directly.
- One GPU run at a time through the shared lock `~/.cache/esmfold2_mlx/gpu.lock`, acquired once per process tree (the owner variable `ESMFOLD2_MLX_LOCK_OWNER`, as in the earlier ports, lets child processes and the second model built in the same process pass); disk under 90% before real-model runs unless `BINDCRAFT_MLX_MIN_FREE_GB` is set; worker memory cap default 28 GiB (`BINDCRAFT_MLX_CAP_GIB`).
- Pin `mlx>=0.32,<0.33` (tested 0.32.2). AlphaFold parameters come from `~/.cache/bindcraft/alphafold` (multimer v3) and `~/.cache/colabfold/params` (pTM); `BINDCRAFT_AF2_PARAMS` overrides.
- No stop rule: speed is measured and reported at the named checkpoints and never stops the work.
- Phase-2 trigger (Ethan, 2026-10-30: "if cpu jax is a bottleneck, port it; otherwise the plan is fine as written"): if the CPU-side JAX work (front, loss, ProteinMPNN, filters) exceeds 20% of the MLX backend's wall time, per step in the hot loop or per trajectory, the front and the loss (and then ProteinMPNN and the filters, as measured) are ported to MLX in a second plan, brainstormed first; otherwise they stay JAX. The 20% threshold is the author's call.
- Never modify `~/RFD4-Proteina`. The worker, memory guard, GPU lock and preflight patterns are adapted by copying from `~/RFD4-Proteina-MLX/rfd4_proteina_mlx/{worker,guard,gpu_lock,preflight}.py`, never imported.
- Style: match BindCraft2: terse, long lines, `#comment` with no space, descriptive snake_case, no docstring sprawl. Every commit message ends with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; chain commits after the test run with `&&`.
- Tests: `.venv/bin/python -m pytest <path> -q`; tests that need captured references or the GPU run only with `BINDCRAFT_MLX_INTEGRATION=1` and the GPU ones hold the shared lock.

## Review Focus

1. A float feature other than `msa_feat` and `target_feat` depends on the sequence, so its gradient is silently dropped (Task 17 checks it with a `jax.jvp` of the front).
2. Padded and dead residues (length buckets of 32, `seq_mask` 0) leak into valid positions or into pLDDT, PAE and ipTM (Tasks 14 and 17: identical valid-position results with extra padding).
3. The worker dies, hits its cap or stalls inside a jitted call: the error must name the cause and the cap, the run must stop, no shared memory may be left behind, and nothing may fall back (Tasks 3, 15 and 17).
4. The design model is drawn at random each step (`_sample_design_model`) and one jit execution calls two states (complex and binder alone) with different shapes; weights are per model, buffers per shape, and the design and validation models share one worker and one lock (Tasks 3, 15 and 17).
5. Recycling and randomness: the `prev` carry, `num_recycle` 0 to 3, `initial_atom_pos` updating under `large_complex_initialization`, and dropout determinism per key and flag (Tasks 14 and 17).

## File Structure

New package `bindcraft/af/mlx_af2/`: `weights.py` (npz loader, `Params`), `layers.py`, `pair_ops.py`, `msa_ops.py`, `evoformer.py`, `embeddings.py`, `templates.py`, `geometry.py`, `structure.py`, `heads.py`, `model_multimer.py`, `outputs.py` (`OUTPUT_KEYS`, `output_spec`), `monomer.py`, `monomer_structure.py`, `worker.py`, `guard.py`, `gpu_lock.py`, `preflight.py`, `bridge.py` (custom_vjp), `predictor.py`. New `bindcraft/backend.py`. Modified: `bindcraft/__init__.py`, `bindcraft/af2.py` (three hooks), `bindcraft/campaign.py` (two constructor calls), `bindcraft/selfcheck.py`, `pyproject.toml`, `docs/installation.md`. Tests under `tests/mlx_af2/`; scripts under `scripts/mlx_af2/`; references in `~/.cache/bindcraft_mlx_refs/` (`BINDCRAFT_MLX_REFS` overrides).

---

### Task 1: Backend switch, dependencies and test scaffolding

**Files:** Create `bindcraft/backend.py`, `bindcraft/af/mlx_af2/__init__.py`, `tests/conftest.py`, `tests/mlx_af2/test_backend.py`. Modify `bindcraft/__init__.py` (`select_apple_gpu`), `bindcraft/selfcheck.py`, `pyproject.toml`.

**Interfaces:** Produces `resolve_backend() -> str` (`'jax'` or `'mlx'`; `ValueError` naming both on anything else); `BINDCRAFT_BACKEND=mlx` makes `select_apple_gpu` default `JAX_PLATFORMS` to `cpu` (an explicit `JAX_PLATFORMS` still wins, so `metal` keeps the parent on metaljax); extras `mlx = ["mlx>=0.32,<0.33", "jax>=0.11,<0.12"]` and `test = ["pytest>=8"]`; conftest fixtures `refs_dir`, `params_dir`, markers `integration` and `gpu` (the latter holds the shared lock for the session, exporting the owner variable so a worker passes, and fails, not skips, when it is held).

- [ ] **Step 1: Ask Ethan, then install** `.venv/bin/pip install "mlx>=0.32,<0.33" pytest` into `~/BindCraft2/.venv` (neither is installed now). Record `pip list` before and after in the ledger. Expected: `mlx 0.32.x`, `pytest 8.x`, and `jax`, `numpy` versions unchanged.
- [ ] **Step 2: Write failing tests** in `tests/mlx_af2/test_backend.py`: `test_default_backend_is_jax`; `test_unknown_backend_names_the_accepted_values`; `test_mlx_backend_defaults_jax_to_cpu` (subprocess: `BINDCRAFT_BACKEND=mlx python -c "import bindcraft, os; print(os.environ['JAX_PLATFORMS'])"` prints `cpu`); `test_explicit_platform_wins` (same with `JAX_PLATFORMS=metal` prints `metal`); `test_jax_backend_leaves_platform_logic_alone` (without the variable the output equals what the current code gives).
- [ ] **Step 3: Run** `pytest tests/mlx_af2/test_backend.py -q`. Expected: FAIL (no `bindcraft.backend`).
- [ ] **Step 4: Implement** the above (the red run is Step 3); add `mlx` to `selfcheck` as an optional module reported only when the backend is `mlx`.
- [ ] **Step 5: Run** the same command. Expected: 5 passed. **Commit** `backend: BINDCRAFT_BACKEND switch, mlx and test extras`.

### Task 2: Behaviour-preserving hooks in `af2.py`

**Files:** Modify `bindcraft/af2.py`, `bindcraft/campaign.py` (`build_validation_model` and `campaign_design_model`). Create `tests/mlx_af2/cases.py`, `scripts/mlx_af2/capture_golden_jax.py`, `tests/mlx_af2/test_hooks.py`.

**Interfaces:** Produces on `AlphaFoldDesignModel`: `_load_parameters(self, model_name: str, data_dir: str | None) -> object` (the `get_model_haiku_params(..., fuse=True)` call and its error), `_model_outputs(self, alphafold_runner, model_parameters, key, model_inputs: dict) -> dict` (no model name and no subbatch size: the per-state runner carries the subbatch and the attention backend, and both compile caches are keyed by model family, so which model runs must come from `model_parameters`, the one argument that differs per model) (calls `recycled_alphafold_outputs`), used at the two closure sites in `_compiled_complex_prediction` and `_compiled_sequence_gradients`; `bindcraft.backend.design_model_class() -> type` (lazy import, returns `AlphaFoldDesignModel` for `jax`), used by `campaign.py`. `cases.design_inputs(case: str)` for `"small"` (about 100 to 140 total residues: binder plus a crop of `settings/target/structures/hPDL1.pdb`) and `"medium"` (about 250 to 300), built with the repo's own `initialize_design_trajectory`; `cases.front_arguments(case: str, state: str) -> dict[str, np.ndarray]` returns the concrete arrays (`sequence`, `flags`, `atoms`, `atom_mask`, `residue_index`, `asym_id`, `entity_id`, `seq_mask`) that `sequence_gradients` feeds to the front, built with the same helper calls.

- [ ] **Step 1: Capture the golden** with the unrefactored code: `JAX_PLATFORMS=cpu .venv/bin/python scripts/mlx_af2/capture_golden_jax.py` runs `predict` and `sequence_gradients` on `small` (fixed keys, dropout off) and saves metrics, positions, gradients and loss to the refs dir. Expected: file written, all arrays finite and non-constant.
- [ ] **Step 2: Write the failing test** `test_jax_path_is_bit_identical_after_hooks` (`np.array_equal` against the golden) and `test_campaign_builds_the_jax_class_by_default`; the first passes before the refactor by construction, so also write `test_hooks_exist_and_are_called` (spy on `_model_outputs` and `_load_parameters`; both are called during `predict` and `sequence_gradients`), which fails now.
- [ ] **Step 3: Refactor** the two sites and `__init__`'s parameter loading onto the hooks; nothing else changes.
- [ ] **Step 4: Run** `JAX_PLATFORMS=cpu BINDCRAFT_MLX_INTEGRATION=1 pytest tests/mlx_af2/test_hooks.py -q`. Expected: 3 passed. **Commit** `af2: model-output and parameter-loading hooks, no behaviour change`.

### Task 3: Prove the seam with a stub remote

**Files:** Create `bindcraft/af/mlx_af2/bridge.py`, `bindcraft/af/mlx_af2/outputs.py`, `tests/mlx_af2/stub_remote.py`, `tests/mlx_af2/test_seam.py`.

**Interfaces:** Produces `outputs.OUTPUT_KEYS: tuple[str, ...]` (the flattened names the tail reads: `structure_module/final_atom_positions`, `structure_module/final_atom_mask`, `predicted_lddt/logits`, `predicted_aligned_error/logits`, `predicted_aligned_error/breaks`, `distogram/logits`, `experimentally_resolved/logits`) and `outputs.output_spec(length: int) -> dict[str, jax.ShapeDtypeStruct]` (provisional dtypes here, checked against the JAX path in Task 4); `bridge.mlx_model_outputs(remote, model_handle, key, model_inputs: dict, num_recycle: int, subbatch_size: int | None) -> dict`, a `jax.custom_vjp` over `jax.pure_callback`: `model_handle` is the traced integer array `_load_parameters` returned and the hook receives as `model_parameters` (a model name baked in at trace time would serve the wrong model on a compile-cache hit, since the caches are keyed by family); `key` and every entry of `model_inputs`, `use_dropout` included, are traced arrays inside the jitted loss; integer inputs get `float0` cotangents; `DIFF_KEYS = ('msa_feat', 'target_feat')`; a remote error surfaces as a `RuntimeError` naming the cause. The remote protocol it calls: `forward(model_handle: int, key: np.ndarray, features: dict[str, np.ndarray], num_recycle, subbatch_size) -> tuple[dict[str, np.ndarray], int]` and `vjp(model_handle: int, call_id, key, features, num_recycle, subbatch_size, cotangents, diff_keys) -> dict[str, np.ndarray]`. `stub_remote.StubRemote` is a pure-numpy "model" (outputs are fixed linear maps of `msa_feat` and `target_feat`, with a hand-written vjp) running in a child process, plus `jax_stand_in(model_inputs) -> dict` with the same math in JAX.

- [ ] **Step 1: Write failing tests** (JAX on the CPU, a test-only subclass of `AlphaFoldDesignModel` that overrides the Task 2 hooks with the bridge and the stub, `small` case): `test_gradient_reaches_the_sequence_logits` (the gradients through the bridge equal those with `jax_stand_in` in place of the bridge, to 1e-6, on every chain), `test_two_states_in_one_jit_call` (complex and binder alone, both reach the remote, gradients match the stand-in), `test_integer_inputs_get_float0_cotangents`, `test_compile_only_never_calls_the_remote` (zero remote calls from `sequence_gradients(compile_only=True)` and from `predict(compile_only=True, compile_blocking=False)`, the two calls `compile_next_length_bucket` makes from its thread), `test_remote_error_surfaces_naming_the_cause` (the stub raises with a message and a cap; the JAX-side exception text contains both), `test_key_and_dropout_flag_arrive_as_values` (changing the jit arguments changes what the stub records, with no retracing), `test_model_identity_is_a_runtime_argument` (one compiled function called with two different handles reaches the stub with both, with no retrace), `test_stub_runs_in_another_process` (its pid differs).
- [ ] **Step 2: Run** `JAX_PLATFORMS=cpu pytest tests/mlx_af2/test_seam.py -q` (expected FAIL: no bridge), **implement**, **run** again. Expected: 7 passed; if any fail for a reason that cannot be fixed inside the bridge, stop and ledger it as a finding against the design before any porting starts. - [ ] **Step 3: Checkpoint 0** (CPU only, reported, never a stop): with the stub answering instantly, time one `sequence_gradients` step at 150, 250 and 400 residues, which is the whole JAX-side cost (front, loss, callbacks, array handling), and set it beside the full step time of the metaljax path measured warm; create `docs/mlx_backend/checkpoints.md` with the table. This is the first number for the phase-2 trigger. **Commit** `mlx_af2: custom_vjp bridge proven against a stub remote`.

### Task 4: Seam captures, tolerances and output contract

**Files:** Create `scripts/mlx_af2/capture_refs.py`, `tests/mlx_af2/helpers.py`, `tests/mlx_af2/test_refs.py`.

**Interfaces:** Produces `helpers.rel_err(a, b) -> float`, `cosine(a, b) -> float`, `assert_live(a)` (finite, not constant, not all zeros); Consumes `outputs.OUTPUT_KEYS` and `outputs.output_spec` from Task 3. Refs layout: `<case>/<variant>/call<i>.npz` holding `in/<feature>`, `key`, `num_recycle`, `out/<flat key>`, `cot/<flat key>` (seeded random cotangents on the float outputs), `grad/msa_feat`, `grad/target_feat` (from `jax.vjp`), with variants `fp32` (the runner built with `bfloat16 = False`), `bf16` (as shipped), each with dropout off; plus `spread.json` (per output, relative error of bf16 against fp32; gradient cosine bf16 against fp32).

- [ ] **Step 1: Write failing tests** `test_refs_are_live_and_complete` (every array passes `assert_live`, every `OUTPUT_KEYS` entry present, shapes consistent between variants), `test_tail_reads_only_output_keys` (run `alphafold_prediction_metrics` and both predict closures on a dict wrapper that records accessed keys; the set is a subset of `OUTPUT_KEYS`), `test_spread_is_recorded_and_nonzero`, `test_output_spec_matches_jax_dtypes_and_shapes` (fixes the dtypes Task 3 left provisional).
- [ ] **Step 2: Implement** `capture_refs.py`: build the model inputs eagerly (call `prepare_design_sequence_features` and `alphafold_input_features` on `cases.front_arguments`, no jit, so every recorded value is a concrete array, not a tracer), then run `recycled_alphafold_outputs` through the JAX runner jitted on them for `small` and `medium`; build the fp32 runner by copying the eight config lines of `_alphafold_runner` with `bfloat16 = False`; for each recorded call compute the cotangent and `jax.vjp` reference.
- [ ] **Step 3: Run** `JAX_PLATFORMS=cpu .venv/bin/python scripts/mlx_af2/capture_refs.py --cases small medium` (minutes on the CPU), then `BINDCRAFT_MLX_INTEGRATION=1 pytest tests/mlx_af2/test_refs.py -q`. Expected: 3 passed. **Commit** `mlx_af2: seam capture script, helpers and output contract`.

### Task 5: Weight loader

**Files:** Create `bindcraft/af/mlx_af2/weights.py`, `tests/mlx_af2/test_weights.py`.

**Interfaces:** Produces `load_params(model_name: str, params_dir: str, dtype: mx.Dtype = mx.bfloat16) -> Params`; `Params.get(path: str) -> mx.array` (cast to `dtype`, mirroring the haiku `bfloat16_getter`: float32 stored, bfloat16 on use), `Params.scope(prefix: str) -> Params`, `Params.layer(i: int) -> Params` (slice of the stacked leading axis), `Params.assert_all_used()`, `Params.unused() -> list[str]`. Reads the flat `.npz` (keys like `alphafold/alphafold_iteration/evoformer/evoformer_iteration/triangle_multiplication_outgoing/center_norm//scale`, stacked over 48 layers) and applies the fusing of `utils.flat_params_to_haiku(fuse=True)`.

- [ ] **Step 1: Write failing tests**: `test_every_tensor_is_mapped_once` (the loader's own tensor-to-scope table covers all 330 tensors of every local multimer model with no duplicates; that the model consumes them all is asserted in Task 14); `test_fused_arrays_equal_the_jax_loaders` (flatten `get_model_haiku_params(fuse=True)` and compare bit for bit); `test_missing_key_names_the_key`; `test_unknown_model_is_refused_by_name`; `test_bf16_cast_on_get_not_on_load`.
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected after: all pass. **Commit** `mlx_af2: npz weight loader with fusing and accounting`.

### Task 6: Core layers

**Files:** Create `bindcraft/af/mlx_af2/layers.py`, `scripts/mlx_af2/jax_units.py`, `tests/mlx_af2/test_layers.py`.

**Interfaces:** Produces functional ops, arguments in the order of the JAX `__call__` minus `is_training` and `safe_key`: `linear(p, x, *, num_input_dims=1)`, `layer_norm(p, x, eps=1e-5)`, `attention(p, q_data, m_data, mask, nonbatched_bias, cfg, *, subbatch_size=None)`, `global_attention(p, q_data, m_data, q_mask, cfg)`, `transition(p, act, mask, cfg)`; `jax_units.run_unit(module_cls, scope_name, params_slice, *inputs, global_config)` runs the vendored haiku module under `hk.transform` with weights sliced from layer 0 of the npz and returns numpy.

- [ ] **Step 1: Write failing tests** (fp32, gate 1e-5, inputs from the `small` seam captures or seeded normals of the right scale): `test_linear_matches_haiku`, `test_layer_norm_matches_haiku`, `test_attention_with_gating_and_bias_matches`, `test_attention_subbatch_equals_unbatched`, `test_global_attention_matches`, `test_transition_matches`; each uses `assert_live` on the reference.
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** (sources: `modules.py`, `common_modules.py`) Expected: all pass. **Commit** `mlx_af2: linear, norm, attention and transition layers`.

### Task 7: Pair-stack operations

**Files:** Create `bindcraft/af/mlx_af2/pair_ops.py`, `tests/mlx_af2/test_pair_ops.py`.

**Interfaces:** `triangle_multiplication(p, left_act, left_mask, cfg, *, outgoing: bool)`, `triangle_attention(p, pair_act, pair_mask, cfg, *, starting: bool)`, consuming `layers` and `Params` from Tasks 5 and 6.

- [ ] **Step 1: Write failing tests** (fp32, 1e-5, layer 0 against `modules.TriangleMultiplication` and `TriangleAttention`): `test_triangle_multiplication_outgoing`, `..._incoming`, `test_triangle_attention_starting`, `..._ending`, `test_fused_projection_weights_are_used` (the unfused-weights result equals the fused one), `test_masked_positions_do_not_leak` (changing masked-out inputs leaves valid outputs identical).
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass. **Commit** `mlx_af2: triangle multiplication and attention`.

### Task 8: MSA-stack operations

**Files:** Create `bindcraft/af/mlx_af2/msa_ops.py`, `tests/mlx_af2/test_msa_ops.py`.

**Interfaces:** `msa_row_attention_with_pair_bias(p, msa_act, msa_mask, pair_act, cfg)`, `msa_column_attention(p, msa_act, msa_mask, cfg)`, `msa_column_global_attention(p, msa_act, msa_mask, cfg)`, `outer_product_mean(p, act, mask, cfg)`.

- [ ] **Step 1: Write failing tests** (fp32, 1e-5): one per op against the vendored module at layer 0 (global attention and column attention at the extra-MSA widths too), plus `test_single_row_msa_matches` (the design uses one MSA row and one extra row: assert a one-row input matches the JAX result, including the global-attention row mean).
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass. **Commit** `mlx_af2: MSA attention operations and outer product mean`.

### Task 9: Evoformer iteration and stacks (checkpoint 1: speed)

**Files:** Create `bindcraft/af/mlx_af2/evoformer.py`, `tests/mlx_af2/test_evoformer.py`, `scripts/mlx_af2/bench_block.py`.

**Interfaces:** `evoformer_iteration(p_layer, msa_act, pair_act, msa_mask, pair_mask, cfg, *, dropout_key=None, is_extra=False, subbatch_size=None) -> tuple[mx.array, mx.array]`; `evoformer_stack(p, msa_act, pair_act, msa_mask, pair_mask, cfg, *, num_block: int, dropout_key=None, remat=True, subbatch_size=None) -> tuple[mx.array, mx.array]` (loop over layers of the stacked weights, `mx.checkpoint` per block when `remat`). Dropout follows `modules.dropout_wrapper` (rates and shared broadcast dims from `cfg`), off when `dropout_key is None`.

- [ ] **Step 1: Write failing tests** (fp32): `test_iteration_matches_layer0_and_layer47` (1e-5 per block), `test_stack_of_48_matches_the_representations` (composite 1e-3 against the JAX `representations` of the seam capture), `test_extra_msa_stack_of_4_matches`, `test_chunked_equals_unchunked` (`subbatch_size=4` equals `None`), `test_remat_gradient_equals_plain_gradient` (`mx.grad` with and without `mx.checkpoint`), `test_dropout_off_is_deterministic_and_on_is_keyed` (same key same output, different key different output, keep fraction near `1 - rate`).
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass.
- [ ] **Step 3: Checkpoint 1** (GPU lock held): `python scripts/mlx_af2/bench_block.py` times one Evoformer block forward and forward-plus-backward at 150, 250, 400 residues in bf16, against the JAX metaljax path (`JAX_PLATFORMS=metal`, warm second run in the process), with peak memory; append to `docs/mlx_backend/checkpoints.md`. Reported, never a stop. **Commit** `mlx_af2: Evoformer iteration, stacks, remat and chunking`.

### Task 10: Multimer embeddings, recycling and extra-MSA stack

**Files:** Create `bindcraft/af/mlx_af2/embeddings.py`, `tests/mlx_af2/test_embeddings.py`.

**Interfaces:** `embed_inputs(p, features: dict[str, mx.array], prev: dict[str, mx.array], cfg) -> Embedded` (a small dataclass: `msa_act, pair_act, msa_mask, pair_mask`), covering `EmbeddingsAndEvoformer.__call__` of `modules_multimer.py` up to the template and Evoformer stages: preprocessing of target and MSA features, left and right single projections, relative-position encoding with `asym_id`, `entity_id`, `sym_id` and the cyclic `offset`, the recycle embedding (`prev_msa_first_row`, `prev_pair`, `prev_pos`), and `extra_msa_stack(p, extra_msa_act, pair_act, extra_msa_mask, pair_mask, cfg, *, dropout_key=None, subbatch_size=None) -> mx.array` (the updated pair activations; consumes Task 9). `use_dgram` is False in this repo.

- [ ] **Step 1: Write failing tests** (fp32, intermediates recorded with `hk.experimental.intercept_methods` while applying the JAX model un-jitted, concrete outside the scanned stacks; where a stage cannot be intercepted, split it with `jax_units`): `test_preprocessing_matches`, `test_relative_position_encoding_with_cyclic_offsets` (a cyclic-peptide case: nonzero `offset`), `test_recycle_embedding_with_and_without_prev`, `test_extra_msa_stack_output_matches`.
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass. **Commit** `mlx_af2: multimer embeddings, recycle embedding and extra-MSA stack`.

### Task 11: Multimer template embedding

**Files:** Create `bindcraft/af/mlx_af2/templates.py`, `tests/mlx_af2/test_templates.py`.

**Interfaces:** `template_embedding(p, query_embedding, template_features: dict, padding_mask_2d, cfg, *, dropout_key=None) -> mx.array` covering `TemplateEmbedding`, `SingleTemplateEmbedding`, `TemplateEmbeddingIteration` and `template_embedding_1d`; the target is a fixed template with the `target_flexibility` mask from `target_template_features`.

- [ ] **Step 1: Write failing tests** (fp32): `test_single_template_embedding_matches`, `test_template_pair_stack_iteration_matches`, `test_template_embedding_full_matches` (1e-3), `test_flexible_target_mask_changes_the_result_as_in_jax` (`target_flexibility` 0.3 against the JAX result).
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass. **Commit** `mlx_af2: multimer template embedding`.

### Task 12: Geometry, invariant point attention and fold iteration

**Files:** Create `bindcraft/af/mlx_af2/geometry.py`, `bindcraft/af/mlx_af2/structure.py` (first part), `tests/mlx_af2/test_structure_backbone.py`.

**Interfaces:** `geometry`: `Vec3`, `Rot3`, `Rigid3` as plain arrays-of-components with `apply`, `compose`, `invert` matching `geometry/{vector,rotation_matrix,rigid_matrix_vector}.py` (no dataclass machinery from JAX); `structure`: `quat_rigid(p, act)`, `point_projection(p, act, rigid, ...)`, `invariant_point_attention(p, inputs_1d, inputs_2d, mask, rigid, cfg)`, `fold_iteration(p, activations, aatype, ...)` per `folding_multimer.py` (`QuatRigid`, `PointProjection`, `InvariantPointAttention`, `FoldIteration`, `make_backbone_affine`, `generate_monomer_rigids`'s backbone part). Softmax and point math in fp32 where the JAX code casts explicitly.

- [ ] **Step 1: Write failing tests** (fp32, 1e-5): `test_rigid_compose_apply_invert_match_jax` (seeded random rigids), `test_ipa_matches` (`small` seam, layer 0 of the structure module), `test_quat_rigid_update_matches`, `test_fold_iteration_matches` (first and last of 8 iterations, positions 1e-4 given the sequence of 8).
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass. **Commit** `mlx_af2: rigid geometry, IPA and fold iteration`.

### Task 13: Sidechains, atom positions and the structure module

**Files:** Modify `bindcraft/af/mlx_af2/structure.py`. Create `tests/mlx_af2/test_structure_full.py`.

**Interfaces:** `multi_rigid_sidechain(p, rigid, representations, aatype, ...)`, `structure_module(p, representations: dict, batch: dict, cfg) -> dict` returning `final_atom_positions`, `final_atom_mask`, `final_atom14_positions`, `traj` with the dtypes of the JAX module; atom14 to atom37 mapping from `all_atom_multimer.py` and `residue_constants`, imported directly in the worker (verified not to import jax).

- [ ] **Step 1: Write failing tests** (fp32): `test_sidechain_angles_and_frames_match` (1e-4), `test_final_atom_positions_and_mask_match` (composite 1e-3 against the seam capture's structure-module output given the captured representations), `test_gly_and_padded_residues_have_expected_masks` (mask matches JAX at glycine and at `seq_mask == 0`).
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass. **Commit** `mlx_af2: sidechains, atom positions, structure module`.

### Task 14: Heads, full multimer forward and recycling (checkpoint 2: speed)

**Files:** Create `bindcraft/af/mlx_af2/heads.py`, `bindcraft/af/mlx_af2/model_multimer.py`, `tests/mlx_af2/test_model_multimer.py`, `scripts/mlx_af2/bench_forward.py`.

**Interfaces:** `heads.py`: `predicted_lddt`, `predicted_aligned_error` (logits and `breaks`), `distogram` (half-logits symmetrised as in `DistogramHead`), `experimentally_resolved`. `model_multimer.py`: `class MultimerModel` with `__init__(self, model_name: str, params: Params, cfg: dict, dtype: mx.Dtype)` and `forward(self, features: dict[str, mx.array], key: mx.array, num_recycle: int, *, subbatch_size: int | None = None, use_dropout: bool = False) -> tuple[dict[str, mx.array], dict[str, mx.array]]` returning `(outputs flat by OUTPUT_KEYS, final prev state)`; recycles run without gradient and carry `prev_msa_first_row`, `prev_pair`, `prev_pos`, and `initial_atom_pos` is replaced by `prev_pos` between recycles when present.

- [ ] **Step 1: Write failing tests** (fp32): `test_full_forward_matches_every_output_key` (each key at composite 1e-3 on `small` and `medium`; positions in Å at 1e-3 relative), `test_params_are_all_consumed` (`assert_all_used` after one forward), `test_num_recycle_0_to_3_match`, `test_large_complex_initialization_updates_atom_pos_between_recycles`, `test_dropout_flag_is_keyed_and_reproducible` (with `use_dropout` true the same key reproduces the outputs exactly, a different key changes them, and the spread of outputs over eight keys is of the order JAX shows over eight keys, compared as a statistic, never pointwise), `test_padding_does_not_change_valid_positions` (the `small` case padded by 32 more dead residues gives identical valid-position outputs and identical pLDDT, PAE, ipTM inputs), `test_bf16_within_jax_spread` (every output key's relative error against JAX fp32 is at most the recorded `spread.json` value; an overshoot is reported as a finding, not absorbed by a margin).
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** (the `medium` case runs on the GPU under the lock) Expected: all pass.
- [ ] **Step 3: Checkpoint 2:** `bench_forward.py` times the forward at 150, 250, 400 residues, bf16, against metaljax (warm), with peak memory; append to `docs/mlx_backend/checkpoints.md`. Reported, never a stop. **Commit** `mlx_af2: heads, full multimer forward and recycling`.

### Task 15: Worker process

**Files:** Create `bindcraft/af/mlx_af2/{worker,guard,gpu_lock,preflight}.py`, `tests/mlx_af2/test_worker.py`.

**Interfaces:** `class RemoteModel` with `__init__(self, params_dir: str, cap_gib: float | None = None, dtype: str = "bf16")`, `start()`, `load(self, model_name: str, cfg: dict) -> int` (returns the model's handle; `cfg` is the effective `af_config.model_config(name).to_dict()` after the parent's overrides (`use_dgram` False, `num_recycle` 0, the dtype), sent by the parent), `forward(self, model_handle: int, key: np.ndarray, features: dict[str, np.ndarray], num_recycle: int, subbatch_size: int | None) -> tuple[dict[str, np.ndarray], int]` (`features` includes `use_dropout`; outputs flat by `OUTPUT_KEYS`, plus a call id), `close()`, `stats() -> dict`, and `RemoteModel.shared(params_dir: str, cap_gib: float | None = None, dtype: str = "bf16") -> RemoteModel` (one process-wide instance, so the design and validation models share one worker); `gpu_lock.hold()` is a context manager that is idempotent within a process tree; `class WorkerError(RuntimeError)`. Arrays cross by shared memory unlinked after every call; weights load lazily per model with an LRU of five; the worker holds `MultimerModel` instances.

- [ ] **Step 1: Write failing tests** (adapted from `~/RFD4-Proteina-MLX/tests/test_worker.py`): `test_worker_matches_in_process_forward` (1e-5 fp32), `test_worker_is_reused_across_calls`, `test_worker_never_imports_jax_or_torch` (`ready_info` reports both False), `test_unknown_model_raises_naming_it`, `test_dead_worker_raises_with_exit_code_then_restarts` (SIGKILL; a restarted worker has no models until `load` is called again), `test_cap_exceeded_error_names_cap` (exit code -9 and the cap in the message), `test_shared_memory_unlinked_after_success_and_error`, `test_models_switch_per_call_and_lru_evicts_the_sixth`, `test_payload_per_call_is_under_16_mb_at_300_residues_excluding_logits`, `test_second_hold_in_the_same_process_tree_passes_and_an_unrelated_process_is_refused`, `test_shared_remote_is_one_worker_for_two_models`.
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass. **Commit** `mlx_af2: persistent MLX worker with memory guard and GPU lock`.

### Task 16: Vector-Jacobian product

**Files:** Modify `bindcraft/af/mlx_af2/model_multimer.py`, `worker.py`. Create `tests/mlx_af2/test_vjp.py`, `scripts/mlx_af2/bench_grad.py`.

**Interfaces:** `MultimerModel.vjp(self, features, key, num_recycle, cotangents: dict[str, mx.array], diff_keys: tuple[str, ...], *, prev: dict | None = None, subbatch_size=None, use_dropout=False) -> dict[str, mx.array]` (recycles run without gradient unless `prev` is given, then the final pass only, with `mx.vjp` and per-block `mx.checkpoint`); `RemoteModel.vjp(self, model_handle, call_id, key, features, num_recycle, subbatch_size, cotangents, diff_keys) -> dict[str, np.ndarray]`; the worker keeps the last 8 recycle states by call id and recomputes on a miss.

- [ ] **Step 1: Write failing tests** (fp32): `test_vjp_matches_jax_vjp` (`msa_feat` and `target_feat` gradients: cosine at least 0.999 and relative error reported, on `small` and `medium`), `test_cached_and_recomputed_prev_give_identical_gradients`, `test_only_requested_keys_are_returned`, `test_no_gradient_flows_through_the_recycle_passes`.
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass.
- [ ] **Step 3: Checkpoint 3:** `bench_grad.py` times forward plus vjp and records peak memory at 250 and 400 residues in bf16 (the memory the cap must cover), against metaljax; append to `docs/mlx_backend/checkpoints.md`. Reported, never a stop. **Commit** `mlx_af2: vjp for the sequence-dependent features`.

### Task 17: `MlxAlphaFoldDesignModel` on the real worker

**Files:** Create `bindcraft/af/mlx_af2/predictor.py`, `tests/mlx_af2/test_predictor.py`. Modify `bindcraft/backend.py` (`design_model_class` returns the subclass for `mlx`).

**Interfaces:** `predictor.MlxAlphaFoldDesignModel(AlphaFoldDesignModel)` overriding `_load_parameters` (checks the weights exist, registers the model with `RemoteModel.shared(...)` and returns `jnp.asarray(handle, jnp.int32)`, the worker's handle for it, so the jitted function is told which model to run as an argument; refuses a family that is not implemented, by name), `_alphafold_runner(model_family, subbatch_size, attention_backend=None)` (a lightweight descriptor, no haiku model), `subbatch_ladder()` (one rung: `(None,)` for `'auto'`, else `(self.subbatch_size,)`), `_model_outputs` (calls `bridge.mlx_model_outputs` from Task 3 with `model_parameters` as the handle and `alphafold_runner.subbatch_size`); it exposes `provides_distogram`, `num_recycle`, `target_flexibility`, `large_complex_initialization` like the base class and enters `gpu_lock.hold()` and the preflight once per process tree, so the design and validation models share one worker and one lock.

- [ ] **Step 1: Write failing tests** (fp32 worker, dropout off, `small`): `test_predict_matches_the_jax_path` (metrics 1e-3, positions 1e-3), `test_sequence_gradients_match_the_jax_path` (cosine at least 0.999 per chain, loss equal to 1e-3), `test_diff_keys_cover_every_sequence_dependent_float_feature` (a `jax.jvp` of the front, `prepare_design_sequence_features` then `alphafold_input_features`, with a random tangent on `sequence`: every float feature with a nonzero tangent is in `DIFF_KEYS`, and both `DIFF_KEYS` entries have one), `test_two_states_in_one_call` (complex plus binder alone, gradients match JAX), `test_model_is_redrawn_each_step` (five models, many calls: outputs differ per model and equal the JAX per-model outputs), `test_design_and_validation_models_share_one_worker_and_one_lock` (building both starts one worker and takes the lock once), `test_worker_killed_during_gradients_names_the_cause_and_stops` (exit code and cap in the message, no fallback to JAX, no shared memory left), `test_unimplemented_family_is_refused_by_name` (a monomer model before Task 19), `test_memory_failures_do_not_trigger_the_jax_ladder` (`bindcraft.af2.is_out_of_memory(error)` is False for a cap kill and for an allocation failure surfaced by the worker, while the message still names the cap), `test_sequence_gradients_run_through_the_repos_shape_logic_unchanged` (on CPU JAX with the bridge, `_widest_shard_that_fits` runs, and `shape_subbatch_size` and `shape_attention_backend` are recorded).
- [ ] **Step 2: Run** (expected FAIL: the class does not exist), **implement**, **run** again. Expected: all pass. **Commit** `mlx_af2: MlxAlphaFoldDesignModel on the real worker`.

### Task 18: Monomer family, trunk (forward only)

**Files:** Create `bindcraft/af/mlx_af2/monomer.py`, `tests/mlx_af2/test_monomer.py`. Modify `capture_refs.py` for `model_1_ptm` and `model_2_ptm`.

**Interfaces:** `class MonomerModel` with the same `forward` signature as `MultimerModel` (no vjp: validation only). The Evoformer ops, the extra-MSA stack and the layers are the shared ones of Tasks 6 to 9; this task adds the monomer `EmbeddingsAndEvoformer` of `modules.py` (its preprocessing, relative position without chain ids, recycle embedding with the distogram of `prev_pos`) and the monomer template embedding (`TemplateEmbedding`, `SingleTemplateEmbedding`, `TemplatePairStack`), with the config differences from `config.CONFIG_DIFFS`.

- [ ] **Step 1: Capture** monomer seam references (`fp32`, `bf16`) for `small` and `medium`, with the monomer features the repo already builds.
- [ ] **Step 2: Write failing tests** (fp32): `test_monomer_embeddings_match`, `test_monomer_template_embedding_matches`, `test_monomer_trunk_representations_match` (1e-3), `test_both_ptm_models_and_their_config_diffs`.
- [ ] **Step 3: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass. **Commit** `mlx_af2: monomer trunk`.

### Task 19: Monomer structure module and heads

**Files:** Create `bindcraft/af/mlx_af2/monomer_structure.py`. Modify `monomer.py`, `predictor.py`, `tests/mlx_af2/test_monomer.py`.

**Interfaces:** the monomer structure module of `folding.py` (`quat_affine.py`, `r3.py`, `all_atom.py`: quaternion affine, IPA, sidechain, atom positions) and the monomer heads (`modules.py` `PredictedLDDTHead`, `PredictedAlignedErrorHead`, `DistogramHead`, `ExperimentallyResolvedHead`), returning the same `OUTPUT_KEYS`; `MlxAlphaFoldDesignModel` now accepts the monomer pool and the refusal test of Task 17 is updated to expect success for `model_1_ptm`.

- [ ] **Step 1: Write failing tests** (fp32): `test_quat_affine_and_r3_ops_match`, `test_monomer_ipa_and_fold_iteration_match`, `test_monomer_final_positions_and_mask_match` (1e-3), `test_monomer_full_forward_all_output_keys` (`small`, `medium`), `test_monomer_bf16_within_spread`, `test_default_validation_model_builds_under_mlx` (a default-settings campaign constructs both design and validation models with no refusal).
- [ ] **Step 2: **Run** (expected FAIL: the module does not exist yet), **implement** by porting the cited JAX source faithfully, **run** again.** Expected: all pass. **Commit** `mlx_af2: monomer structure module, heads and predictor support`.

### Task 20: End to end and the parent's device

**Files:** Create `scripts/mlx_af2/e2e_compare.py`, `tests/mlx_af2/test_e2e.py`, `docs/mlx_backend/e2e.md`. Modify `bindcraft/campaign.py` only if a run shows a defect.

**Interfaces:** consumes the whole backend; produces a comparison of the JAX path and the MLX path on `examples/pdl1_denovo.json` with a short trajectory budget, over at least five seeds each: distributions of final design loss, pLDDT, ipTM and trajectory time; and a measurement of ProteinMPNN plus the filters with the parent's JAX on the CPU against `JAX_PLATFORMS=metal`.

- [ ] **Step 1: Write failing tests** (marked `integration`): `test_trajectory_only_run_exits_cleanly_with_mlx_and_multimer_validation` (`validation_model: multimer`), `test_default_run_completes_with_monomer_validation`, `test_no_silent_fallback` (kill the worker mid-trajectory: nonzero exit and the cause in the output), `test_jax_default_path_unaffected` (without `BINDCRAFT_BACKEND` the run equals the golden of Task 2).
- [ ] **Step 2: Run** the comparison under the GPU lock (disk and preflight pass): `BINDCRAFT_BACKEND=mlx .venv/bin/python scripts/mlx_af2/e2e_compare.py --seeds 5`. Expected: both backends complete; distributions are written to `docs/mlx_backend/e2e.md` with an honest statement of overlap (bf16 trajectories are noisy; the check is distributional, not pointwise). Apply the phase-2 trigger from Global Constraints to the measured shares of the CPU-side JAX work (front and loss per step; ProteinMPNN and filters per trajectory) and record the verdict in `docs/mlx_backend/e2e.md`. Decide the parent's device from the MPNN and filter timings (CPU unless it costs more than the AlphaFold gain; the choice is documented, with `JAX_PLATFORMS=metal` as the override).
- [ ] **Step 3: Run the tests; commit** `mlx_af2: end-to-end comparison and parent device measurement`.

### Task 21: Final benchmarks, documentation and handoff (checkpoint 4)

**Files:** Create `docs/mlx_backend/README.md`, `docs/mlx_backend/HANDOFF.md`. Modify `docs/installation.md`, `README.md` (one paragraph under the Apple silicon notes).

**Interfaces:** documents `BINDCRAFT_BACKEND=mlx`, the extras, the weights, the memory cap, the lock, exit behaviour, the checkpoint tables, the rulings made during execution, the deferred minors, and the measured ratios against metaljax for one block, the forward, forward plus gradient, a full trajectory and the validation refold set, each warm with peak memory.

- [ ] **Step 1: Run the full suite** `BINDCRAFT_MLX_INTEGRATION=1 .venv/bin/python -m pytest tests/mlx_af2 -q` and the JAX-only subset with no MLX installed in a scratch venv check of imports (`import bindcraft.campaign` works without `mlx`). Expected: all pass; the second shows the JAX path has no MLX import.
- [ ] **Step 2: Measure and write** the tables; **write** the docs; read every claim against the logs.
- [ ] **Step 3: Commit** `docs: MLX AlphaFold backend guide, handoff and benchmarks`. Then the whole-branch review and `finishing-a-development-branch`; nothing is pushed.
