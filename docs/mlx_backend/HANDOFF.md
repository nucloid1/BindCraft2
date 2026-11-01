# MLX AlphaFold backend: handoff

Written at the end of Task 21 of the plan (`docs/superpowers/plans/2026-10-30-mlx-alphafold-backend.md`, spec `docs/superpowers/specs/2026-10-30-mlx-alphafold-backend-design.md`). The guide for using the backend
is `README.md` in this folder; the measurements are in `checkpoints.md` and `e2e.md` (raw results in `e2e_results/`). This file is for whoever continues the work: what is established, what is not, what is open, and every
decision taken on the way.

## Where the work is

- Repo `~/BindCraft2`, branch **`mlx-backend`**, local only: **nothing has been pushed**. It carries the user's Metal fork (`bf5e0cc`, branch `apple-silicon-metal`, left untouched and still 147 commits behind upstream: whether to update it
  is an open question) and a merge of upstream (`205fe64`, upstream `main` at `b6fb3ae`; backup branch `backup/mlx-backend-before-upstream-merge`; local `main` was fast-forwarded to upstream). The MLX work is 36 commits on top, listed below.
- The executor's ledger (one line per task, every ruling, every finding) is `.superpowers/sdd/2026-10-30-mlx-alphafold-backend/progress.md`; the speed work (2026-11-01, plan `docs/superpowers/plans/2026-11-01-mlx-speed-work.md`) has its own,
  `.superpowers/sdd/2026-11-01-mlx-speed-work/progress.md`. Both are git-ignored scratch: if they are gone, this file and `git log` are the record.
- On 2026-11-01 Ethan said "build the speed work" and "and merge to main": after the speed work, both suites and its review, `mlx-backend` was fast-forwarded into local `main` (still nothing pushed).
- Memory notes for the assistant are in `~/.claude/projects/-Users-ethanputnam-Bindcraft-Hunter/memory/` (`bindcraft2-mlx-backend.md`).

## What is established

- **Design** (AlphaFold-Multimer, forward and gradient) and **validation** (multimer or pTM monomer, forward only) run through MLX on the Apple GPU; fp32 agrees with the vendored JAX modules to 1e-7..1e-5 per stage and with the captured seam
  references of the real model (single trajectory inputs, outputs, cotangents and gradients) to 1e-3 or better on live residues; the gradient cosine against `jax.vjp` is 0.9999 or better, with one documented exception (a chaotic fp32 state, `checkpoints.md`).
- **Real campaigns run end to end** on the backend (design stages, ProteinMPNN, validation, filters), including `hard_target` and the desperation ladder; a killed worker stops the run nonzero with the cause named; with `BINDCRAFT_BACKEND` unset the JAX path is untouched and no MLX module is imported.
- **Against metaljax** (M3 Max, bf16, warm), after the speed work: 1.98x per gradient round of real trajectories (4.48 against 8.87 s at 195 residues; 1.28x before it), 1.8x to 3.0x per forward, 2.0x per design step at 250 residues and 2.9x at 400; outcome distributions over five seeds overlap, and the speed work left every trajectory identical (`e2e.md`). The CPU-side JAX work is about 0.2% of a gradient round: the plan's phase-2 trigger (port it if above 20%) is not met.
- Verification of the speed work: the default suite `BINDCRAFT_MLX_INTEGRATION=1 BINDCRAFT_MLX_MIN_FREE_GB=20 JAX_PLATFORMS=cpu pytest tests/mlx_af2 -q` **289 passed in 57:40** at `4fe248c`, and in GPU mode
  (`BINDCRAFT_MLX_TEST_DEVICE=gpu`) **281 passed, 9 skipped, 1 xfailed** in 14:42 at `15d56f6` (which adds two review tests); the xfail is the documented fp32 chaotic-gradient case.
- Verification before the speed work: the whole suite on a clean tree, `BINDCRAFT_MLX_INTEGRATION=1 BINDCRAFT_MLX_MIN_FREE_GB=20 JAX_PLATFORMS=cpu pytest tests/mlx_af2 -q`: **249 passed in 55:31**, no failures, skips or xfails (2026-11-01, at `a582405`; the fp32 chaotic-gradient case is an xfail only in GPU test mode, a documented finding of the shipped model, `checkpoints.md`).

## What is NOT established

- **Quality of finished designs.** Five seeds, one trajectory each, a short budget, designs far below acceptance. Stage survival was 3 of 5 on MLX and 1 of 5 on JAX (Fisher p = 0.52, not distinguishable); a run of 10 to 20 seeds at the default budget is the next thing to do.
- **metaljax's bf16 rounding.** The MLX model rounds to bfloat16 at every operation; the JAX-CPU bfloat16 model does not (XLA keeps float32 between fused ops), so JAX-CPU numbers are tighter than real bfloat16 allows. Whether the shipped Metal path rounds like one or the other is unmeasured.
- **The strict-rounding monomer bf16 ensemble** was not captured (about 40 CPU minutes); the monomer bf16 gate is loose on purpose (`Task 19` ruling below).
- **Multi-worker campaigns** (`design_workers`) under this backend: one GPU and one lock, so a second process fails at once naming the lock holder; parallelism is not offered.
- Only one machine (an M3 Max with 48 GB); only macOS 27.0.1, MLX 0.32.4, metaljax 0.11.9.

## Open questions for Ethan

1. **Exact variance in the triangle-multiplication layer norms?** Haiku's fast variance (E[x^2] - E[x]^2) cancels badly on large-mean pair activations: the shipped fp32 gradient carries up to about 5% noise on some states (CPU fp32 5.3e-2 from an fp64 truth, GPU 6.6e-4, exact variance 7.8e-4). The port copies the formula for parity. Switching to the exact variance would be about 60x more accurate and would break formula parity with JAX. Not done. Unanswered.
2. **Zero every undefined torsion angle in all backends?** One undefined template torsion angle per target chain start is rounding junk in the shipped monomer models (eager (0,0), CPU jit (0.7455, 0.6664)), worth up to 1.4e-2 relative in `single`. MLX takes the deterministic value. Zeroing it everywhere would make the templates platform-independent and would change the shipped JAX numbers. Not done. Unanswered.
3. **The per-call deadline for a stalled worker** (`BINDCRAFT_MLX_CALL_TIMEOUT_S`, default 1800 s, 0 turns it off) is NOT in the JAX version: it came from the review, because a frozen worker would otherwise hang a run forever with the GPU lock held. Kept as built and on by default, pending his word.
4. Update the `apple-silicon-metal` branch with upstream (147 commits behind)? And the ESMFold2-MLX project's merge, PR or keep question, from an earlier session. Both unanswered.

## Decisions Ethan made

AlphaFold model only (the loss, ProteinMPNN, filters and optimizer stay JAX, on the CPU); MLX is a new opt-in backend beside the others, not a replacement; bfloat16 like the shipped model; no stop rule (speed is reported, never stops the work);
the Neural Engine dropped after a measured spike (the GPU was 1.5x to 2.8x faster on every op); "if CPU JAX is a bottleneck, port it, otherwise the plan is fine as written" (measured: not a bottleneck); native inline execution of the plan;
the disk rule (below 90% before a real-model run) is never loosened by default, and on 2026-10-31 he said "its fine, don't worry about memory" about the 95-97% disk, so runs used the explicit opt-in `BINDCRAFT_MLX_MIN_FREE_GB=20`; and he said "drop it"
about the preflight's rule that refused to start while another process held more than 8 GB (removed). Standing: never modify `~/RFD4-Proteina`; one GPU run at a time through the shared lock.

## Speed work: built 2026-11-01

Ethan asked for the speed headroom to be investigated (`checkpoints.md`, "Speed headroom: what the investigation found"), then said "build the speed work". Built, with the numerics unchanged (`checkpoints.md`, "Speed work"):
compiled Evoformer blocks on the GPU (`compiled.py`: one compiled forward and one compiled vjp closure per stack, the layer's parameters passed in as arrays, one trace per shape; `BINDCRAFT_MLX_COMPILE=0` turns it off);
the model's vjp as an explicit chain run outside any MLX transformation (MLX runs a compiled function inline and unfused under `mx.vjp`: measured), with each Evoformer stack differentiated by a reverse loop
(`evoformer.stack_vjp`); tapes, so the vjp after a design forward skips the final pass's Evoformer forward (kept per call by the worker under `BINDCRAFT_MLX_TAPE_GIB`, default a fifth of the cap); and an allocator
cache of a quarter of the cap instead of 2 GiB. Results: design step 9.9 -> 6.2 s at 250 residues and 35.3 -> 18.8 s at 400; a real trajectory's gradient round 6.92 -> 4.48 s (1.98x faster than metaljax); every
trajectory identical. A fresh review (model `fable`) found no Critical issue; its Important test gap was fixed, one Important finding was re-graded Minor with a ruling, the rest are deferred minors below.
On the way, four existing test assertions turned out never to run (`assert_live(a) and rel_err(...) < tol` short-circuits on `None`); they were fixed first and pass.

## How to run things

```bash
cd ~/BindCraft2
BINDCRAFT_MLX_INTEGRATION=1 JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/mlx_af2 -q            # the whole suite: units on the MLX CPU device, gpu-marked end-to-end campaigns on the GPU under the lock
BINDCRAFT_MLX_INTEGRATION=1 BINDCRAFT_MLX_TEST_DEVICE=gpu JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/mlx_af2 -q   # every test on the GPU (an accuracy run on the real device)
JAX_PLATFORMS=cpu .venv/bin/python scripts/mlx_af2/capture_refs.py [--monomer] [--bf16-ensemble --strict-rounding] [--noise-floor]   # the references the tests are judged against, into ~/.cache/bindcraft_mlx_refs
.venv/bin/python scripts/mlx_af2/e2e_compare.py --compare --seeds 5 ; scripts/mlx_af2/bench_{block,grad,forward}.py     # the benchmarks
TREE=<a tree> PYTHONPATH=<the same tree> .venv/bin/python scripts/mlx_af2/experiments/before_after.py   # the speed work's before/after measurement, the same code against any tree
```

A whole-suite run takes about 58 minutes on the CPU device (57:40 at the last run) and about 15 in GPU mode (14:42). The disk was 95 to 97% full throughout: runs used `BINDCRAFT_MLX_MIN_FREE_GB=20`, and a watchdog killed nothing. `task-done` records a trailing worker-exit `BrokenPipeError` instead of the result on some runs:
correct the ledger line by hand. zsh does not word-split an unquoted variable: put loops in a bash script file.

## Map

`bindcraft/af/mlx_af2/`: `weights.py` (npz loader, fusing of the triangle projections), `layers.py`, `pair_ops.py`, `msa_ops.py`, `evoformer.py` (blocks, stacks, tapes, `stack_vjp`), `compiled.py` (compiled blocks), `embeddings.py` (front, join, finish), `templates.py`, `geometry.py`, `all_atom.py`, `structure.py` (multimer),
`heads.py`, `model_multimer.py` (recycling, forward, the vjp chain), `monomer.py` and `monomer_structure.py` (pTM family), `outputs.py`, `bridge.py` (the `custom_vjp` over `pure_callback`), `worker.py` (process, shared memory, call ids, tapes and their budget, allocator cache, deadline), `guard.py`, `gpu_lock.py`,
`preflight.py` (disk only), `predictor.py` (`MlxAlphaFoldDesignModel`). `bindcraft/backend.py`, small hooks in `bindcraft/af2.py` (three) and `bindcraft/__init__.py` (the parent's JAX on the CPU). `tests/mlx_af2/`, `scripts/mlx_af2/`, `docs/mlx_backend/`.

## Rulings made during execution (verbatim from the ledger, in order)

- Setup: Ruling: work in place on branch `mlx-backend` of ~/BindCraft2, no git worktree — the .venv (the only env with jax, haiku, ml_collections) lives in the checkout and the branch is not main — a worktree would need a second venv — cost if wrong: none, branch is local and unpushed.
- Setup: Ruling: MLX parity tests (fp32) run on the MLX CPU device unless marked `gpu`, so they do not wait for the GPU lock that Ethan's RFD4 job holds (pid 8394, since 20:51); gpu-marked tests (bf16 spread, benchmarks, worker on GPU) wait for the lock and are never run without it — cost if wrong: CPU/GPU kernel differences, closed by re-running the parity suite on the GPU under the lock before the final review.
- Setup: Ruling: before any code, merged upstream/main (b6fb3ae, 147 commits ahead of the old base 5342aef) into mlx-backend at Ethan's request ("make sure bc2 is up to date per the repo") — conflicts only in README, docs, install.sh, pyproject (oneapi removal kept, metal kept) — backup branch backup/mlx-backend-before-upstream-merge at 2440cc5 — local main fast-forwarded to b6fb3ae (it had no local-only commits); apple-silicon-metal (Ethan's pushed Metal branch) left untouched and still 147 commits behind — cost if wrong: reset to the backup branch.
- Setup: Ruling: renamed flag large_complex_initialization (was bigbang_initialization) and the new memory ladder (executable_fits, is_out_of_memory, subbatch_ladder) are handled in Task 17: the MLX subclass overrides subbatch_ladder to one rung and worker memory errors must not match is_out_of_memory — plan, design updated and committed.
- Setup: Ruling: gpu_lock.py (adapted from ~/RFD4-Proteina-MLX) is pulled forward from Task 15 into Task 1 because the conftest `gpu` marker needs it, made re-entrant within one process, with its tests — cost if wrong: none, Task 15 only extends it.
- Task 2: Ruling: the `_model_outputs` hook is `(alphafold_runner, model_parameters, key, model_inputs)`, with no model_name and no subbatch_size — the per-state runner already carries subbatch and attention backend, and both compile caches (`prediction_compile_cache`, `gradient_compile_cache`) are keyed by model FAMILY with model_parameters as a jit argument, so one compiled function serves all five multimer models while `_sample_design_model` redraws the model every step; the plan's bridge baked `model_name` in at trace time, which would run the first model on every cache hit — cost if wrong: the MLX backend would silently use one model; fixed in the plan (Tasks 2, 3, 15, 16, 17: the model travels as an integer handle in `model_parameters`).
- Task 3: Ruling: outputs.py is jax-free (worker imports it): it holds OUTPUT_KEYS, output_shapes(length) -> (shape, dtype name), nest(); the jax ShapeDtypeStruct form is bridge.output_spec — the plan said outputs.output_spec — cost if wrong: none, plan text updated.
- Task 3: Ruling: the bridge drops `prev` (three zero arrays, one pair-sized, 33 MB at 256 residues) from the callback and the worker starts its recycles from zeros; pinned by test_front_always_starts_from_zero_prev (also with large_complex_initialization) — cost if wrong: a caller passing a nonzero prev would be ignored; none exists (recycled_alphafold_outputs gets the front's zeros).
- Task 5: Ruling: tests/conftest.py gains an autouse `mlx_device` fixture that sets MLX's default device to the CPU for every test except gpu-marked ones (which hold the shared lock and get the GPU), enforcing the setup ruling that unit parity tests never touch the GPU another run may hold; test_conftest.py pins it — the gpu-marked half (test_mlx_uses_the_gpu_under_the_gpu_marker) cannot run while Ethan's RFD4 job (pid 8394) holds the lock, so the suite runs with -m "not gpu" and that one test is deferred — cost if wrong: none for correctness; the GPU path itself is first exercised in Task 9's checkpoint and Task 14.
- Task 6: Ruling: `attention` takes `bias` (the JAX Attention.__call__ argument; callers build it as 1e9 * (mask - 1)), not `mask` as the plan's signature said — plus keyword-only `subbatch_size` and `chunk_size` (query chunking of accel._attend_chunked, the repo's default on Apple silicon; same arithmetic, smaller logits) — cost if wrong: none, callers in Tasks 7-8 pass bias as JAX does.
- Task 7: Ruling: `triangle_attention` has no `starting` flag (the plan's signature had one): the config's orientation says it (starting node = per_row, ending node = per_column), as the JAX module does — cost if wrong: none; tests still named ..._starting/_ending.
- Task 9: Ruling: `evoformer_stack` takes `is_extra` (the plan's signature lacked it) and per-block dropout keys from mx.random.split(key, num_block); a block draws 9 sub keys in the order the JAX block does (random streams differ from JAX's, so dropout is compared as statistics and by the shared-axis structure, never pointwise); remat is mx.checkpoint per block, its gradient equals the plain one to 1e-5.
- Task 10: Ruling: the whole `EmbeddingsAndEvoformer.__call__` assembly (embed_inputs, the extra-MSA stack, the main stack, single activations, the float32 output cast) lives in embeddings.py as `embeddings_and_evoformer(...)`, not only the embeddings — the plan split it across Tasks 10 and 14 — with a `template_embedding(query_pair, mask_2d, key) -> (pair update, extra MSA rows, rows mask)` hook that Task 11 fills; Task 14 then adds heads and recycling around it — cost if wrong: none.
- Task 12: Ruling: Tasks 11 and 12 are executed in the order 12, 11 — the template embedding needs make_backbone_affine, Rigid3 inverse/apply and chi angles from the geometry package that Task 12 builds, so templates first would have been circular — cost if wrong: none; ledger lines keep the plan's numbers.
- Task 13: Ruling: structure_module returns the JAX head's outputs by the same names with the per-iteration ones stacked on a leading axis of num_layer (traj, sidechains/{angles_sin_cos, unnormalized_angles_sin_cos, atom_pos, frames}, final_atom14_*, final_atom_*, final_rigids, act); generate_rigids exposes the per-iteration activations the sidechain head needs; backbone_trajectory stays as a thin wrapper for Task 12's tests.
- Task 11: Ruling: dtype handling follows SingleTemplateEmbedding (features cast to the query dtype; unit vectors and the backbone mask in bfloat16 only under bf16; multichain mask applied as a float); the "template_dgram" input path of the JAX module (precomputed distograms) is not ported — the repo's front never supplies it — cost if wrong: a future feature supplying template_dgram would silently ignore it; the hook reads only template_aatype, template_all_atom_positions, template_all_atom_mask.
- Task 14: Ruling: the bf16 gate is judged against a STRICT-rounding JAX ensemble (capture_refs.py --bf16-ensemble --strict-rounding, key bf16_ensemble_strict in spread.json, 6 members, eps 4e-3) with the default band printed but not gated — strict rounding is what bfloat16 hardware and MLX do; the default band is flattered by XLA-CPU excess precision. Gates: geometric-mean ratio <= 1.5, each output <= 3x strict max, PAE and distogram ratio <= 1.10 (their bands are <1% wide; port measures 0.99-1.01; the same PAE ratio against the default band is 1.30). Control: a bf16 LayerNorm mutant passes the coarse gates (ratio 1.05 on PAE/distogram), so the coarse gates catch only gross bf16 errors; I did not tighten further because the mutant is not meaningfully worse. Cost if wrong: if "bf16 like today" meant "as accurate as default JAX-CPU bf16", MLX is 1.3-1.5x noisier on PAE/distogram logits; the shipped Apple GPU path (metaljax) rounding behaviour is UNMEASURED (needs the GPU lock); a float32 residual stream would recover the accuracy at memory/speed cost and is not built.
- Task 14: Ruling: precise=True softmax kept although it has no measurable end-to-end effect — it is 12x more accurate on peaked attention and MLX's docs compute in float32; speed unmeasured (GPU deferred); first thing to drop if Checkpoint 1 shows it costing >2%.
- Task 15: Ruling: a restarted worker REPLAYS the parent's registrations (name and config, in handle order) instead of "has no models until load is called again" as the plan text says — handles are integers baked into traced programs and a restart mid-run must not turn them into unknown handles; weights still load lazily, per model, on first use after the restart. Cost if wrong: a restart costs a reload per model on first use (seconds), none otherwise.
- Task 15: Ruling: test names differ from the brief's list but every named behavior has a test: matches_in_process_forward, reused_across_calls, never_imports_jax_or_torch, unknown model (three tests: no weights names the model, unimplemented family refused by name, unknown handle names the handle), dead worker exit code then new worker serves, cap exceeded names the cap and does not match is_out_of_memory, shared memory unlinked after success and error, five resident with eviction re-read, payload < 16 MB at 300 residues, shared remote one worker per (params, dtype); the lock tests (idempotent hold within a process tree, stranger refused) live in test_gpu_lock.py from Task 1. gpu_lock.py already existed (Task 1). Added beyond the brief: preflight.py (disk guard, env BINDCRAFT_MLX_MIN_FREE_GB opt-in floor / BINDCRAFT_MLX_MAX_DISK_PCT, never relaxed by default) with 4 tests; conftest mlx_device also sets BINDCRAFT_MLX_DEVICE so worker subprocesses started by tests stay on the CPU.
- Task 16: Ruling: MultimerModel gained forward_with_state (outputs, state after, state BEFORE the final pass) and final_features (the initial_atom_pos rewrite a resume must apply); forward keeps its signature. vjp's use_dropout defaults to None (follow the features' use_dropout flag, as forward does) instead of the plan's False — a False default would differentiate a different function than the forward call computed when dropout is on. Cost if wrong: none for use_dropout=False callers.
- Task 16: Ruling: call ids are assigned by the PARENT (RemoteModel.last_call_id, never reset) and the worker serves a cached state only if handle, number of recycles and key match; a worker-side counter would restart at 0 after a worker restart and could hand a pending backward pass another forward's state (silently wrong gradient). Mutation controls on worker.py, each failing its own test: cache ignored (3 tests), hit check weakened (the other-inputs test), no eviction (the eight-states test), call counter reset on restart (the restart test — this mutant first SURVIVED, the test only called once after the restart, and was strengthened to two calls; then caught). worker.py restored identical.
- Task 17: Ruling: worker_error hyphenates "out of memory" / "resource exhausted" in text the worker raises (case-insensitive, space or underscore) — the repo's retry ladder (af2.is_out_of_memory) reruns such failures with smaller shards and chunked attention, which would only repeat a failure whose budget is the cap; the cause stays readable (out-of-memory) and the cap is still named. Cost if wrong: a message-matching tool looking for the exact phrase will not find it.
- Task 17: Ruling: the model's name is not baked into the trace: _load_parameters returns the worker's handle as an int32 array (the runtime argument); five models share one traced program per shape (asserted: at most one trace of _model_outputs across calls to three models). Cap: BINDCRAFT_MLX_CAP_GIB or 75% of hw.memsize (36 GiB on this 48 GB machine). The session (preflight and, for a GPU worker only, the shared lock) starts once per process when the first model loads; a CPU worker takes no GPU lock. Class attribute dtype = 'bf16' (a subclass asks for 'fp32', as the tests do). Test fixtures force BINDCRAFT_MLX_DEVICE=cpu at module scope BEFORE the worker starts, because conftest's per-test device fixture runs after module fixtures.
- Task 18: Ruling: MLX computes the deterministic eager value for undefined degenerate angles (zeros) and the trunk test injects the reference's own jitted torsion rows; test_torsion_angles_disagree_with_jit_only_where_they_are_undefined pins that every DEFINED angle agrees under jit to 1e-5 and at most 2 undefined entries differ. Cost if wrong: MLX's monomer trunk differs from the CPU-jit JAX path by up to 1.4e-2 relative at the first target residues' template rows, an arbitrary value either way.
- Task 18: Ruling: use_template_unit_vector, backprop_dgram and use_dgram are refused by name (NotImplementedError) because the vendored unit-vector branch is not runnable (undefined names) and the repo never switches the others on; the pointwise template attention runs unsharded (the JAX path shards by 128 rows, numerically identical); model_3_ptm was added to the captures to cover template.enabled False; the loader's pre-planned monomer scopes (__layer_stack_no_state, FUSED_SCOPES) worked unchanged.
- Task 19: Ruling: the bf16 monomer gate is LOOSE on purpose — PAE, distogram and resolved logits within 3x the JAX bf16 error (the JAX-CPU bf16 reference is flattered by XLA excess precision, see the multimer finding), positions and pLDDT below a gross-error ceiling of 0.25 because they are chaotic (a medium-case JAX bf16 ensemble spans 3x and more; MLX 0.07 against one JAX draw of 0.014-0.022 failed the first, tighter version). No strict-rounding monomer ensemble was captured (about 40 CPU-minutes); the monomer ops are the multimer's and fp32 parity is 1e-7. Cost if wrong: a bf16 error in monomer positions or pLDDT up to 0.25 relative would pass.
- Task 19: Ruling: initial_atom_pos (the large-complex initialisation, which needs the quaternion of a rotation matrix) is refused by name for the monomer models, as is a gradient (MonomerModel.vjp raises NotImplementedError naming the model: validation only; the worker reports it as 'no gradient'); the campaign builds its validation models without large-complex initialisation. The final atom37 positions use the batch's own residx_atom37_to_atom14 and atom37_atom_exists tables, as the JAX code does (the multimer port derives them from aatype; equal for live residues). Cost if wrong: a design run with a monomer design model and bigbang initialisation fails with the named error instead of running.
- Task 19: Ruling: outputs.model_family (multimer / monomer for names ending _ptm / None) is the one rule for the predictor and the worker; the Task 17 and Task 15 refusal tests now use model_1 (a monomer without a PAE head, still unimplemented) instead of model_1_ptm, as the plan says; test_default_validation_model_builds_under_mlx builds a multimer design model and the monomer validation pool through the repo's factory with a fake remote.
- Task 20: Ruling: test_jax_default_path_unaffected checks the behavior (completes, no MLX worker or session output, design_model_class is the base class) rather than comparing a campaign with the Task 2 golden: the golden is the bit-identical seam output captured in Task 2 and is pinned by test_hooks; a campaign has no stored golden to equal. Cost if wrong: a change that altered JAX numbers without touching the seam would not be caught here (test_hooks and the suite would still catch it at the seam).
- Task 20: Ruling: bindcraft/campaign.py was not modified — no run showed a defect. The e2e tests are marked gpu (a campaign on the CPU device is too slow for the ordinary run) and run with BINDCRAFT_MLX_INTEGRATION=1 -m gpu; the CPU 'not gpu' contract excludes them, so this task's completion run is the whole suite in GPU mode (BINDCRAFT_MLX_TEST_DEVICE=gpu).
- Final: Ruling: the reviewer set aside per-op numerical parity of the ported layers, Evoformer, templates, geometry, structure module and heads — stands: verified by the suite at 1e-7..1e-5 in fp32 with liveness guards and mutation controls, and its read-only limits forbade re-running it — cost if wrong: none beyond what the suite covers.
- Final: Ruling: the looseness of the bf16 gates (Task 14 coarse gates plus the tight PAE/distogram gate, Task 19's monomer gate) — stands, ruled at Tasks 14 and 19 with the cost stated — cost if wrong: a bf16 error in chaotic outputs below the stated ceilings would pass.
- Final: Ruling: the fp32 chaotic-gradient xfail and the exact-variance question — stands as a documented finding about the shipped model, left to Ethan — cost if wrong: fp32 design gradients keep up to ~5% cancellation noise on some states (masked by bf16 noise in the shipped model). Still unanswered.
- Final: Ruling: the undefined-torsion jit/eager discrepancy (monomer) — stands (<= 2 undefined entries, pinned by a test, MLX takes the deterministic value) — cost if wrong: up to 1.4e-2 relative at those rows against the CPU-jit JAX path.
- Final: Ruling: not porting template_dgram, use_template_unit_vector, backprop_dgram, use_dgram — stands: the repo never sets them, three of the four are refused by name, template_dgram has no producer in the front — cost if wrong: a future feature supplying template_dgram would be silently ignored.
- Final: Ruling: speed headroom and the 400-residue vjp profile — stood as the TODO Ethan asked to be noted, not plan scope; SUPERSEDED 2026-11-01 by the investigation (section above and `checkpoints.md`): nothing built, build-or-not is Ethan's decision — cost: speed left on the table.
- Final: Ruling: apply_desperation_settings and the desperation ladder themselves — upstream code, not changed; their interaction with the monomer validation is the Critical finding, fixed — cost if wrong: none.
- Final: Ruling: multi-worker (design_workers) campaigns under this backend — stands: one GPU and one lock, so the MLX backend serialises and a second process fails loudly naming the lock holder; stated in HANDOFF.md — cost if wrong: a user wanting parallel workers gets a refusal, not parallelism.
- Final: Ruling: the measured ratios in the tables were not re-run by the reviewer — stands: every row traces to checkpoints.md, e2e.md and e2e_results/ — cost if wrong: none beyond measurement noise.
- Final: Ruling: pyproject.toml adds repo-wide pytest config (testpaths, addopts) — stands: no other tests exist in the repo, so nothing changes for anyone — cost if wrong: a future test directory inherits the options.
- Final: Ruling: the spec's claim that the campaign sets provides_distogram — stands as a stale spec line: campaign.py reads it with a True default and the MLX path always returns a distogram — cost if wrong: none.

## Deferred minors (not fixed; from the final review)

- MAX_LOADED = 5 models resident against 7 registrations in a default campaign (5 multimer + 2 monomer share one worker): each eviction re-reads a 373 MB npz, about 1% of a default-budget trajectory; MAX_LOADED = 8 would remove it.
- a narrow shared-memory leak window: the worker packs its output block and untracks it before sending, so a worker killed in exactly that window leaves a block (about 90 MB at 400 residues) that nothing names (macOS keeps it until unlinked or reboot).
- precise=True softmax is a deliberate divergence from the JAX arithmetic (ruled at Task 14, measured free at Checkpoint 1 and on the real stack); listed so the divergence is not lost.
- predictor.model_config hard-codes use_cueq = False where the base runner uses self.use_cueq (harmless on Apple silicon).
- atexit.register(self.close) inside RemoteModel.start accumulates one registration per worker restart.
- worker.untrack uses the private SharedMemory._name (bare except degrades it to resource-tracker warnings); SharedMemory(track=False) is the supported API on this Python.
- RemoteModel.read_outputs updates last_shm_names and last_seconds outside self.lock (bookkeeping only; bridge.REMOTE_CALLS serialises in practice).

## Rulings of the speed work (verbatim from its ledger, in order)

- Setup: Ruling: executing inline on branch mlx-backend (HEAD ff897cc at start), after Ethan's "build the speed work" and "and merge to main" (2026-11-01) — the merge is a local fast-forward of main after both suites and a fresh review, not before — cost if wrong: none, nothing pushed.
- Setup: Ruling: the stack functions evoformer_stack keeps its signature; the monomer's forward-only call (monomer.py:128) shares it, so a compiled forward reaches the monomer validation passes too — measured in Task S4, not assumed — cost if wrong: a monomer parity or speed regression found by the GPU-mode suite.
- Task S1: Ruling: remat=True keeps its name and now means the explicit reverse loop (no mx.checkpoint); remat=False stays the plain loop and records a tape when given one — the monomer's forward-only call is unchanged — cost if wrong: none, both paths are gradient-tested against plain autodiff.
- Task S2: Ruling: test_compiled.py's CPU tests were written before compiled.py but first run after it existed, so they were never seen RED; covered instead by mutation controls: kill switch ignored, compile on every device, has() always true — caught; 'ArrayParams ignores the dtype' survived and is equivalent (every parameter name is read in one dtype), its non-equivalent form (always the default dtype) is caught by the bf16 cases added for it — cost if wrong: a dtype bug in ArrayParams that only shows on the GPU path would be caught by the gpu-marked equality tests.
- Task S2: Ruling: BINDCRAFT_MLX_COMPILE=0 is the only switch (no force-on for the CPU: MLX's CPU compile runs a C++ compiler, ~3 s per kernel) — cost if wrong: none.
- Task S3: Ruling: tape budget BINDCRAFT_MLX_TAPE_GIB, default 0.2 x cap (8 GiB without a cap), oldest dropped first; tapes released after the vjp that used them (JAX calls a forward's backward once) — an added knob, needed to bound memory when several states' forwards precede their vjps — cost if wrong: a too-small budget only costs speed (misses recompute).
- Task S3: Ruling: the vjp cache hit also requires equal feature shapes (stronger than before, for the recycle state too) — cost if wrong: none.
- Task S3: Ruling: the monomer's single_pass takes tapes= and refuses a non-None one by name (validation models have no gradient); the worker only makes tapes for the multimer family — cost if wrong: none.
- Task S3: Ruling: test_vjp_final_pass_uses_the_dropout_key_the_forwards_final_pass_used spied on single_pass, which the vjp's final pass no longer calls; rewritten to read the keys at embeddings.front and heads_of (both paths call them), keeping its three assertions per stage (two passes with keys, distinct per pass, vjp == forward); mutation 'final pass given the recycle key' caught — cost if wrong: none, the dropout oracle and the hit/miss dropout tests also pin it.
- Task S2: Ruling: the stack-level trace test now measures the direct path (taped forward + stack_vjp) at shapes no other test in the module uses; evoformer_stack(remat=True) under mx.vjp is inline by construction and stays correct (equality tests) — cost if wrong: none.
- Task S4: lead C confirmed on the new code (alternating, 2 rounds): 400-residue vjp 12.83 s (2 GiB cache) / 11.59 (4) / 11.21-11.25 (9) / 11.21 (unbounded, grows to 16.2 GiB). Ruling: the worker's allocator cache becomes 0.25 x cap (9 GiB at the default 36 GiB cap) via allocator_cache_gib (test RED -> GREEN); safety run: cap 16 GiB at 400 residues, 3 steps, no guard kill, active+cache peak 12.36 GiB; cap 36: peak 15.69 GiB, step 18.7 s (was 21.7) — cost if wrong: up to a quarter of the cap held as cache; MLX releases it before 0.9 x cap, measured.
- Final: Ruling: I1 (the tape budget bounds the tapes held between calls, not what one forward allocates) re-graded Minor — its proposed fix (evict before the forward too) is a no-op, because the held tapes are already within the budget whenever a forward starts; the memory peak is the vjp's, where the tape replaces the block-input storage the old remat path kept anyway (worker peak measured unchanged at 250 and 400 residues); in BindCraft's usual two states (the complex and the binder alone) the extra held tape during the other state's vjp is the complex's own, which its vjp needs — cost if wrong: a campaign with two LARGE states near the memory cap holds up to 0.2 x cap more than before during the other state's vjp and could hit the limit where it used to fit; deferred improvement: estimate the incoming tape from the feature shapes and evict room for it, or drop other calls' tapes when free memory is short.

## Deferred minors of the speed work (from its review)

- M1 evoformer_stack(remat=True, tape=) returns the tape's outputs after a shape-only check, so two same-length calls with different sequences would share a forward; reached only from tests (the model's vjp uses stack_vjp) — a trap for a future caller.
- M2 ArrayParams.has answers from the spec discovered on the first model of a stack config; a model with an optional tensor (a bias) the first one lacked would run without it; unreachable for the five multimer models (identical parameter sets).
- M3 the per-layer weight slices cached for the compiled blocks are copies beside the stacked originals: about +150 MB per resident model in bf16 (up to 5 models).
- M4 the compiled functions and MLX's per-shape traces are never evicted; a campaign sampling many lengths accumulates ~4 traces per new length (~0.03 s each); their memory is unmeasured.
- M5 test_the_models_vjp_runs_the_compiled_blocks_outside_any_transformation passes vacuously when BINDCRAFT_MLX_COMPILE=0 is set in the environment (it should force compile on and assert the first vjp traced).
- M6 on a tape miss the front (embeddings, templates) is computed twice (the forward's inputs, then inside its mx.vjp).
- M7 the reverse loop keeps every block input to the end; on a miss, releasing each as it is used would halve the loop's resident tape.
- M8 a vjp that raises leaves its call's tapes held until MAX_CALLS or the budget evicts them (the campaign is over at that point anyway).

## The MLX work, commit by commit

Speed work (2026-11-01), newest first:

- `15d56f6 tests: the model refuses a tape from another call; a reused call id never hits its tape`
- `4fe248c worker: allocator cache a quarter of the memory cap (was 2 GiB)`
- `af21d0b vjp: the final pass's backward as a chain outside any transformation`
- `a4d26bf tapes across the design forward and its vjp: model, bridge, worker`
- `cf28c34 stacks: explicit reverse loop with tapes; compiled blocks on the GPU`
- `d81422a tests: four assertions that never ran (assert_live(...) and comparison short-circuits on None)`
- `616c235 plan: MLX speed work (compiled block closures, explicit reverse loop, forward tape)`

The backend itself:

- `2440cc5 docs: fix call numbering in the design`
- `449b0b5 docs: phase-2 trigger for porting the CPU-side JAX, measured at Task 3 and Task 20`
- `b79568d docs: plan fixes after review (stub-remote seam proof, shared worker and lock, eager captures)`
- `48edd61 docs: implementation plan for the MLX AlphaFold backend`
- `c9059a1 docs: refine the MLX design's seam to a custom_vjp over the worker call`
- `288b8e0 docs: drop the Neural Engine from the MLX design, keep the measurements as a record`
- `5185d35 docs: correct the Neural Engine comparison for fp16 buffers`
- `be3336b docs: record Neural Engine spike B results in the MLX design`
- `842fbfd docs: fold builder-check findings and Neural Engine research into the MLX design`
- `73c9d54 docs: design for an MLX backend for the AlphaFold model`
- `413d977 mlx_af2: fixes from the whole-branch review`
- `ca6e463 docs: MLX AlphaFold backend guide and benchmarks; refuse the backend without mlx`
- `a547e54 mlx_af2: end-to-end comparison and parent device measurement`
- `f460395 mlx_af2: monomer structure module, heads and predictor support`
- `17fc697 mlx_af2: monomer trunk`
- `d5fc40e mlx_af2: GPU checkpoint numbers and an opt-in all-GPU test run`
- `04d09b7 mlx_af2: MlxAlphaFoldDesignModel on the real worker`
- `9a73e24 mlx_af2: bench_block.py, the Checkpoint 1 benchmark`
- `fe91cfb mlx_af2: vjp for the sequence-dependent features`
- `cb01796 mlx_af2: persistent MLX worker with memory guard and GPU lock`
- `01edac2 mlx_af2: output heads and the multimer model with recycling`
- `863b37d mlx_af2: multimer template embedding`
- `796ed92 mlx_af2: sidechains, atom positions and the full structure module`
- `cdded72 mlx_af2: rigid geometry, invariant point attention, fold iteration and the backbone loop`
- `4ae7f62 mlx_af2: multimer embeddings, recycle embedding, relative encoding and the stacks' assembly`
- `b3002c2 mlx_af2: Evoformer iteration and stacks with per-block remat and shared-axis dropout`
- `2f4b70b mlx_af2: MSA row, column and global attention and the outer product mean`
- `8f855da mlx_af2: triangle multiplication and triangle attention`
- `922d647 mlx_af2: linear, layer norm, attention, global attention and transition`
- `faede9f mlx_af2: npz weight loader with triangle fusing, accounting and a CPU device for unit tests`
- `b683947 mlx_af2: JAX reference captures at the model-call seam, helpers and the bf16 spread`
- `4c0a7e5 mlx_af2: custom_vjp bridge proven against a stub remote`
- `29e89e9 docs: plan passes the model to the worker as a runtime handle, not a trace-time name`
- `44b459c af2: model-output and parameter-loading hooks, no behaviour change`
- `f52e3c5 backend: BINDCRAFT_BACKEND switch, shared GPU lock, mlx and test extras`
- `6de2c35 docs: re-anchor the MLX design and plan on the merged upstream code`
