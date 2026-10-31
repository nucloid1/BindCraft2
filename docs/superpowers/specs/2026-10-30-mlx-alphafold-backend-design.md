# BindCraft2: an MLX backend for the AlphaFold model

Status: design, awaiting Ethan's approval. Claims are tagged VERIFIED (a command ran and its output was read), INFERRED (code reading or
arithmetic, not measured) or UNKNOWN.

## Intent and why

This fork already runs AlphaFold 2 on the Apple GPU through the third-party `metaljax` JAX plugin (commit b4556f7; the README measures about 3.5x
over the same code on the CPU). Ethan wants the Mac path to run on Metal through MLX instead of through that plugin, for speed and so the Mac path
no longer depends on a plugin that is itself experimental. Earlier MLX ports of ESMFold2, Mol-1 and RFD4-Proteina reached 1.6 to 2.3x over
torch on MPS; none was measured against a JAX Metal path, so the gain here is a hypothesis, not a promise.

## Decided by Ethan

- **Scope: the AlphaFold model only.** Its forward pass and its gradient run in MLX in a worker process. The design loop, the 907-line loss
  module (`bindcraft/loss.py`), the sequence optimizer, ProteinMPNN and the filters stay as they are.
- **MLX is a new backend added beside the existing ones.** CUDA, ROCm, oneAPI and metaljax keep working and are the parity reference.
- **bf16, as today.** AlphaFold runs in bf16 now (`global_config.bfloat16 = True`); the MLX model mirrors it. fp32 is kept as an option for parity
  testing.
- **No stop rule.** Speed is measured and reported at every checkpoint but never stops the work (Ethan wants JAX and the plugin off the Mac path
  regardless).
- **No Neural Engine (decided 2026-10-30, "continue without the ANE").** Ethan asked for it "if possible" twice; a measured spike showed the GPU faster on every
  AlphaFold-shaped op tested, so it is out of every phase. The section "Neural Engine" is kept as a record.

Standing rules from earlier work: nothing in a repo this work does not own is modified; one GPU run at a time (the shared lock
`~/.cache/esmfold2_mlx/gpu.lock`); disk under 90% unless Ethan opts in (`..._MIN_FREE_GB`); never a silent fallback to another backend.

## What exists today (the facts the design rests on)

- **Where AlphaFold is called (VERIFIED, `bindcraft/af2.py`).** `AlphaFoldDesignModel` implements `DifferentiableProteinPredictor`
  (`predict(...)` and `sequence_gradients(...)`, `bindcraft/prediction.py`) and is built in `bindcraft/campaign.py:262,265` for the design models and
  the validation models. The vendored model is `bindcraft/af/alphafold/` (11.6k lines of JAX and haiku).
- **The gradient seam (VERIFIED).** `sequence_design_loss` is one `jit(value_and_grad(..., argnums=2))` over three parts: (1) sequence logits to model
  input features (`prepare_design_sequence_features`, `alphafold_input_features`: a one-row `msa_feat` of 49 channels, `target_feat`, template
  features of the fixed target, offsets, chain ids, a dropout flag); (2) the model (`recycled_alphafold_outputs`); (3) metrics and the design loss
  from the outputs. Only part 2 is AlphaFold. Every state of a design (complex, binder alone, extra targets) is one model call with its own shapes.
- **Recycling (VERIFIED).** The model config has `num_recycle = 0`; recycles are run by hand and every pass except the last is under
  `stop_gradient`, so the gradient flows through one pass only. `design_recycles` is 1 and `validation_recycles` 3 by default.
- **What the loss reads from the model (VERIFIED).** `structure_module.final_atom_positions` and `final_atom_mask`, `predicted_lddt.logits`,
  `predicted_aligned_error.logits` and `breaks`, `distogram.logits`, `experimentally_resolved.logits`, and `prev` (the recycle state).
- **The model (VERIFIED, `config.py`).** 48 Evoformer blocks (MSA width 256, pair width 128), a 4-block extra-MSA stack (one extra row), a template
  stack for the fixed target, an 8-layer structure module with invariant point attention, and the heads. Two families are used: `multimer_v3`
  (five models) and the monomer pTM models (`model_1_ptm`, `model_2_ptm`), selected by `select_design_and_validation_models`.
- **Precision and memory switches (VERIFIED).** `bfloat16 = True` (parameters and most compute in bf16 through a haiku `bfloat16_context`, with
  parts of the structure module cast explicitly), `use_remat = True` (per-block rematerialisation), `subbatch_size = auto` (chunking over rows once
  a complex has more than 384 residues). Dropout is on during design (`design_dropout`).
- **Padding (VERIFIED).** Lengths are padded to buckets of 32 and padded residues are masked ("dead" residues), so shapes repeat across designs.
- **The numbers on this machine (VERIFIED, my micro-benchmark, fp32, N=256, pair width 128).** Raw 4096-square matmul: 11.4 TFLOP/s on both JAX
  Metal and MLX (bf16: 12.9 on both). On pair-sized operations MLX is faster: triangle contraction 0.81 vs 1.35 ms, pair projection 8.5 vs 4.3
  TFLOP/s, elementwise gating 0.39 vs 0.88 ms, layer norm 0.91 vs 1.23 ms. These are single operations, not the full model.
- **Time split (INFERRED).** The repository does not import PyRosetta; AlphaFold forward and gradient steps plus the validation refolds are most
  of a design's wall time. Not measured.
- **Expected gain (INFERRED).** About 1.5 to 2x on the AlphaFold compute. UNKNOWN until a block is measured.

## Goals

1. `bindcraft design ...` on an Apple silicon Mac can run with the MLX backend selected, producing designs through the same loop, losses, MPNN
   stage and filters as today.
2. The MLX model's forward outputs and its sequence gradient agree with the repo's own JAX path (fp32 against JAX on the CPU; bf16 within the spread
   JAX's own bf16 has against its fp32).
3. Faster than the current metaljax path, measured and reported honestly at each checkpoint (one block, the forward pass, forward plus gradient,
   a full trajectory, the validation refold set), with memory beside it.

## Non-goals

- The Neural Engine, in any phase (see the record below). The loss, ProteinMPNN, the optimizer and the filters in MLX in this phase; Ethan's rule (2026-10-30): if the CPU-side JAX work exceeds 20% of the MLX backend's wall time, it is ported in a second phase, otherwise not (the 20% is the author's call). Training, new modalities,
  changing any loss or setting. Multi-GPU. Running on Intel Macs. Replacing or removing the JAX paths.

## Anti-goals (a failure even if it technically works)

- Designs that differ from the JAX path for a reason that is not numerics (a changed loss, a dropped template, a different recycle rule).
- Any silent fallback: a model family the MLX backend does not implement, a dead worker or a memory-cap hit must stop the run with a message that
  names the cause, never continue on JAX.
- Breaking or slowing the existing backends; the MLX code must be unreachable unless selected.
- Claiming parity from a single bf16 trajectory (the README records that bf16 trajectories are noisy and not reproducible between backends).

## Shape of the solution

- **A subclass of the existing predictor, chosen by one switch.** `MlxAlphaFoldDesignModel(AlphaFoldDesignModel)` overrides three hooks that `af2.py` gains in a
  behaviour-preserving refactor (loading parameters, building the runner, producing model outputs). The switch is the environment variable `BINDCRAFT_BACKEND=mlx` (my call:
  JAX's platform is fixed at import, before any settings file is read, so a settings key cannot select it). Opt-in; making it the Apple default is Ethan's decision after
  verification.
- **MLX in a worker process.** The model (embeddings, template stack, extra-MSA stack, Evoformer, structure module, heads) runs in a persistent
  child process that never imports JAX; arrays cross by shared memory. This is the pattern of the earlier ports (an MLX process that has done
  other CPU-side framework work ran about a third slower in Mol-1's measurements, cured by a process boundary). The parent keeps JAX, forced onto the
  CPU platform on this backend, for the front (sequence to features), the loss and the optimizer; the GPU then belongs to MLX alone, and the metaljax
  plugin is not needed on this path. **My call, and the one to check first:** ProteinMPNN, the filters and the rest of `bindcraft/` are JAX too
  (`proteinmpnn.py`, `MPNN_stage.py`, `filters.py`), so they move to the CPU with the parent, at a cost nobody has measured. It is measured at the
  first checkpoint. A setting can leave the parent on metaljax instead (the plugin then stays a dependency and the two processes share the GPU). The cross-boundary arrays are small (logits and positions of the order of 10 MB per step at 300 residues).
- **The call contract (mechanism refined while planning; same semantics).** The one call that runs AlphaFold inside the jitted loss, `recycled_alphafold_outputs`, becomes a
  JAX `custom_vjp` whose forward and backward each call the MLX worker through `jax.pure_callback`. Forward: all recycles (no gradient) and the final pass, returning only the
  outputs the loss reads. Backward: the worker recomputes the final pass and applies the vjp to JAX's cotangents, returning gradients for the two float features that depend on the
  sequence (`msa_feat`, `target_feat`). JAX's own autodiff then yields forward, loss, backward with the front, the loss and the optimizer untouched, and the extra forward per step
  the design already accepts. The worker may cache the recycle state from the forward by call id, and recomputes when the cache misses, so a miss costs time and never correctness.
- **Weights.** Read straight from the existing `.npz` files at worker start (no separate converter; my call), with the triangle-projection fusing that
  `get_model_haiku_params(fuse=True)` does, and an accounting check that every tensor is used exactly once. Locations: `~/.cache/bindcraft/alphafold` (multimer v3) and
  `~/.cache/colabfold/params` (pTM); `BINDCRAFT_AF2_PARAMS` overrides.
- **Numerics.** Mirror what JAX does: bf16 parameters and compute where the haiku context applies it, fp32 where the structure module casts
  explicitly, fp32 for softmaxes and norms exactly where the JAX code has them. Rematerialisation per block with `mx.checkpoint`, row chunking with
  the same `subbatch_size` rule. Dropout uses MLX's RNG (not bit-identical to JAX's; parity is checked with dropout off).
- **Families (VERIFIED in `settings.py`, `campaign.py`).** Design draws only from the multimer pool and `sequence_gradients` is called only on design
  models, so multimer v3 is the only family that needs a vjp. The monomer pTM family is forward-only (validation) but is needed for any default run:
  with no scaffold and one copy, `resolve_validation_model` picks the monomer pool (`settings.py:590-593`). Order (my call): multimer first, with
  `trajectory_only` or `validation_model: multimer` as the first end-to-end milestone; the monomer forward next, which makes the default run end to end.
  A family not yet implemented is refused by name. Whether validation runs with dropout off is read from `campaign.py` by the builder (not checked here).
- **What the predictor must reproduce (VERIFIED in code).** (1) The JAX path casts predicted atom positions to float16 before the loss
  (`af2.py:281,343`); that cast stays in the JAX tail. (2) The campaign sets `provides_distogram`, `num_recycle`, `target_flexibility` and
  `bigbang_initialization` on the model after construction (`campaign.py:61-75`); each must change what the MLX path does, as the JAX path keys its compile
  cache on them. (3) `sequence_gradients(..., compile_only=True)` is called from a daemon thread to precompile the next length bucket
  (`campaign.py:101-110`); on MLX it must be a warm-up or queue behind the worker's other requests, never a concurrent second request.
- **Failure behaviour.** A dead worker or a cap hit raises an error that names the cause and the cap, exits non-zero and leaves no shared memory behind;
  the run is never continued on another backend.
- **Operations.** The shared GPU lock for the whole run, the preflight (disk, heavy processes), and a worker memory cap, as in the earlier ports.

## Neural Engine (set aside; kept as a record)

Out of scope at Ethan's direction on 2026-10-30. Recorded so the question is not reopened without the evidence.

- **Ways to reach it.** Core ML (public, fp16, inference). Core AI, Apple's successor: VERIFIED present as `CoreAI.framework` in the macOS 27 SDK; it names the Neural
  Engine as a compute unit and exports no symbol containing gradient, backward, autodiff or train (UNKNOWN: its operation coverage and precision). Private interfaces:
  VERIFIED present (`AppleNeuralEngine.framework`, `ANECompiler.framework`); REPORTED in the open-source `maderix/ANE` (its own README): it runs forward and activation-gradient
  passes on the chip for small transformers, tested on M4 only. "Inference only" describes Core ML's policy, not the chip.
- **Spike B (private route) was run on 2026-10-30**, a throwaway harness outside this repo, on this M3 Max (macOS 27.0.1): the `maderix/ANE` bridge built from source,
  every result checked live and against a numpy fp32 reference. It works on an M3 Max, with relative error about 3e-4 on convolutions and the contraction (random unit-scale
  data, not AlphaFold activations). The project's own peak probe was not trusted: it never checks eval success or output.
- **Result (MEASURED, N=256, kernel time, Neural Engine with fp16 buffers against MLX fp16 on the GPU):** transition 2.84 vs 1.86 ms (GPU 1.5x faster); 1x1 projection
  0.59 vs 0.37 (1.6x); backward-style conv 1.69 vs 0.88 (1.9x); triangle contraction 1.56 vs 0.55 (2.8x); layer norm plus gate 4.02 vs 1.37 (2.9x, fp32 buffers). Dense
  compute is about 6.0 TFLOP/s on the Neural Engine against the GPU's 9.2. A first run with fp32 buffers overstated the gap (up to 3.9x) and was corrected. The fp16
  rerun ran while another job held the GPU, so the Neural Engine figures may be a few percent pessimistic.
- **Concurrency (MEASURED, fp32 Neural Engine buffers, one 8 s repeat):** transition, both units kept 99% and 95% of solo speed, 1.6x combined; contraction, the GPU fell to
  78%, about 1.0x combined. A fp16 rerun was not possible (the GPU lock was taken).
- **Limits found:** rows must be a multiple of 64 bytes; about 125 programs can be loaded at once; fp16 only (the repo runs bf16, so overflow on real activations is untested).
- **Not done:** Spike A (public route), a whole Evoformer block, energy measurement, and the fp16 concurrency rerun.

## Verification

- **Reference:** the repo's own JAX path, on the CPU in fp32 for parity and on metaljax in bf16 for "today". Captures of real inputs and
  outputs on the CPU (as in the RFD4 port), per stage: embeddings, one Evoformer block, the Evoformer stack, the structure module, the heads, the full
  forward, and the sequence gradient.
- **Gates (fp32, MLX model with `bfloat16 = False`):** per-stage relative error 1e-5 where the stage is a plain composition of matmuls, norms and
  gates; composite (full forward outputs, positions) 1e-3; sequence gradient cosine at least 0.999 against JAX on the same inputs (the README's own
  JAX Metal against CPU figure is 0.9997). Each reference is checked live (never all zeros) and every weight is accounted for.
- **bf16 (the default):** judged against JAX's own spread (JAX bf16 against JAX fp32 on identical inputs): the MLX bf16 outputs and gradient must
  lie within that spread. Trajectory-level checks compare distributions over several seeds (final design loss, pLDDT, ipTM), not single runs.
- **Tests that name the failure modes:** padded and dead residues give identical results at valid positions; dropout off is deterministic; a worker
  that is killed or hits its cap fails loudly with the cap named; every model family and recycle count in the pool; multi-state designs (complex plus
  binder alone); chunking above 384 residues equals the unchunked result.
- **Measured at checkpoints, reported, never a stop:** one block; the model forward at 150, 250 and 400 residues; forward plus gradient; one full
  trajectory; the validation refold set; each beside the current metaljax path (warm, second run in a process, 10% noise floor) with peak memory.

## Risks and unknowns

1. Backward through 48 Evoformer blocks with MLX autograd and `mx.checkpoint` at 250 to 400 residues: time and memory unproven (UNKNOWN).
2. The extra forward per step may eat a fifth to a quarter of the gain (INFERRED).
3. The structure module and the all-atom code are many small operations; MLX could be slower than JAX's jit there until they are compiled or fused
   (UNKNOWN).
4. bf16 fidelity: XLA and MLX may accumulate reductions differently; the yardstick above is how this is judged, and exact agreement is not promised.
5. The template stack for the fixed target and the optional `initial_atom_pos` inputs must both be ported (they are used in design).
6. The monomer family doubles part of the structure-module work.
7. The time share of AlphaFold in a full design is not measured; if it is smaller than assumed the end-to-end gain shrinks.
8. Moving the parent's JAX to the CPU slows ProteinMPNN and the filters by an unmeasured amount (see the parent-device call).

## Left to the builder

The module layout (for example `bindcraft/af/mlx/`), the worker protocol details, the converter format, test organisation, whether and where to use
`mx.compile`, and kernel questions (a hand-written Metal kernel only under the earlier rule: an operation class at least a quarter of the time and
not already near the GPU's matmul ceiling).

## Calls the author made (for Ethan to check)

1. The parent's JAX runs on the CPU on this backend, with a setting to keep it on metaljax, decided by a measurement at the first checkpoint.
2. Multimer first, monomer forward second; `trajectory_only` or `validation_model: multimer` is the first end-to-end milestone.
3. The Neural Engine is out of scope (Ethan's decision after Spike B showed the GPU faster on every op tested); the results are kept above as a record.
4. The work lives on branch `mlx-backend` in `~/BindCraft2`, off `apple-silicon-metal`, committed locally, nothing pushed.
5. The extra forward pass per gradient step is accepted as the price of keeping the loss in JAX.
6. The seam is a `custom_vjp` over `pure_callback` and the switch is `BINDCRAFT_BACKEND=mlx`, not a settings key or a predictor that orchestrates the three phases itself.
7. The Evoformer blocks are shared by both families (they live in `modules.py`), so they are ported once and the monomer family needs only its own embeddings, templates, structure module and heads.
8. The 20% threshold for the phase-2 trigger (Ethan said "if cpu jax is a bottleneck" without a number), measured first with an instant stub at Task 3 and finally in Task 20.
