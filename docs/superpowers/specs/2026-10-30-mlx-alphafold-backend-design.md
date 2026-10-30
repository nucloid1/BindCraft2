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
- **Neural Engine, "if possible".** See "Neural Engine" below: it is not part of this port, and an optional later phase is written down with a
  feasibility gate.

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

- The Neural Engine (see below). The loss, ProteinMPNN, the optimizer and the filters in MLX (a possible later phase). Training, new modalities,
  changing any loss or setting. Multi-GPU. Running on Intel Macs. Replacing or removing the JAX paths.

## Anti-goals (a failure even if it technically works)

- Designs that differ from the JAX path for a reason that is not numerics (a changed loss, a dropped template, a different recycle rule).
- Any silent fallback: a model family the MLX backend does not implement, a dead worker or a memory-cap hit must stop the run with a message that
  names the cause, never continue on JAX.
- Breaking or slowing the existing backends; the MLX code must be unreachable unless selected.
- Claiming parity from a single bf16 trajectory (the README records that bf16 trajectories are noisy and not reproducible between backends).

## Shape of the solution

- **A new predictor class** implementing the same protocol as `AlphaFoldDesignModel`, chosen where `campaign.py` builds the model, by an explicit
  backend setting (opt-in at first; making it the Apple default is Ethan's decision after it is verified).
- **MLX in a worker process.** The model (embeddings, template stack, extra-MSA stack, Evoformer, structure module, heads) runs in a persistent
  child process that never imports JAX; arrays cross by shared memory. This is the pattern of the earlier ports (an MLX process that has done
  other CPU-side framework work ran about a third slower in Mol-1's measurements, cured by a process boundary). The parent keeps JAX, forced onto the
  CPU platform on this backend, for the front (sequence to features), the loss and the optimizer; the GPU then belongs to MLX alone, and the metaljax
  plugin is not needed on this path. The cross-boundary arrays are small (logits and positions of the order of 10 MB per step at 300 residues).
- **The call contract.** `forward(model, features, dropout/rng) -> outputs` and `vjp(model, features, cotangents) -> d(features)`. JAX computes the
  loss from the outputs and its gradient with respect to the outputs (the cotangents); MLX returns the gradient with respect to the float input
  features (`msa_feat`, `target_feat`, ...), which JAX chains into the sequence logits through the front part with `jax.vjp`. Because the cotangents
  depend on the outputs, a gradient step costs a forward pass, then the vjp (which recomputes the forward inside, as the rematerialising JAX path does):
  one extra forward per step (INFERRED, about a quarter of a step). Moving the loss into MLX would remove it; that is the larger scope Ethan
  declined for now.
- **Weights.** A converter turns the haiku `.npz` parameters (with the triangle projections fused as upstream's `fuse=True` does) into MLX files,
  bit for bit, with a manifest that accounts for every tensor, as in the earlier ports. AlphaFold parameters are read from the existing locations
  (`~/.cache/bindcraft/alphafold` has the multimer v3 files, `~/.cache/colabfold/params` the pTM files; `BINDCRAFT_AF2_PARAMS` overrides).
- **Numerics.** Mirror what JAX does: bf16 parameters and compute where the haiku context applies it, fp32 where the structure module casts
  explicitly, fp32 for softmaxes and norms exactly where the JAX code has them. Rematerialisation per block with `mx.checkpoint`, row chunking with
  the same `subbatch_size` rule. Dropout uses MLX's RNG (not bit-identical to JAX's; parity is checked with dropout off).
- **Families.** Multimer v3 first (the design default), then the monomer pTM models. A family not yet implemented is refused by name.
- **Failure behaviour.** A dead worker or a cap hit raises an error that names the cause and the cap, exits non-zero and leaves no shared memory behind;
  the run is never continued on another backend.
- **Operations.** The shared GPU lock for the whole run, the preflight (disk, heavy processes), and a worker memory cap, as in the earlier ports.

## Neural Engine

- **VERIFIED (what MLX supports):** MLX targets the CPU and the GPU only.
- **Known (platform facts, not measured here):** the Neural Engine is reached through Core ML (or undocumented private interfaces, which this work will
  not use); it runs inference only (no gradients), computes in fp16 (no fp32), prefers fixed shapes and a restricted operation set.
- **Consequence:** the design loop, which needs the gradient through the whole model on every step, cannot run on it. Only forward-only work could:
  the validation refolds (3 recycles over several models per candidate).
- **Optional later phase, gated:** a feasibility spike after the MLX backend has parity: convert one Evoformer block to Core ML (a rewrite of the
  model into Core ML's format, not an MLX feature), run it in fp16 on the Neural Engine, and check (a) fp16 numerics against the fp32 reference
  (AlphaFold's pair activations and attention logits are the risk), (b) that it is faster than the GPU at the shapes in use, (c) whether running
  validation on it concurrently with GPU design work shortens a campaign. Realistic upside: concurrency and power, not raw speed. Not started
  unless the spike is approved separately.

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

## Left to the builder

The module layout (for example `bindcraft/af/mlx/`), the worker protocol details, the converter format, test organisation, whether and where to use
`mx.compile`, and kernel questions (a hand-written Metal kernel only under the earlier rule: an operation class at least a quarter of the time and
not already near the GPU's matmul ceiling).
