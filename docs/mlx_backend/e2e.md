# End to end: real campaigns on the JAX path and the MLX backend (2026-10-31)

Machine: M3 Max, 48 GB (about 36 GB usable), macOS 27.0.1, MLX 0.32.4, metaljax 0.11.9. Target `hPDL1` (115 residues), binder length 80, `examples/pdl1_denovo.json`, one trajectory per campaign,
trajectory-only unless stated, run by `scripts/mlx_af2/e2e_compare.py` under the shared GPU lock (the raw results are in `docs/mlx_backend/e2e_results/`). The JAX path is the repo as shipped on this machine
(metaljax on the Apple GPU); the MLX backend is `BINDCRAFT_BACKEND=mlx` with the parent's JAX on the CPU (the default, see below). The step budget is short on purpose (screen 15, refine 8, anneal 12, harden 3 rounds), so
what is compared is the machinery and the distributions, not the quality of a finished design.

## JAX against MLX, five seeds each

| | JAX (metaljax) | MLX backend |
|---|---|---|
| exit codes | 0 x5 | 0 x5 |
| design seconds per gradient round | **8.87 ± 0.18** (8.75 to 9.17) | **6.92 ± 0.11** (6.85 to 7.12) |
| gradient rounds run (trajectories stop at different stages) | 15, 15, 15, 15, 38 | 15, 15, 35, 35, 38 |
| final-round ipTM | 0.29 ± 0.28 (0.16 to 0.79) | 0.25 ± 0.11 (0.16 to 0.44) |
| reported pLDDT | 0.52 ± 0.21 (0.31 to 0.86) | 0.63 ± 0.18 (0.36 to 0.80) |
| weighted loss (final gradient round) | 7.1 ± 1.9 (3.8 to 8.3) | 6.7 ± 0.9 (5.8 to 8.1) |

**Speed.** MLX is 1.28x faster per gradient round at 195 residues, and the ranges are disjoint. The figure does not move with the length of the trajectory (15 rounds or 38), so compile time is not what is measured; the
two trajectories that went on to the mutate stage (the slowest of each backend) carry that stage's time in their total, and on the screen-only trajectories alone the ratio is 8.79 s against 6.89 s, also 1.28x. It
agrees with the whole-step benchmark (1.37x at 250 residues, `checkpoints.md`) and is smaller than the 1.75x at 400 residues, as the memory-bound metaljax path loses more there. Total design time is not a fair comparison
here, because the stage thresholds stop trajectories at different stages.

**Outcomes.** The ranges of ipTM, pLDDT and loss overlap (on 44%, 80% and 53% of their union). bf16 trajectories are noisy and seeds do not pair across backends, so the check is distributional and nothing more is claimed:
five seeds cannot show that the two backends design equally well, only that neither is visibly broken. One difference to watch: 1 of 5 JAX trajectories got past the screen stage against 3 of 5 MLX ones. That is not distinguishable
from chance at this size (two-sided Fisher exact p = 0.52), but it is the first thing a larger run (10 to 20 seeds, the default budget) should look at; a systematic lead for MLX would need an explanation, because the MLX
model's bf16 is strict (rounds at every op) where the JAX CPU model's is not (`checkpoints.md`).

## The parent's device

The whole pipeline (design stages, ProteinMPNN, the monomer validation models, the filters), MLX backend, stage thresholds at zero so a short trajectory reaches every phase, two seeds each:

| parent JAX on | campaign wall time (s) | design stages (s) |
|---|---|---|
| cpu | 163.0, 158.6 | 88.0, 85.3 |
| metal | 164.0, 168.0 | 87.7, 81.0 |

No difference beyond the noise (3% slower on metal in the mean). **Decision: the parent's JAX runs on the CPU when the backend is mlx** (`import bindcraft` sets `JAX_PLATFORMS=cpu` unless the user sets it; tested
in `test_backend.py`), which leaves the GPU to the MLX worker at no cost; `JAX_PLATFORMS=metal` is the override.

## Phase-2 trigger (port the CPU-side JAX work if it exceeds 20% of MLX-backend wall time): NOT met

- Front and loss per gradient round: 9, 19 and 33 ms at 147, 259 and 419 residues (Checkpoint 0, `bench_seam.py`), about 14 ms at 195 residues, against about 6,900 ms per round: **0.2%**.
- After the trajectory, 73 s of the 163 s wall time of this tiny-budget campaign (45%), but that is validation, on MLX: a monomer validation forward (bf16, three recycles) takes 2.15 s at 147 residues and 7.75 s at 259 (about
  4 s at 195), each of the ten candidates is scored by two models as a complex and as a binder alone, and the files written after validation account for 7 s of the 73. ProteinMPNN plus the filters, the CPU-side JAX share, is therefore
  at most about 10 s per trajectory (the file times cannot split ProteinMPNN from validation exactly), against about 860 s of design at the default budget (125 gradient rounds x 6.9 s): **about 1%**. Even against this tiny
  budget's 163 s it is 6%.

## What the tests establish (`tests/mlx_af2/test_e2e.py`, marked gpu, 4 passed)

- A trajectory-only run with `validation_model: multimer` exits cleanly (design pool cut to three models so multimer models are left to validate on).
- The default run completes with the monomer validation pool: ProteinMPNN redesigns, the validation models score them, the filters run, the campaign ends normally.
- No silent fallback: the MLX worker is killed (SIGKILL) in the middle of a trajectory; the campaign exits nonzero, says `MLX worker` and `exit code -9`, and does not print `campaign done`.
- Without `BINDCRAFT_BACKEND` the campaign runs the JAX path, starts no MLX worker and prints nothing of the MLX session.

## Limits of this evidence

Five seeds, one trajectory each, a short budget and designs far below acceptance: the quality of finished designs, the default-budget wall time and the stage survival rates are not measured here. Seeds do not pair across
backends. The JAX path is the metaljax one; a CUDA machine would be a different comparison. The weighted loss is the sum of weight x term over the last gradient round's terms as the log prints the weights.
