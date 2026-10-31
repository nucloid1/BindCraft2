"""Run predict and sequence_gradients of the JAX path on the small case (CPU, fixed keys, dropout off) and save everything they return. Run before and after a refactor of
af2.py: the two files must be identical. Usage: JAX_PLATFORMS=cpu python scripts/mlx_af2/capture_golden_jax.py [--out PATH]"""
import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests' / 'mlx_af2'))

import numpy as np

import cases
from bindcraft.protein import StructurePrediction

def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, StructurePrediction):
        for chain, protein in value.protein_complex.items():
            out[f'{prefix}/{chain}/atoms'] = np.asarray(protein.atoms)
            out[f'{prefix}/{chain}/atom_mask'] = np.asarray(protein.atom_mask)
        flatten(f'{prefix}/metrics', value.metrics, out)
    elif isinstance(value, dict):
        for name, item in value.items():
            flatten(f'{prefix}/{name}', item, out)
    else:
        out[prefix] = np.asarray(value)

def capture(case: str = 'small') -> dict:
    _, protein_states, _, losses = cases.design_inputs(case)
    model = cases.build_model(case=case)
    out: dict = {}
    flatten('predict', model.predict(protein_states, model=cases.MULTIMER_MODEL), out)
    predictions, gradients, loss = model.sequence_gradients(protein_states, losses, model=cases.MULTIMER_MODEL)
    flatten('grad/predictions', predictions, out)
    flatten('grad/sequence', gradients, out)
    flatten('grad/loss', loss, out)
    return out

if __name__ == '__main__':
    path = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else Path(os.path.expanduser(os.environ.get('BINDCRAFT_MLX_REFS', '~/.cache/bindcraft_mlx_refs'))) / 'golden_jax_small.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = capture()
    bad = [name for name, array in arrays.items() if not np.isfinite(array.astype(np.float64)).all()]
    assert not bad, f'non-finite values in {bad}'
    np.savez(path, **arrays)
    print(f'wrote {len(arrays)} arrays to {path}')
