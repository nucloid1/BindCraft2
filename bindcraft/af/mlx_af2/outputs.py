#What the loss reads from AlphaFold's outputs, as flat names with '/' for nesting. No jax here: the worker imports this module.
OUTPUT_KEYS = ('structure_module/final_atom_positions', 'structure_module/final_atom_mask', 'predicted_lddt/logits', 'predicted_aligned_error/logits', 'predicted_aligned_error/breaks', 'distogram/logits', 'experimentally_resolved/logits')
ATOM_TYPES, PLDDT_BINS, PAE_BINS, DISTOGRAM_BINS = 37, 50, 64, 64

def output_shapes(length: int) -> dict[str, tuple[tuple[int, ...], str]]:
    #(shape, dtype name) of every output for a padded length of residues; the dtypes are checked against the JAX path in the reference captures
    return {'structure_module/final_atom_positions': ((length, ATOM_TYPES, 3), 'float32'), 'structure_module/final_atom_mask': ((length, ATOM_TYPES), 'float32'),
            'predicted_lddt/logits': ((length, PLDDT_BINS), 'float32'), 'predicted_aligned_error/logits': ((length, length, PAE_BINS), 'float32'),
            'predicted_aligned_error/breaks': ((PAE_BINS - 1,), 'float32'), 'distogram/logits': ((length, length, DISTOGRAM_BINS), 'float32'),
            'experimentally_resolved/logits': ((length, ATOM_TYPES), 'float32')}

def nest(flat: dict) -> dict:
    nested: dict = {}
    for name, value in flat.items():
        *parents, leaf = name.split('/')
        node = nested
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = value
    return nested

def model_family(model_name: str) -> str | None:
    #the families the MLX backend implements: the AlphaFold-Multimer models, and the monomer models with a predicted-aligned-error head (the pTM ones); anything else is None
    if 'multimer' in model_name:
        return 'multimer'
    return 'monomer' if model_name.endswith('_ptm') else None
