"""Real design states for the MLX backend's references and tests, built with the repo's own loaders (no synthetic coordinates)."""
import os

import jax
import jax.numpy as jnp

from bindcraft.af2 import (AlphaFoldDesignModel, interface_asym_ids, pad_design_chains, protein_state_shapes, real_residue_weights, renamed_protein_states, residue_chain_ids,
                           residue_entity_ids, canonical_chain_names, canonical_state_names)
from bindcraft.prediction import collect_shared_chains, concatenate_chain_arrays
from bindcraft.protein_preparation import initialize_design_trajectory
from bindcraft.settings import build_design_settings, load_settings

#large: hIL2R, two chains (196 + 191) with a 30-residue binder, about 420 residues
#small: hPDL1 (115) + a 30-residue binder = 145, bucket 160; medium: hIL7RA (195) + a 60-residue binder = 255, bucket 256
CASES = {'small': ('hPDL1', 30), 'medium': ('hIL7RA', 60), 'pair': (['hPDL1', 'mPDL1'], 30), 'large': ('hIL2R', 30)}  #pair: two states (human and mouse PD-L1) with different shapes, as in examples/pdl1_ortholog_pair.json
MULTIMER_MODEL = 'model_1_multimer_v3'

def params_path() -> str:
    return os.path.expanduser(os.environ.get('BINDCRAFT_AF2_PARAMS') or '~/.cache/bindcraft/alphafold')

def case_settings(case: str) -> dict:
    target, binder_length = CASES[case]
    return load_settings({'modality': 'binder', 'target': target, 'binder_lengths': [binder_length, binder_length]})

def design_inputs(case: str, seed: int = 0):
    #(design_settings, protein_states, multi_chain_binders, losses) exactly as a trajectory starts
    design_settings = build_design_settings(case_settings(case))
    protein_states, multi_chain_binders, losses = initialize_design_trajectory(design_settings, jax.random.PRNGKey(seed))
    return design_settings, protein_states, multi_chain_binders, losses

def build_model(model_class=AlphaFoldDesignModel, case: str = 'small', models: tuple[str, ...] = (MULTIMER_MODEL,), **overrides):
    #a deterministic model: dropout off, one recycle, the length bucket the repo uses for the case
    settings = case_settings(case)
    arguments = dict(target_pad_length=128 if case == 'pair' else 0, presets=models, data_dir=params_path(), max_cache_size=4, num_recycle=1, models=models, dropout=False, length_bucket_size=settings.get('length_bucket_size', 32) or 32)
    arguments.update(overrides)
    return model_class(**arguments)

def front_arguments(case: str, state: str | None = None, seed: int = 0) -> dict:
    #the concrete arrays sequence_gradients hands to the front (prepare_design_sequence_features, alphafold_input_features) for one state, built with the same helper calls
    design_settings, protein_states, multi_chain_binders, _ = design_inputs(case, seed)
    state = state or next(iter(protein_states))
    state_names = canonical_state_names(protein_states)
    chain_names = canonical_chain_names(protein_states, state_names)
    padded_states = pad_design_chains(renamed_protein_states(protein_states, state_names, chain_names), 32, 0)
    shapes = {name: (names, lengths) for name, names, lengths in protein_state_shapes(padded_states)}
    canonical = state_names[state]
    state_chain_names, chain_lengths = shapes[canonical]
    _, shared_chains = collect_shared_chains(padded_states)
    arrays = concatenate_chain_arrays(state_chain_names, padded_states[canonical], 'atoms', 'atom_mask', 'flags', 'residue_index')
    flags = arrays['flags']
    return {'sequence': jnp.concatenate([shared_chains[name].sequence for name in state_chain_names], axis=0), 'atoms': arrays['atoms'], 'atom_mask': arrays['atom_mask'], 'flags': flags,
            'residue_index': arrays['residue_index'], 'asym_id': residue_chain_ids(chain_lengths), 'entity_id': residue_entity_ids(state_chain_names, chain_lengths, multi_chain_binders),
            'interface_asym_id': interface_asym_ids(state_chain_names, chain_lengths), 'seq_mask': real_residue_weights(flags)}

def model_inputs(case: str = 'small', state: str | None = None, dropout: bool = False, large_complex_initialization: bool = False) -> dict:
    #the feature dict AlphaFold receives for one state, built eagerly with the repo's own front (no jit, so every value is a concrete array)
    from bindcraft.af2 import alphafold_input_features, prepare_design_sequence_features
    a = front_arguments(case, state)
    sequence_features, sequence_profile = prepare_design_sequence_features(a['sequence'], a['flags'], jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.01), jnp.asarray(2.0), None)
    return alphafold_input_features(sequence_features, sequence_profile, a['atoms'], a['atom_mask'], a['residue_index'], a['asym_id'], a['seq_mask'], a['flags'], jnp.asarray(dropout), 'direction', a['entity_id'], 0.0, large_complex_initialization)
