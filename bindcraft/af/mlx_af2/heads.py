#The output heads the loss reads (modules.PredictedLDDTHead, PredictedAlignedErrorHead, DistogramHead, ExperimentallyResolvedHead). The masked-MSA head is not ported: nothing reads it.
import mlx.core as mx

from bindcraft.af.mlx_af2 import layers

def predicted_lddt(p, representations: dict, cfg: dict) -> dict:
    act = layers.layer_norm(p.scope('input_layer_norm'), representations['structure_module'])
    act = mx.maximum(layers.linear(p.scope('act_0'), act), 0)
    act = mx.maximum(layers.linear(p.scope('act_1'), act), 0)
    return {'logits': layers.linear(p.scope('logits'), act)}

def predicted_aligned_error(p, representations: dict, cfg: dict) -> dict:
    return {'logits': layers.linear(p.scope('logits'), representations['pair']), 'breaks': mx.linspace(0.0, cfg['max_error_bin'], cfg['num_bins'] - 1)}

def distogram(p, representations: dict, cfg: dict) -> dict:
    half = layers.linear(p.scope('half_logits'), representations['pair'])
    return {'logits': half + half.swapaxes(-2, -3), 'bin_edges': mx.linspace(cfg['first_break'], cfg['last_break'], cfg['num_bins'] - 1)}

def experimentally_resolved(p, representations: dict, cfg: dict) -> dict:
    return {'logits': layers.linear(p.scope('logits'), representations['single'])}
