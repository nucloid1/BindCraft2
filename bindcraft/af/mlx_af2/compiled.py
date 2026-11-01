#Compiled Evoformer blocks for the GPU. mx.compile fuses a block's many small elementwise operations (the layer norms, the gating, the residual adds); measured on the M3 Max in bf16, a block's forward is 26 to 32%
#faster and its forward + backward 22 to 26% faster, with results equal to the uncompiled ones (docs/mlx_backend/checkpoints.md). Two things make it pay:
#- the block's parameters are passed in as a list of arrays (ArrayParams serves them by name) instead of being read from Params inside the function, so one compiled function serves every layer of a stack and
#  traces once per shape, not once per layer;
#- the vjp closure is compiled as a whole, so the backward pass is fused too (compiling the block and differentiating it afterwards fuses only the forward).
#On by default on the GPU only: on the CPU, MLX builds fused kernels with a C++ compiler, seconds per kernel. BINDCRAFT_MLX_COMPILE=0 turns it off.
import collections
import json
import os
import weakref

import mlx.core as mx

COMPILE_ENV = 'BINDCRAFT_MLX_COMPILE'
TRACES: collections.Counter = collections.Counter()  #how often a compiled function's body ran ('forward', 'backward'): once per new shape; diagnostics and tests

def enabled() -> bool:
    return os.environ.get(COMPILE_ENV, '').strip() != '0' and mx.default_device().type == mx.gpu

class Recorder:
    #serves the real parameters of one layer and logs every (name, dtype) the block reads, in order: the block's parameter spec
    def __init__(self, inner, log: dict, prefix: str = ''):
        self.inner, self.log, self.prefix = inner, log, prefix

    def scope(self, prefix: str) -> 'Recorder':
        return Recorder(self.inner, self.log, f'{self.prefix}{prefix.strip("/")}/')

    def get(self, path: str, dtype: mx.Dtype | None = None) -> mx.array:
        dtype = dtype or self.inner.store.dtype
        self.log[(self.prefix + path, dtype)] = None
        return self.inner.get(self.prefix + path, dtype)

    def has(self, path: str) -> bool:
        return self.inner.has(self.prefix + path)

class ArrayParams:
    #the Params interface a block uses (scope, get, has) over a dict {(name, dtype): array}; inside a compiled function the arrays are its inputs
    def __init__(self, arrays: dict, default_dtype: mx.Dtype, prefix: str = ''):
        self.arrays, self.default_dtype, self.prefix = arrays, default_dtype, prefix

    def scope(self, prefix: str) -> 'ArrayParams':
        return ArrayParams(self.arrays, self.default_dtype, f'{self.prefix}{prefix.strip("/")}/')

    def get(self, path: str, dtype: mx.Dtype | None = None) -> mx.array:
        key = (self.prefix + path, dtype or self.default_dtype)
        if key not in self.arrays:
            raise KeyError(f'parameter {key[0]!r} ({key[1]}) is not in the block\'s parameter spec')
        return self.arrays[key]

    def has(self, path: str) -> bool:
        return any(name == self.prefix + path for name, _ in self.arrays)

def discover(p, cfg: dict, msa, pair, msa_mask, pair_mask, *, is_extra: bool, with_key: bool, subbatch_size: int | None = None) -> list[tuple[str, mx.Dtype]]:
    #the (name, dtype) of every parameter one block of the stack p reads, from building (never evaluating) the graph of its first layer on the given inputs
    from bindcraft.af.mlx_af2 import evoformer
    log: dict = {}
    evoformer.evoformer_iteration(Recorder(p.layer(0), log), msa, pair, msa_mask, pair_mask, cfg, dropout_key=mx.random.key(0) if with_key else None, is_extra=is_extra, subbatch_size=subbatch_size)
    return list(log)

_LAYER_WEIGHTS: 'weakref.WeakKeyDictionary' = weakref.WeakKeyDictionary()  #Store -> {(prefix, layer, spec): [arrays]}: the layer slices are made once per model, not once per call

def layer_weights(p, index: int, spec: list) -> list[mx.array]:
    cache = _LAYER_WEIGHTS.setdefault(p.store, {})
    key = (p.prefix, index, tuple(spec))
    if key not in cache:
        layer = p.layer(index)
        cache[key] = [layer.get(name, dtype) for name, dtype in spec]
    return cache[key]

class StackFunctions:
    #the compiled forward and vjp of one block, for every layer of one stack configuration
    def __init__(self, spec: list, default_dtype: mx.Dtype, cfg: dict, is_extra: bool, subbatch_size: int | None):
        from bindcraft.af.mlx_af2 import evoformer
        self.spec = spec

        def block(weights, msa, pair, msa_mask, pair_mask, key):
            return list(evoformer.evoformer_iteration(ArrayParams(dict(zip(spec, weights)), default_dtype), msa, pair, msa_mask, pair_mask, cfg, dropout_key=key, is_extra=is_extra, subbatch_size=subbatch_size))

        def forward(weights, msa, pair, msa_mask, pair_mask, *key):
            TRACES['forward'] += 1
            return block(weights, msa, pair, msa_mask, pair_mask, key[0] if key else None)

        def backward(weights, msa, pair, cotangent_msa, cotangent_pair, msa_mask, pair_mask, *key):
            TRACES['backward'] += 1
            return mx.vjp(lambda m, q: block(weights, m, q, msa_mask, pair_mask, key[0] if key else None), [msa, pair], [cotangent_msa, cotangent_pair])[1]

        self.forward, self.backward = mx.compile(forward), mx.compile(backward)

_FUNCTIONS: dict = {}

def functions_for(p, cfg: dict, msa, pair, msa_mask, pair_mask, *, is_extra: bool, with_key: bool, subbatch_size: int | None) -> StackFunctions:
    #one StackFunctions per stack scope, compute dtype, kind, dropout use, sub-batch size and block config; the five multimer models share them (their parameters are inputs)
    key = (p.prefix, str(p.store.dtype), is_extra, with_key, subbatch_size, json.dumps(cfg, sort_keys=True, default=str))
    if key not in _FUNCTIONS:
        spec = discover(p, cfg, msa, pair, msa_mask, pair_mask, is_extra=is_extra, with_key=with_key, subbatch_size=subbatch_size)
        _FUNCTIONS[key] = StackFunctions(spec, p.store.dtype, cfg, is_extra, subbatch_size)
    return _FUNCTIONS[key]
