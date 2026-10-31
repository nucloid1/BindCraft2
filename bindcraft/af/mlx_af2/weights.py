#AlphaFold parameters for the MLX model, read straight from the .npz files the JAX path reads, with the fusing of the triangle projections that utils.flat_params_to_haiku(fuse=True) does.
#Tensors stay float32 as stored and are cast to the compute dtype when read, as the haiku bfloat16_getter does. Every tensor read is recorded, so a port can prove it used them all.
import os

import mlx.core as mx
import numpy as np

PREFIX = 'alphafold/alphafold_iteration/'
#the four scopes whose triangle multiplications the JAX loader fuses (they are layer stacks: their tensors carry a leading axis over the layers)
FUSED_SCOPES = ('evoformer_iteration', 'extra_msa_stack', 'template_embedding/single_template_embedding/template_embedding_iteration', 'template_embedding/single_template_embedding/template_pair_stack/__layer_stack_no_state')
STACK_MARKERS = ('evoformer_iteration', 'extra_msa_stack', 'template_embedding_iteration', '__layer_stack_no_state')
TRIANGLE_OPERATIONS = ('triangle_multiplication_incoming', 'triangle_multiplication_outgoing')

def parameter_file(model_name: str, params_dir: str) -> str:
    for candidate in (('params', f'params_{model_name}.npz'), (f'params_{model_name}.npz',), ('params', f'{model_name}.npz'), (f'{model_name}.npz',)):
        path = os.path.join(params_dir, *candidate)
        if os.path.isfile(path):
            return path
    raise ValueError(f'no AlphaFold parameters for {model_name!r} in {params_dir!r}; run "bindcraft fetch-weights" to download them')

def fused_tensors(raw: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    #name -> array, name -> the raw keys it came from; names are the haiku scope and parameter joined with '/', without the model prefix
    tensors = {key.replace('//', '/').removeprefix(PREFIX): array for key, array in raw.items()}
    sources = {key.replace('//', '/').removeprefix(PREFIX): [key] for key in raw}
    for scope in FUSED_SCOPES:
        for operation in TRIANGLE_OPERATIONS:
            base = f'evoformer/{scope}/{operation}'
            if f'{base}/center_layer_norm/scale' not in tensors:
                continue
            for kind in ('gate', 'projection'):
                for part in ('bias', 'weights'):
                    left, right = f'{base}/left_{kind}/{part}', f'{base}/right_{kind}/{part}'
                    tensors[f'{base}/{kind}/{part}'] = np.concatenate([tensors.pop(left), tensors.pop(right)], -1)
                    sources[f'{base}/{kind}/{part}'] = sources.pop(left) + sources.pop(right)
            for old, new in (('center_layer_norm', 'center_norm'), ('layer_norm_input', 'left_norm_input')):
                for part in ('scale', 'offset'):
                    tensors[f'{base}/{new}/{part}'] = tensors.pop(f'{base}/{old}/{part}')
                    sources[f'{base}/{new}/{part}'] = sources.pop(f'{base}/{old}/{part}')
    return tensors, sources

class Store:
    def __init__(self, tensors: dict[str, np.ndarray], sources: dict[str, list[str]], dtype, model_name: str):
        self.tensors, self.sources, self.dtype, self.model_name = tensors, sources, dtype, model_name
        self.used: set[str] = set()
        self.cache: dict[tuple[str, mx.Dtype], mx.array] = {}

class Params:
    def __init__(self, store: Store, prefix: str = '', layer_index: int | None = None):
        self.store, self.prefix, self.layer_index = store, prefix, layer_index

    def scope(self, prefix: str) -> 'Params':
        return Params(self.store, f'{self.prefix}{prefix.strip("/")}/', self.layer_index)

    def layer(self, index: int) -> 'Params':
        #the stacked tensors (the Evoformer, the extra-MSA stack, the template stack) are read at one layer; others are returned whole
        return Params(self.store, self.prefix, index)

    def get(self, path: str, dtype: mx.Dtype | None = None) -> mx.array:
        #cast to dtype (default: the load dtype) when read; a layer that keeps its parameters in float32, as the repo's LayerNorm does, asks for mx.float32
        name, dtype = f'{self.prefix}{path}', dtype or self.store.dtype
        if name not in self.store.tensors:
            raise KeyError(f'no AlphaFold parameter {name!r} in {self.store.model_name}')
        self.store.used.add(name)
        if (name, dtype) not in self.store.cache:
            array = mx.array(self.store.tensors[name])
            self.store.cache[(name, dtype)] = array.astype(dtype) if mx.issubdtype(array.dtype, mx.floating) else array
        value = self.store.cache[(name, dtype)]
        return value[self.layer_index] if self.layer_index is not None and any(marker in name for marker in STACK_MARKERS) else value

    def has(self, path: str) -> bool:
        return f'{self.prefix}{path}' in self.store.tensors

    def stored(self, name: str) -> np.ndarray:
        return self.store.tensors[name]

    def names(self) -> list[str]:
        return sorted(self.store.tensors)

    def sources(self, name: str) -> list[str]:
        return list(self.store.sources[name])

    def unused(self) -> list[str]:
        return sorted(set(self.store.tensors) - self.store.used)

    def assert_all_used(self) -> None:
        unused = self.unused()
        assert not unused, f'{len(unused)} unused AlphaFold parameters in {self.store.model_name}: {unused[:5]}'

def load_params(model_name: str, params_dir: str, dtype: mx.Dtype = mx.bfloat16) -> Params:
    data = np.load(parameter_file(model_name, params_dir), allow_pickle=False)
    tensors, sources = fused_tensors({key: data[key] for key in data.files})
    return Params(Store(tensors, sources, dtype, model_name))
