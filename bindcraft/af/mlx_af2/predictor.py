"""The repo's AlphaFold design model with the AlphaFold call served by the MLX worker (BINDCRAFT_BACKEND=mlx).

Everything else is the base class: the front, the loss, the optimizer, the compile caches, the memory ladder's shape logic and the validation path run in JAX on the CPU. Only the one call that runs
AlphaFold inside the jitted functions changes (_model_outputs, through the custom_vjp bridge), the loading of the parameters (the worker reads the weights; this side keeps an integer handle, which is
the runtime argument that says which model runs, because the compile caches are keyed by model family), and the runner (a descriptor: there is no haiku model here).

A design model and a validation model in one process share one worker (RemoteModel.shared) and one GPU lock; the lock is taken, and the disk preflight run, once per process tree, when the first model
loads. The worker runs on the GPU unless BINDCRAFT_MLX_DEVICE=cpu, and only a GPU worker needs the lock."""
from __future__ import annotations

import atexit
import contextlib
import copy
import os
import subprocess
from types import SimpleNamespace

import jax.numpy as jnp

from bindcraft.af.alphafold.model import config as af_config
from bindcraft.af.mlx_af2 import bridge, gpu_lock, preflight
from bindcraft.af.mlx_af2.worker import DEVICE_ENV, RemoteModel
from bindcraft.af2 import AlphaFoldDesignModel

CAP_ENV = 'BINDCRAFT_MLX_CAP_GIB'
CAP_SHARE_OF_MEMORY = 0.75  #36 GiB of a 48 GB machine: unified memory is shared with the OS and the GPU's own cap sits below the total
session: contextlib.ExitStack | None = None

def default_cap_gib() -> float:
    if CAP_ENV in os.environ:
        return float(os.environ[CAP_ENV])
    memory = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, timeout=10).stdout)
    return CAP_SHARE_OF_MEMORY * memory / 2 ** 30

def start_session() -> None:
    #once per process: the disk and process checks every real-model run starts with, then the shared GPU lock until the process exits (hold() is re-entrant, and a descendant of the holder is let through)
    global session
    if session is not None:
        return
    preflight.checks()
    stack = contextlib.ExitStack()
    if os.environ.get(DEVICE_ENV, 'gpu') != 'cpu':
        stack.enter_context(gpu_lock.hold())
    atexit.register(stack.close)
    session = stack

class MlxAlphaFoldDesignModel(AlphaFoldDesignModel):
    dtype = 'bf16'  #bfloat16 activations like the shipped model; a subclass may ask for 'fp32'
    remote: RemoteModel

    def model_config(self, model_name: str) -> dict:
        #the effective config the JAX runner would be built from (AlphaFoldDesignModel._alphafold_runner), as the plain dict the worker takes
        config = copy.deepcopy(af_config.model_config(model_name))
        settings = config.model.global_config
        settings.use_dgram, settings.use_remat, settings.bfloat16 = False, self.use_remat, self.dtype == 'bf16'
        settings.subbatch_size, settings.attention_backend, settings.use_cueq = None, 'stock', False
        config.model.num_recycle = 0
        return config.model.to_dict()

    def _load_parameters(self, model_name: str, data_dir: str | None) -> object:
        if 'multimer' not in model_name:
            raise ValueError(f'the MLX backend implements the AlphaFold-Multimer models only so far, not {model_name!r}; use a multimer model or leave BINDCRAFT_BACKEND unset (jax)')
        if not data_dir:
            raise ValueError(f'no AlphaFold parameters for {model_name!r} in data_dir {data_dir!r}; run "bindcraft fetch-weights" to download them')
        start_session()
        self.remote = RemoteModel.shared(data_dir, default_cap_gib(), self.dtype)
        return jnp.asarray(self.remote.load(model_name, self.model_config(model_name)), jnp.int32)

    def _alphafold_runner(self, model_family: tuple, subbatch_size: int | None, attention_backend: str | None = None):
        return SimpleNamespace(subbatch_size=subbatch_size, attention_backend=attention_backend or self.attention_backend)

    def subbatch_ladder(self) -> tuple:
        #the repo's ladder retries a failed gradient with smaller shards and then the chunked attention; the worker's memory failure is final (its cap is the budget), so one rung
        return (None,) if self.subbatch_size == 'auto' else (self.subbatch_size,)

    def _model_outputs(self, alphafold_runner, model_parameters, key, model_inputs: dict) -> dict:
        return bridge.mlx_model_outputs(self.remote, model_parameters, key, model_inputs, self.num_recycle, alphafold_runner.subbatch_size)
