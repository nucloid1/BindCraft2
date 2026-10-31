"""The MLX AlphaFold model in its own persistent process (adapted from the RFD4 port's worker).

Why a process: earlier ports measured MLX running about a third slower in a process where torch or another framework had done CPU-side work, cured by a process boundary; and the parent here is the
JAX side (front, loss, optimizer), which must never share a process with the GPU's owner. The worker never imports jax or torch.

Protocol over the worker's stdin/stdout (stray prints go to stderr): each message is an 8-byte big-endian length and a pickle. Arrays never travel over the pipe: the parent puts them in a shared-memory
block, the worker puts its output in a block it creates and hands over (the parent copies it out and unlinks it).

    parent -> worker  ("load", {name, cfg})                                  register a model, answered with its handle (weights are read on first use)
                      ("forward", {handle, num_recycle, subbatch_size, in_shm, layout})
                      ("stats", {})                                          the memory guard's sampled peaks and how many models hold weights
                      ("quit", None)
    worker -> parent  ("ready", {pid, jax_imported, torch_imported, device}) once, then ("ok", {...}) | ("error", traceback)

A model handle is an integer the worker assigns at load; the parent embeds it in the jitted loss as the runtime argument that says which model to run. Weights stay resident for at most
MAX_LOADED models (least recently used goes first, and is read again if used again). A worker that dies (its memory guard, the OS, a crash) raises WorkerError naming the exit code and the memory cap;
the next call starts a new one and the parent re-registers the models it had loaded, in order, so handles stay valid; the call that failed is never retried and there is never a fallback to another backend.
"""
from __future__ import annotations

import argparse
import atexit
import os
import pickle
import struct
import subprocess
import sys
import threading
import traceback
from multiprocessing import resource_tracker, shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Layout = List[Tuple[str, Tuple[int, ...], str, int]]  #(name, shape, dtype, byte offset)
ALIGN = 64
DTYPES = ('bf16', 'fp32')
MAX_LOADED = 5
DEVICE_ENV = 'BINDCRAFT_MLX_DEVICE'

def send(stream, obj: Any) -> None:
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack('>Q', len(blob)))
    stream.write(blob)
    stream.flush()

def receive(stream) -> Any:
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError('peer closed the pipe')
    (size,) = struct.unpack('>Q', head)
    blob = stream.read(size)
    if len(blob) < size:
        raise EOFError('peer closed the pipe mid-message')
    return pickle.loads(blob)

def pack(arrays: Dict[str, np.ndarray]) -> Tuple[shared_memory.SharedMemory, Layout]:
    #copy the arrays into one new shared-memory block; the caller owns it (close and unlink)
    arrays = {name: np.ascontiguousarray(array) for name, array in arrays.items()}
    layout: Layout = []
    offset = 0
    for name, array in arrays.items():
        layout.append((name, tuple(array.shape), array.dtype.str, offset))
        offset += (array.nbytes + ALIGN - 1) // ALIGN * ALIGN
    block = shared_memory.SharedMemory(create=True, size=max(offset, ALIGN))
    for (name, shape, dtype, start), array in zip(layout, arrays.values()):
        np.ndarray(shape, dtype=dtype, buffer=block.buf, offset=start)[...] = array
    return block, layout

def views(block: shared_memory.SharedMemory, layout: Layout) -> Dict[str, np.ndarray]:
    return {name: np.ndarray(shape, dtype=dtype, buffer=block.buf, offset=start) for name, shape, dtype, start in layout}

def untrack(block: shared_memory.SharedMemory) -> None:
    #python registers every block with this process's resource tracker, which would unlink it when the process exits; ownership belongs to the other side
    try:
        resource_tracker.unregister(block._name, 'shared_memory')
    except Exception:
        pass

def payload_bytes(arrays: Dict[str, np.ndarray]) -> int:
    return sum(np.asarray(array).nbytes for array in arrays.values())

#----------------------------------------------------------------------------------------- worker side
def serve(params_dir: str, cap_gib: Optional[float], dtype_name: str) -> None:
    import contextlib
    import time

    import mlx.core as mx

    from bindcraft.af.mlx_af2.guard import memory_guard
    from bindcraft.af.mlx_af2.model_multimer import MultimerModel
    from bindcraft.af.mlx_af2.weights import load_params, parameter_file

    device = os.environ.get(DEVICE_ENV, 'gpu')
    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    dtype = {'bf16': mx.bfloat16, 'fp32': mx.float32}[dtype_name]
    proto_out = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)  #anything printed from here on goes to stderr
    proto_in = sys.stdin.buffer
    guard = memory_guard(max_gib=cap_gib, cache_gib=2.0) if cap_gib else contextlib.nullcontext()
    with guard as peak:
        registry: Dict[int, dict] = {}
        resident: List[int] = []  #handles holding weights, least recently used first
        send(proto_out, ('ready', {'pid': os.getpid(), 'jax_imported': 'jax' in sys.modules, 'torch_imported': 'torch' in sys.modules, 'device': device}))

        def model_for(handle: int):
            if handle not in registry:
                raise KeyError(f'model handle {handle!r} was never loaded in this worker (loaded: {sorted(registry)})')
            entry = registry[handle]
            if handle in resident:
                resident.remove(handle)
            resident.append(handle)
            while len(resident) > MAX_LOADED:
                registry[resident.pop(0)]['model'] = None  #released; read again if it is used again
                mx.clear_cache()
            if entry['model'] is None:
                entry['model'] = MultimerModel(entry['name'], load_params(entry['name'], params_dir, dtype), entry['cfg'], dtype)
            return entry['model']

        while True:
            try:
                kind, request = receive(proto_in)
            except EOFError:
                return
            if kind == 'quit':
                return
            try:
                start = time.perf_counter()
                if kind == 'load':
                    name = request['name']
                    if 'multimer' not in name:
                        raise ValueError(f'the MLX backend does not implement the model family of {name!r} yet')
                    parameter_file(name, params_dir)  #refuses a model with no weights, by name, now rather than on first use
                    handle = next((h for h, entry in registry.items() if entry['name'] == name and entry['cfg'] == request['cfg']), None)
                    if handle is None:
                        handle = len(registry)
                        registry[handle] = {'name': name, 'cfg': request['cfg'], 'model': None}
                    reply = {'handle': handle}
                elif kind == 'forward':
                    block = shared_memory.SharedMemory(name=request['in_shm'])
                    untrack(block)
                    try:
                        arrays = views(block, request['layout'])
                        features = {name[2:]: mx.array(array) for name, array in arrays.items() if name.startswith('f/')}
                        outputs, _ = model_for(request['handle']).forward(features, arrays['key'], request['num_recycle'], subbatch_size=request['subbatch_size'])
                        host = {name: np.array(value) for name, value in outputs.items()}
                        del arrays, features
                    finally:
                        block.close()
                    out_block, layout = pack(host)
                    untrack(out_block)  #the parent unlinks it
                    out_block.close()
                    reply = {'out_shm': out_block.name, 'out_layout': layout}
                elif kind == 'stats':
                    reply = {'peak': dict(peak or {}), 'loaded_models': len(resident), 'registered_models': len(registry)}
                else:
                    raise ValueError(f'unknown request {kind!r}')
                reply['seconds'] = time.perf_counter() - start
                send(proto_out, ('ok', reply))
            except Exception:
                send(proto_out, ('error', traceback.format_exc()))

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='MLX AlphaFold worker (started by bindcraft.af.mlx_af2.worker.RemoteModel)')
    parser.add_argument('--params', required=True)
    parser.add_argument('--cap-gib', type=float, default=None, help='stop this worker if its memory passes this many GiB')
    parser.add_argument('--dtype', choices=DTYPES, default='bf16')
    arguments = parser.parse_args(argv)
    serve(arguments.params, arguments.cap_gib, arguments.dtype)

#----------------------------------------------------------------------------------------- parent side
class WorkerError(RuntimeError):
    pass

class RemoteModel:
    """The multimer model in a persistent child process; numpy in, numpy out. The first call starts the worker."""
    instances: Dict[tuple, 'RemoteModel'] = {}
    instances_lock = threading.Lock()

    def __init__(self, params_dir: str, cap_gib: Optional[float] = None, dtype: str = 'bf16') -> None:
        if dtype not in DTYPES:
            raise ValueError(f'dtype must be one of {DTYPES}, got {dtype!r}')
        self.params_dir, self.cap_gib, self.dtype = params_dir, cap_gib, dtype
        self.proc: Optional[subprocess.Popen] = None
        self.ready_info: dict = {}
        self.registered: List[Tuple[str, dict]] = []  #what the worker has been told to load, in handle order, replayed after a restart
        self.last_seconds: Optional[float] = None
        self.last_shm_names: List[str] = []
        self.payload_log: List[Tuple[str, int]] = []
        self.last_error: Optional[WorkerError] = None
        self.lock = threading.RLock()

    @classmethod
    def shared(cls, params_dir: str, cap_gib: Optional[float] = None, dtype: str = 'bf16') -> 'RemoteModel':
        #one worker per process for the design and validation models alike, so two models never double the memory
        with cls.instances_lock:
            key = (params_dir, cap_gib, dtype)
            if key not in cls.instances:
                cls.instances[key] = cls(params_dir, cap_gib, dtype)
            return cls.instances[key]

    @property
    def pid(self) -> Optional[int]:
        return None if self.proc is None else self.proc.pid

    def start(self) -> None:
        with self.lock:
            if self.proc is not None:
                if self.proc.poll() is None:
                    return
                raise WorkerError(self.death('while idle'))  #said once; the next call starts a new worker
            command = [sys.executable, '-m', 'bindcraft.af.mlx_af2.worker', '--params', self.params_dir, '--dtype', self.dtype]
            if self.cap_gib:
                command += ['--cap-gib', str(self.cap_gib)]
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=dict(os.environ))
            atexit.register(self.close)
            try:
                kind, info = receive(self.proc.stdout)
            except EOFError:
                raise WorkerError(self.death('during startup'))
            if kind != 'ready':
                raise WorkerError(f'unexpected first message from the MLX worker: {kind!r}')
            self.ready_info = info
            for name, cfg in self.registered:  #a restarted worker is told again what it had loaded, so handles stay valid; the call that failed is never retried
                self.exchange_once('load', {'name': name, 'cfg': cfg}, {}, replay=True)

    def death(self, when: str) -> str:
        code = None if self.proc is None else self.proc.wait(timeout=10)
        self.proc = None
        cap = f'; its memory cap was {self.cap_gib} GiB' if self.cap_gib else ''
        return f'MLX worker died {when} (exit code {code}{cap}); see its stderr above'

    def worker_error(self, traceback_text: str) -> str:
        #the cap note rides on every reported error when a cap is set, so a memory kill is told apart by its exit code -9 (see death), never by the note alone
        cap = f'\n(worker memory cap: {self.cap_gib} GiB)' if self.cap_gib else ''
        return f'MLX worker raised:\n{traceback_text}{cap}'

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            send(proc.stdin, ('quit', None))
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    def exchange(self, kind: str, request: dict, arrays: Dict[str, np.ndarray]) -> dict:
        with self.lock:
            try:
                self.start()
                return self.exchange_once(kind, request, arrays)
            except WorkerError as error:
                self.last_error = error
                raise

    def exchange_once(self, kind: str, request: dict, arrays: Dict[str, np.ndarray], replay: bool = False) -> dict:
        block, layout = pack(arrays)
        self.last_shm_names = [block.name]
        self.payload_log.append((kind, payload_bytes(arrays)))
        try:
            try:
                send(self.proc.stdin, (kind, {**request, 'in_shm': block.name, 'layout': layout}))
                reply, info = receive(self.proc.stdout)
            except (EOFError, BrokenPipeError):
                raise WorkerError(self.death(f'during {kind}'))
            if reply == 'error':
                raise WorkerError(self.worker_error(info))
            return info
        finally:
            block.close()
            block.unlink()

    def stats(self) -> dict:
        reply = self.exchange('stats', {}, {})
        return {**reply['peak'], 'loaded_models': reply['loaded_models'], 'registered_models': reply['registered_models']}

    def load(self, model_name: str, cfg: dict) -> int:
        #register a model and return its handle; the weights are read when it is first used
        with self.lock:
            handle = self.exchange('load', {'name': model_name, 'cfg': cfg}, {})['handle']
            if (model_name, cfg) not in self.registered:
                self.registered.append((model_name, cfg))
            return handle

    def forward(self, model_handle: int, key: np.ndarray, features: Dict[str, np.ndarray], num_recycle: int, subbatch_size: Optional[int]) -> Tuple[Dict[str, np.ndarray], int]:
        arrays = {f'f/{name}': np.asarray(value) for name, value in features.items()}
        arrays['key'] = np.asarray(key, dtype=np.uint32)
        info = self.exchange('forward', {'handle': int(model_handle), 'num_recycle': int(num_recycle), 'subbatch_size': subbatch_size}, arrays)
        block = shared_memory.SharedMemory(name=info['out_shm'])
        self.last_shm_names.append(block.name)
        try:
            out_views = views(block, info['out_layout'])
            outputs = {name: array.copy() for name, array in out_views.items()}
            del out_views  #release the buffer exports, or close() raises BufferError
        finally:
            block.close()
            block.unlink()
        self.last_seconds = info['seconds']
        return outputs, 0

if __name__ == '__main__':
    main()
