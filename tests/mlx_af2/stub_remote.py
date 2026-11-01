"""A stand-in for the MLX worker: a pure-numpy "model" whose outputs are fixed linear maps of msa_feat and target_feat, with a hand-written vjp, served from a child process
so the bridge crosses a real process boundary. Which model runs shows in the output scale (1 + 0.5 * handle). numpy only: no jax in the child."""
import multiprocessing as mp
import os
import threading

import numpy as np

from bindcraft.af.mlx_af2.outputs import OUTPUT_KEYS, PAE_BINS

def stub_weights() -> dict:
    rng = np.random.default_rng(7)
    shapes = {'pt': (20, 111), 'pm': (49, 111), 'lm': (49, 50), 'lt': (20, 50), 'et': (20, 37), 'dt': (20, 64), 'dm': (49, 64), 'pt2': (20, 64), 'pm2': (49, 64)}
    return {name: (0.1 * rng.standard_normal(shape)).astype(np.float32) for name, shape in shapes.items()}

def stub_outputs(handle: int, features: dict) -> dict:
    w, scale = stub_weights(), np.float32(1.0 + 0.5 * handle)
    t, m = features['target_feat'], features['msa_feat'][0]
    length = t.shape[0]
    delta = np.float32(0.05) * (t @ w['pt'] + m @ w['pm']).reshape(length, 37, 3)
    u, v, pu, pv = t @ w['dt'], m @ w['dm'], t @ w['pt2'], m @ w['pm2']
    return {'structure_module/final_atom_positions': features['all_atom_positions'] + scale * delta, 'structure_module/final_atom_mask': features['atom37_atom_exists'].astype(np.float32),
            'predicted_lddt/logits': scale * (m @ w['lm'] + t @ w['lt']), 'predicted_aligned_error/logits': scale * (pu[:, None, :] - pv[None, :, :]),
            'predicted_aligned_error/breaks': np.linspace(0.0, 31.0, PAE_BINS - 1, dtype=np.float32), 'distogram/logits': scale * (u[:, None, :] + v[None, :, :]),
            'experimentally_resolved/logits': scale * (t @ w['et'])}

def stub_vjp(handle: int, features: dict, cotangents: dict) -> dict:
    w, scale = stub_weights(), np.float32(1.0 + 0.5 * handle)
    length = features['target_feat'].shape[0]
    c_pos = cotangents['structure_module/final_atom_positions'].reshape(length, 111)
    c_lddt, c_exp = cotangents['predicted_lddt/logits'], cotangents['experimentally_resolved/logits']
    c_dis, c_pae = cotangents['distogram/logits'], cotangents['predicted_aligned_error/logits']
    d_t = np.float32(0.05) * scale * (c_pos @ w['pt'].T) + scale * (c_lddt @ w['lt'].T) + scale * (c_exp @ w['et'].T) + scale * (c_dis.sum(1) @ w['dt'].T) + scale * (c_pae.sum(1) @ w['pt2'].T)
    d_m = np.float32(0.05) * scale * (c_pos @ w['pm'].T) + scale * (c_lddt @ w['lm'].T) + scale * (c_dis.sum(0) @ w['dm'].T) - scale * (c_pae.sum(0) @ w['pm2'].T)
    return {'target_feat': d_t.astype(np.float32), 'msa_feat': d_m[None].astype(np.float32)}

class StubError(RuntimeError):
    pass

def serve(connection) -> None:
    calls, failure, issued = [], None, 0
    while True:
        operation, payload = connection.recv()
        if operation == 'close':
            return
        if operation == 'calls':
            connection.send(('ok', list(calls)))
        elif operation == 'fail_next':
            failure = payload
            connection.send(('ok', None))
        elif operation in ('forward', 'vjp'):
            handle, key, features = payload['handle'], payload['key'], payload['features']
            calls.append({'op': operation, 'retain': payload.get('retain'), 'handle': handle, 'key': tuple(int(x) for x in np.asarray(key).ravel()), 'use_dropout': bool(features['use_dropout']), 'features': sorted(features), 'num_recycle': payload['num_recycle'], 'subbatch_size': payload['subbatch_size']})
            if failure is not None:
                message, failure = failure, None
                connection.send(('error', message))
                continue
            if operation == 'forward':
                issued += 1
                connection.send(('ok', (stub_outputs(handle, features), issued)))
            else:
                connection.send(('ok', stub_vjp(handle, features, payload['cotangents'])))

class StubRemote:
    def __init__(self):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=serve, args=(child,), daemon=True)
        self.process.start()
        child.close()
        self.pid = self.process.pid
        self.guard, self.active, self.max_active = threading.Lock(), 0, 0  #how many requests were ever in flight at once

    def ask(self, operation, payload=None):
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.connection.send((operation, payload))
            status, value = self.connection.recv()
        finally:
            with self.guard:
                self.active -= 1
        if status == 'error':
            raise StubError(value)
        return value

    def forward(self, model_handle, key, features, num_recycle, subbatch_size, retain=False):
        return self.ask('forward', {'handle': int(model_handle), 'key': key, 'features': features, 'num_recycle': num_recycle, 'subbatch_size': subbatch_size, 'retain': retain})

    def vjp(self, model_handle, call_id, key, features, num_recycle, subbatch_size, cotangents, diff_keys):
        gradients = self.ask('vjp', {'handle': int(model_handle), 'key': key, 'features': features, 'num_recycle': num_recycle, 'subbatch_size': subbatch_size, 'cotangents': cotangents})
        return {name: gradients[name] for name in diff_keys}

    def calls(self):
        return self.ask('calls')

    def fail_next(self, message: str) -> None:
        self.ask('fail_next', message)

    def close(self) -> None:
        if self.process.is_alive():
            self.connection.send(('close', None))
            self.process.join(timeout=10)
