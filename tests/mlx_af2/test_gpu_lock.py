"""gpu_lock.hold: one GPU run at a time, fail fast naming the holder, re-entrant inside one process, descendants run under the parent's lock.
Every contender is started with ESMFOLD2_MLX_LOCK_OWNER removed (a process that inherits it from a live ancestor is a legitimate descendant, not a contender) and
ESMFOLD2_MLX_LOCK pointed at a private lock file, so these tests never touch the session lock."""
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bindcraft.af.mlx_af2.gpu_lock import hold

REPO = Path(__file__).resolve().parents[2]
OWNER = 'ESMFOLD2_MLX_LOCK_OWNER'
TRY = "from bindcraft.af.mlx_af2.gpu_lock import hold\nwith hold():\n    print('acquired', flush=True)\n"
HOLD = "import time\nfrom bindcraft.af.mlx_af2.gpu_lock import hold\nwith hold():\n    print('ready', flush=True)\n    time.sleep(120)\n"

def child_env(lock_path, owner=None, keep_owner=False):
    env = dict(os.environ)
    env['ESMFOLD2_MLX_LOCK'] = str(lock_path)
    if not keep_owner:
        env.pop(OWNER, None)
    if owner is not None:
        env[OWNER] = str(owner)
    return env

def run_script(script, env, timeout=30):
    return subprocess.run([sys.executable, '-I', '-B', '-c', script], env=env, cwd=REPO, capture_output=True, text=True, timeout=timeout)

@pytest.fixture
def lock_path(tmp_path, monkeypatch):
    path = tmp_path / 'gpu.lock'
    monkeypatch.setenv('ESMFOLD2_MLX_LOCK', str(path))
    monkeypatch.delenv(OWNER, raising=False)
    return path

def start_holder(lock_path):
    process = subprocess.Popen([sys.executable, '-I', '-B', '-c', HOLD], env=child_env(lock_path), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ready, _, _ = select.select([process.stdout], [], [], 30)
    line = process.stdout.readline() if ready else ''
    if line.strip() != 'ready':
        process.kill()
        process.wait()
        pytest.fail(f'holder never reported ready: {line!r} {process.stderr.read()!r}')
    return process

def stop(process):
    if process.poll() is None:
        process.kill()
        process.wait()
    process.stdout.close()
    process.stderr.close()

def test_second_holder_fails_fast_naming_the_pid(lock_path):
    with hold():
        start = time.monotonic()
        result = run_script(TRY, child_env(lock_path))
        elapsed = time.monotonic() - start
    assert result.returncode != 0
    assert elapsed < 15
    assert str(os.getpid()) in result.stderr
    assert 'acquired' not in result.stdout

def test_second_hold_in_the_same_process_passes(lock_path):
    with hold():
        with hold():
            assert os.environ[OWNER] == str(os.getpid())
        assert os.environ[OWNER] == str(os.getpid())  #the inner exit must not release the outer hold
        assert run_script(TRY, child_env(lock_path)).returncode != 0
    assert run_script(TRY, child_env(lock_path)).returncode == 0

def test_lock_is_released_when_the_holder_is_killed(lock_path):
    holder = start_holder(lock_path)
    try:
        blocked = run_script(TRY, child_env(lock_path))
        assert blocked.returncode != 0 and str(holder.pid) in blocked.stderr
        holder.kill()  #SIGKILL runs no cleanup, the OS must release the flock
        holder.wait(timeout=10)
        result = run_script(TRY, child_env(lock_path))
        assert result.returncode == 0, result.stderr
    finally:
        stop(holder)

def test_descendant_inherits_the_lock_but_a_stranger_does_not(lock_path):
    with hold():
        child = run_script(TRY, child_env(lock_path, keep_owner=True))
        assert child.returncode == 0, child.stderr
        grandchild_script = f"import subprocess, sys\nr = subprocess.run([sys.executable, '-I', '-B', '-c', {TRY!r}], capture_output=True, text=True)\nsys.stdout.write(r.stdout); sys.stderr.write(r.stderr); sys.exit(r.returncode)\n"
        grandchild = run_script(grandchild_script, child_env(lock_path, keep_owner=True))
        assert grandchild.returncode == 0, grandchild.stderr
        stranger = run_script(TRY, child_env(lock_path))
        assert stranger.returncode != 0 and str(os.getpid()) in stranger.stderr

def test_an_owner_naming_a_non_ancestor_grants_nothing(lock_path):
    holder = start_holder(lock_path)
    try:
        forged = run_script(TRY, child_env(lock_path, owner=holder.pid))
        assert forged.returncode != 0 and str(holder.pid) in forged.stderr
        stale = run_script(TRY, child_env(lock_path, owner=2 ** 22 + 12345))
        assert stale.returncode != 0 and str(holder.pid) in stale.stderr
    finally:
        stop(holder)

def test_normal_exit_releases_the_lock_and_restores_the_owner_variable(lock_path):
    before = os.environ.get(OWNER)
    with hold():
        assert os.environ[OWNER] == str(os.getpid())
    assert os.environ.get(OWNER) == before
    assert run_script(TRY, child_env(lock_path)).returncode == 0

def test_lock_file_is_created_under_a_missing_directory(tmp_path, monkeypatch):
    path = tmp_path / 'deep' / 'er' / 'gpu.lock'
    monkeypatch.setenv('ESMFOLD2_MLX_LOCK', str(path))
    monkeypatch.delenv(OWNER, raising=False)
    with hold():
        assert path.read_text().strip() == str(os.getpid())
