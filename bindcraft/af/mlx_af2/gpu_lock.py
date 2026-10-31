#One GPU run at a time. hold() takes an exclusive non-blocking flock on a lock file (default ~/.cache/esmfold2_mlx/gpu.lock, override with ESMFOLD2_MLX_LOCK), writes the
#holder's pid into it and exports ESMFOLD2_MLX_LOCK_OWNER=<pid>. A second run fails at once, naming the holder; the OS drops the flock when the holder dies, however it dies.
#The names are kept from ESMFold2-MLX on purpose: every MLX project of Ethan's shares one lock file, so a run in any of them excludes a run in the others.
#A process whose environment names a live proper ancestor as the owner runs under that ancestor's lock (workers and scripts a run launches): for it hold() does nothing.
#A pid that is not an ancestor, forged or stale, grants nothing. Inside one process hold() is re-entrant: the campaign builds a design and a validation model that share it.
import contextlib
import fcntl
import os
import subprocess

PATH_ENV = 'ESMFOLD2_MLX_LOCK'
OWNER_ENV = 'ESMFOLD2_MLX_LOCK_OWNER'
DEFAULT_PATH = '~/.cache/esmfold2_mlx/gpu.lock'
held_depth = 0

def lock_path() -> str:
    return os.path.expanduser(os.environ.get(PATH_ENV, DEFAULT_PATH))

def parent_of(pid: int) -> int | None:
    try:
        output = subprocess.run(['ps', '-o', 'ppid=', '-p', str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return int(output) if output else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None

def is_proper_ancestor(pid: int) -> bool:
    if pid == os.getpid():
        return False
    current, seen = os.getppid(), set()
    while current and current > 1 and current not in seen:
        if current == pid:
            return True
        seen.add(current)
        current = parent_of(current)
    return False

@contextlib.contextmanager
def hold():
    global held_depth
    if held_depth:
        held_depth += 1
        try:
            yield
        finally:
            held_depth -= 1
        return
    owner = os.environ.get(OWNER_ENV, '')
    if owner.isdigit() and is_proper_ancestor(int(owner)):
        yield
        return
    path = lock_path()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = os.pread(descriptor, 32, 0).decode(errors='replace').strip() or 'unknown'
            raise SystemExit(f'another GPU run holds {path} (pid {holder}); only one GPU run at a time')
        os.ftruncate(descriptor, 0)
        os.pwrite(descriptor, f'{os.getpid()}\n'.encode(), 0)
        previous = os.environ.get(OWNER_ENV)
        os.environ[OWNER_ENV] = str(os.getpid())
        held_depth = 1
        try:
            yield
        finally:
            held_depth = 0
            if previous is None:
                os.environ.pop(OWNER_ENV, None)
            else:
                os.environ[OWNER_ENV] = previous
    finally:
        os.close(descriptor)
