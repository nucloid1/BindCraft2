import subprocess
import types

import pytest

from bindcraft.af.mlx_af2 import preflight

def usage(used_fraction):
    total = 1000 * 10 ** 9
    return types.SimpleNamespace(total=total, used=int(total * used_fraction), free=int(total * (1 - used_fraction)))

def test_a_disk_under_the_limit_passes(monkeypatch):
    monkeypatch.setattr(preflight.shutil, 'disk_usage', lambda path: usage(0.5))
    preflight.check_disk()

def test_a_full_disk_is_refused_by_default(monkeypatch):
    monkeypatch.setattr(preflight.shutil, 'disk_usage', lambda path: usage(0.95))
    monkeypatch.delenv(preflight.MIN_FREE_ENV, raising=False)
    with pytest.raises(SystemExit, match='preflight: disk is 95% full'):
        preflight.check_disk()

def test_the_opt_in_floor_allows_a_full_disk_with_enough_free_and_says_so(monkeypatch, capsys):
    monkeypatch.setattr(preflight.shutil, 'disk_usage', lambda path: usage(0.95))
    monkeypatch.setenv(preflight.MIN_FREE_ENV, '30')
    preflight.check_disk()
    assert 'WARNING' in capsys.readouterr().err
    monkeypatch.setenv(preflight.MIN_FREE_ENV, '80')
    with pytest.raises(SystemExit, match='below'):
        preflight.check_disk()

def test_other_processes_memory_is_not_checked(monkeypatch):
    #only the disk is checked: a browser or another job holding gigabytes does not stop a run (Ethan dropped that rule), so nothing asks ps about other processes
    def forbidden(*arguments, **keywords):
        raise AssertionError('preflight asked the system about other processes')
    monkeypatch.setattr(preflight.shutil, 'disk_usage', lambda path: usage(0.5))
    monkeypatch.setattr(subprocess, 'run', forbidden)
    preflight.checks()
    assert not hasattr(preflight, 'heavy_processes')
