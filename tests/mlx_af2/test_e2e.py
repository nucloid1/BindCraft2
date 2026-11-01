"""Real campaigns through `bindcraft design` with the MLX backend (and the JAX default, for the control), as subprocesses of the whole pipeline. Marked gpu: a campaign on the CPU device is far too slow for the
ordinary run, so these run under the shared GPU lock (BINDCRAFT_MLX_INTEGRATION=1 pytest -m gpu), which the campaigns they start inherit. Budgets are tiny: what is checked is that the pipeline runs and fails
loudly, not the quality of a design."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import e2e_compare as E

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

def descendants(root: int) -> list[tuple[int, str]]:
    table = {}
    for line in subprocess.run(['ps', '-axo', 'pid=,ppid=,command='], capture_output=True, text=True).stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2])
    found = []
    for pid, (parent, command) in table.items():
        ancestor = parent
        while ancestor in table and ancestor != root:
            ancestor = table[ancestor][0]
        if ancestor == root:
            found.append((pid, command))
    return found

def test_trajectory_only_run_exits_cleanly_with_mlx_and_multimer_validation(tmp_path):
    #validation_model multimer needs multimer models left over, so the design pool is cut to three of the five
    result = E.run_campaign(tmp_path / 'campaign', 1, 'mlx', parent_device='cpu', trajectory_only=True, budget=E.SMOKE_BUDGET, length=60, extra={'validation_model': 'multimer', 'design_models': 3})
    assert result['returncode'] == 0, result['output_tail']
    assert 'trajectory-only' in result['output_tail'] and 'campaign done' in result['output_tail']
    assert result.get('design_seconds')

def test_default_run_completes_with_monomer_validation(tmp_path):
    #the whole pipeline with the default validation pool (the monomer models): design stages, ProteinMPNN, validation, filters; thresholds at zero so a tiny trajectory gets that far
    result = E.run_campaign(tmp_path / 'campaign', 1, 'mlx', parent_device='cpu', trajectory_only=False, lenient=True, budget=E.SMOKE_BUDGET, length=60)
    assert result['returncode'] == 0, result['output_tail']
    assert 'validation model_1_ptm,model_2_ptm' in result['output']  #the default validation pool, the monomer models
    assert 'redesigns passed' in result['output'] and 'campaign done' in result['output']  #ProteinMPNN's redesigns were validated and filtered

def test_no_silent_fallback(tmp_path):
    #the MLX worker is killed in the middle of a trajectory: the campaign says why and does not carry on as if nothing happened (no JAX fallback, no clean exit)
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'BINDCRAFT_BACKEND': 'mlx', 'JAX_PLATFORMS': 'cpu'}
    process = subprocess.Popen(E.campaign_command(tmp_path / 'campaign', 1, trajectory_only=True, budget={**E.DEFAULT_BUDGET, 'screen_steps': 40}, length=80), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               env=environment, cwd=E.REPO)
    killed = None
    deadline = time.time() + 600
    while time.time() < deadline and process.poll() is None and killed is None:
        workers = [pid for pid, command in descendants(process.pid) if 'bindcraft.af.mlx_af2.worker' in command]
        if workers:
            time.sleep(25)  #well into the first trajectory: weights read, graphs compiled, gradient steps running
            os.kill(workers[0], signal.SIGKILL)
            killed = workers[0]
        time.sleep(1)
    output, _ = process.communicate(timeout=900)
    assert killed is not None, f'no MLX worker was ever started:\n{output[-2000:]}'
    assert process.returncode != 0, 'the campaign exited cleanly after its worker was killed'
    assert 'exit code -9' in output and 'MLX worker' in output, output[-3000:]
    assert 'campaign done' not in output

def test_jax_default_path_unaffected(tmp_path):
    #without BINDCRAFT_BACKEND the campaign runs the JAX path as before: it completes, starts no MLX worker and prints nothing of the MLX session
    result = E.run_campaign(tmp_path / 'campaign', 1, 'jax', trajectory_only=True, budget=E.SMOKE_BUDGET, length=60)
    assert result['returncode'] == 0, result['output_tail']
    assert 'MLX' not in result['output_tail'] and 'preflight WARNING' not in result['output_tail']
    from bindcraft.af2 import AlphaFoldDesignModel
    from bindcraft.backend import design_model_class
    assert 'BINDCRAFT_BACKEND' not in os.environ and design_model_class() is AlphaFoldDesignModel
