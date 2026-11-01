"""End to end (reported, never a stop): real campaigns on examples/pdl1_denovo.json with the JAX path (as shipped: metaljax on the Apple GPU) and the MLX backend, over several seeds, and the parent's device.
A campaign here is one trajectory with a short stage budget; each campaign is a subprocess of `bindcraft design`, run under the shared GPU lock this script holds for its whole life (the children run under it).
Modes:
  --compare          trajectory-only campaigns, --seeds each for the backends named, then a distribution table of the final iptm, pLDDT, weighted loss and the design time
  --parent           the whole pipeline (ProteinMPNN, validation models, filters, lenient stage thresholds so a short trajectory reaches them) with the parent's JAX on cpu and on metal, MLX backend
  --smoke            one tiny campaign, to see that the pipeline runs
Usage: python scripts/mlx_af2/e2e_compare.py --compare --seeds 5 [--backends jax mlx] [--length 80] [--out DIR] [--budget screen,refine,anneal,harden]
Writes <out>/results.json and <out>/report.md; the documented run is in docs/mlx_backend/e2e.md."""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
SETTINGS_FILE = REPO / 'examples' / 'pdl1_denovo.json'
STAGES = ('screen', 'refine', 'anneal', 'harden')
DEFAULT_BUDGET = {'screen_steps': 15, 'refine_steps': 8, 'anneal_steps': 12, 'harden_steps': 3, 'mutate_steps': 2}
SMOKE_BUDGET = {'screen_steps': 3, 'refine_steps': 2, 'anneal_steps': 3, 'harden_steps': 2, 'mutate_steps': 2}

def campaign_command(folder: Path, seed: int, *, trajectory_only: bool = True, budget: dict | None = None, length: int = 80, lenient: bool = False, extra: dict | None = None) -> list[str]:
    settings = {'project_folder': str(folder), 'campaign_seed': seed, 'max_trajectories': 1, 'binder_lengths': f'[{length},{length}]', 'number_of_final_designs': 1,
                'trajectory_only': 'true' if trajectory_only else 'false', **(DEFAULT_BUDGET if budget is None else budget)}
    if lenient:  #stage thresholds at zero, so a short trajectory is carried through every stage to ProteinMPNN, the validation models and the filters
        settings.update({f'min_{metric}_{stage}': 0.0 for stage in STAGES + ('mutate', 'final') for metric in ('plddt', 'iptm')})
        settings.update({'min_ptm_final': 0.0, 'min_monomer_plddt_final': 0.0, 'max_ipae_final': 1.0})
    settings.update(extra or {})
    command = [sys.executable, '-m', 'bindcraft.cli', 'design', str(SETTINGS_FILE)]
    for key, value in settings.items():
        command += ['--set', f'{key}={value}']
    return command

def run_campaign(folder: Path, seed: int, backend: str, *, parent_device: str | None = None, timeout: float = 7200, **options) -> dict:
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    environment.pop('BINDCRAFT_BACKEND', None)
    if backend == 'mlx':
        environment['BINDCRAFT_BACKEND'] = 'mlx'
    if parent_device:
        environment['JAX_PLATFORMS'] = parent_device
    started = time.time()
    process = subprocess.run(campaign_command(folder, seed, **options), capture_output=True, text=True, env=environment, timeout=timeout, cwd=REPO)
    output = process.stdout + process.stderr
    return {'backend': backend, 'seed': seed, 'parent_device': parent_device, 'returncode': process.returncode, 'wall_seconds': round(time.time() - started, 1), 'output_tail': output[-3000:], 'output': output, **parse_campaign(folder, output)}

def parse_campaign(folder: Path, output: str) -> dict:
    parsed: dict = {}
    trajectories = folder / '1_Trajectories' / '!_Trajectories.csv'
    if trajectories.exists():
        rows = list(csv.DictReader(open(trajectories)))
        timing = dict(part.split('=', 1) for part in rows[-1]['Timing'].split(';') if '=' in part) if rows else {}
        parsed['timing'] = timing
        parsed['design_seconds'] = float(timing['design']) if 'design' in timing else None
        parsed['terminated'] = rows[-1].get('terminated') if rows else None
    weights = {name: float(value) for name, value in re.findall(r'([a-z_]+)=(-?[0-9.]+)', ' '.join(line for line in output.splitlines() if line.startswith('losses ')))}
    for path in sorted((folder / '1_Trajectories').glob('design_*/*_losses.csv')):
        rows = list(csv.DictReader(open(path)))
        with_losses = [row for row in rows if row['phase'] != 'mutate' and any(value not in ('', None) for key, value in row.items() if '.' in key)]  #the gradient rounds: the mutate rounds that end a trajectory carry no gradient losses
        if with_losses:
            last = with_losses[-1]
            parsed['final_round'] = {'phase': last['phase'], 'round': int(last['round'])}
            parsed['rounds'] = len(with_losses)  #gradient rounds run: trajectories stop at different stages, so total design time mixes different amounts of work
            if parsed.get('design_seconds'):
                parsed['seconds_per_round'] = round(parsed['design_seconds'] / len(with_losses), 2)
            for name, suffix in (('final_iptm', '.iptm'), ('final_ptm', '.ptm')):
                key = next((key for key in last if key.endswith(suffix) and last[key] not in ('', None)), None)
                parsed[name] = float(last[key]) if key else None
            terms = {key.split('.', 1)[1]: float(value) for key, value in last.items() if '.' in key and value not in ('', None)}
            parsed['weighted_loss'] = round(sum(weights.get(name, 0.0) * value for name, value in terms.items()), 4)
    matches = re.findall(r'i_pTM=([0-9.]+)\s+pLDDT=([0-9.]+)', output)
    if matches:
        parsed['reported_iptm'], parsed['reported_plddt'] = float(matches[-1][0]), float(matches[-1][1])
    return parsed

METRICS = (('seconds_per_round', 'design s per gradient round'), ('rounds', 'gradient rounds'), ('design_seconds', 'design time (s)'), ('final_iptm', 'final-round ipTM'), ('reported_plddt', 'reported pLDDT'), ('weighted_loss', 'weighted loss'))

def distribution(values: list[float]) -> dict:
    values = [value for value in values if value is not None]
    if not values:
        return {'n': 0}
    return {'n': len(values), 'mean': statistics.fmean(values), 'std': statistics.stdev(values) if len(values) > 1 else 0.0, 'min': min(values), 'max': max(values)}

def overlap_statement(a: dict, b: dict) -> str:
    if not a.get('n') or not b.get('n'):
        return 'no data'
    low, high = max(a['min'], b['min']), min(a['max'], b['max'])
    if low > high:
        return f'ranges DISJOINT ({a["min"]:.3g}..{a["max"]:.3g} against {b["min"]:.3g}..{b["max"]:.3g})'
    union = max(a['max'], b['max']) - min(a['min'], b['min'])
    share = (high - low) / union if union else 1.0
    return f'ranges overlap on {100 * share:.0f}% of their union ({low:.3g}..{high:.3g})'

def report(results: list[dict], title: str) -> str:
    lines = [f'# {title}', '', f'{len(results)} campaigns. Distributions over seeds, one trajectory each.', '']
    groups: dict = {}
    for result in results:
        groups.setdefault(f'{result["backend"]}' + (f' (parent {result["parent_device"]})' if result.get('parent_device') else ''), []).append(result)
    lines += ['| group | n | exit codes | ' + ' | '.join(label for _, label in METRICS) + ' |', '|' + '---|' * (3 + len(METRICS))]
    summaries = {}
    for name, group in groups.items():
        summaries[name] = {key: distribution([result.get(key) for result in group]) for key, _ in METRICS}
        cells = [f'{summaries[name][key]["mean"]:.3g} ± {summaries[name][key]["std"]:.2g} ({summaries[name][key]["min"]:.3g}..{summaries[name][key]["max"]:.3g})' if summaries[name][key]['n'] else 'n/a' for key, _ in METRICS]
        lines.append(f'| {name} | {len(group)} | {sorted({result["returncode"] for result in group})} | ' + ' | '.join(cells) + ' |')
    names = list(groups)
    if len(names) == 2:
        lines += ['', f'Overlap, {names[0]} against {names[1]}:'] + [f'- {label}: {overlap_statement(summaries[names[0]][key], summaries[names[1]][key])}' for key, label in METRICS]
    return '\n'.join(lines) + '\n'

def main(argv=None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--parent', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--backends', nargs='+', default=['jax', 'mlx'], choices=('jax', 'mlx'))
    parser.add_argument('--length', type=int, default=80)
    parser.add_argument('--out', default=str(REPO / 'results' / 'e2e'))
    parser.add_argument('--parent-devices', nargs='+', default=['cpu', 'metal'])
    arguments = parser.parse_args(argv)
    from bindcraft.af.mlx_af2 import gpu_lock, preflight
    preflight.check_disk()
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with gpu_lock.hold():
        if arguments.smoke:
            plan = [('mlx', 1, {'trajectory_only': True, 'budget': SMOKE_BUDGET, 'length': 60}, 'cpu')]
        elif arguments.parent:
            plan = [('mlx', seed, {'trajectory_only': False, 'lenient': True, 'budget': SMOKE_BUDGET, 'length': arguments.length}, device) for seed in range(1, arguments.seeds + 1) for device in arguments.parent_devices]
        else:
            plan = [(backend, seed, {'length': arguments.length}, 'cpu' if backend == 'mlx' else None) for seed in range(1, arguments.seeds + 1) for backend in arguments.backends]
        for backend, seed, options, device in plan:
            folder = out / f'{backend}{"_" + device if device and backend == "mlx" else ""}_seed{seed}'
            result = run_campaign(folder, seed, backend, parent_device=device, **options)
            results.append(result)
            print(json.dumps({key: value for key, value in result.items() if key not in ('output_tail', 'output')}), flush=True)
            (out / 'results.json').write_text(json.dumps([{key: value for key, value in kept.items() if key != 'output'} for kept in results], indent=1))
    (out / 'report.md').write_text(report(results, 'End-to-end comparison'))
    return results

if __name__ == '__main__':
    main()
