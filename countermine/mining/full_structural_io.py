"""Shared CPU-only I/O and immutable evidence binding for Step 2D."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

from .structural_analysis import _atomic_bytes, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'cache/countermine_rgb/step2d'


def read_json(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    value = json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError('Step 2D metadata JSON must contain an object')
    return value


def relative_path(path, repo_root=ROOT):
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError as error:
        raise ValueError('artifact/source must be inside the repository') from error


def guard_step2d(paths, repo_root=ROOT):
    root = Path(repo_root).resolve()
    def allowed(path):
        return (path.is_relative_to(root / 'cache/countermine_rgb/step2d')
                or path.is_relative_to(root / 'outputs/step2d')
                or (path.parent == root / 'docs/audits' and path.name.startswith('step2d_')))
    for value in paths:
        path = Path(value)
        if path.is_symlink() or not allowed(path.absolute()) or not allowed(path.resolve()):
            raise ValueError('Step 2D writes must stay in step2d cache/output or step2d_* audit files')


def validate_portable_json(value):
    def walk(item):
        if isinstance(item, dict):
            for key, child in item.items():
                walk(key)
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
        elif isinstance(item, str):
            if item.startswith(('/', '\\')) or (len(item) > 2 and item[1:3] in (':/', ':\\')):
                raise ValueError('Step 2D JSON must not contain absolute paths')
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Step 2D JSON must be finite')
    walk(value)
    json.dumps(value, allow_nan=False)


def atomic_bytes(path, payload, *, repo_root=ROOT):
    guard_step2d([path], repo_root)
    _atomic_bytes(Path(path), payload)


def write_json(path, value, *, repo_root=ROOT):
    validate_portable_json(value)
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode(), repo_root=repo_root)


def write_csv(path, frame, *, repo_root=ROOT):
    # Match Step 2A's shortest float serialization; never round ECDF raw inputs.
    atomic_bytes(path, frame.to_csv(index=False, lineterminator='\n').encode(), repo_root=repo_root)


def code_hashes(paths, repo_root=ROOT):
    root = Path(repo_root)
    return {relative_path(root / name, root): sha256_file(root / name) for name in paths}


def git_commit(repo_root=ROOT):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sequence_hash(pair_uids):
    return hashlib.sha256((json.dumps(list(pair_uids), separators=(',', ':'), ensure_ascii=False) + '\n').encode()).hexdigest()


def verify_hashes(hashes, repo_root=ROOT):
    root = Path(repo_root).resolve()
    for name, digest in hashes.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not (root / path).resolve().is_relative_to(root):
            raise ValueError('provenance paths must remain inside the repository')
        if sha256_file(root / path) != digest:
            raise ValueError(f'provenance checksum differs: {name}')


def frozen_provenance(repo_root=ROOT):
    """Validate the published A/B/C chain, never reinterpret old run metadata."""
    from .joint_null import load_frozen_inputs, validate_step2c_snapshot
    root = Path(repo_root).resolve()
    inputs = load_frozen_inputs(root / 'cache/countermine_rgb/step2b',
                                root / 'docs/audits/step2b_countermine_graph_metrics.json', repo_root=root)
    path = root / 'docs/audits/step2c_null_topology_metrics.json'
    snapshot = read_json(path)
    validate_step2c_snapshot(snapshot)
    saved = snapshot['provenance']
    for key in ('step2a_snapshot_sha256', 'step2b_snapshot_sha256', 'step2a_candidate_metrics_sha256',
                'step2a_random_metrics_sha256', 'graph_artifact_sha256', 'input_file_sha256', 'step2c_code_sha256'):
        if saved[key] != inputs['provenance'][key]:
            raise ValueError(f'frozen Step 2C source binding differs: {key}')
    hashes = dict(saved['input_file_sha256'])
    hashes.update(saved['step2c_code_sha256'])
    step2b = read_json(root / 'docs/audits/step2b_countermine_graph_metrics.json')
    hashes.update(step2b['provenance']['graph_code_sha256'])
    measurement = read_json(root / 'cache/countermine_rgb/step2a/measurement_config.json')
    hashes.update(measurement['source_sha256'])
    hashes.update(measurement['vendor_provenance']['source_sha256'])
    hashes[relative_path(path, root)] = sha256_file(path)
    for name, digest in snapshot['figure_sha256'].items():
        if Path(name).name != name or not name.startswith('step2c_'):
            raise ValueError('invalid frozen Step 2C curated figure name')
        hashes['docs/audits/' + name] = digest
    # Curated output hashes identify the completed run even if its runtime
    # directory is nested (as in the recorded runtime_fix experiment).
    stages = saved['step2c_stage_summary_sha256']
    matches = []
    for directory in [root / 'cache/countermine_rgb/step2c',
                      *(p for p in (root / 'cache/countermine_rgb/step2c').rglob('*') if p.is_dir())]:
        if all((directory / name).is_file() and sha256_file(directory / name) == digest
               for name, digest in stages.items()):
            matches.append(directory)
    if not matches:
        raise ValueError('completed frozen Step 2C runtime summaries are missing or changed')
    directory = sorted(matches)[0]
    for name, digest in {**stages, **snapshot['runtime_artifact_sha256']}.items():
        if Path(name).name != name:
            raise ValueError('invalid frozen Step 2C runtime artifact name')
        hashes[relative_path(directory / name, root)] = digest
    verify_hashes(hashes, root)
    return {'step2a_snapshot_sha256': saved['step2a_snapshot_sha256'],
            'step2b_snapshot_sha256': saved['step2b_snapshot_sha256'],
            'step2c_snapshot_sha256': sha256_file(path),
            'manifest_sha256': sha256_file(root / 'cache/gsv_mini/manifest.csv'),
            'raw_candidate_csv_sha256': sha256_file(root / 'cache/gsv_mini/rgb_candidates_raw.csv'),
            'input_file_sha256': hashes, 'real_rgb_only': True, 'synthetic_images_used': False,
            'git_commit': git_commit(root)}
