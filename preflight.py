"""Read-only standard-library checks; never imports torch or starts computation."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--upstream', help='Official source root; defaults to I2IPR_UPSTREAM')
args = ap.parse_args()
here = Path(__file__).resolve().parent
issues = []
for path in here.glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
versions = {}
for package in ['torch', 'diffusers', 'numpy', 'Pillow', 'einops', 'tqdm',
                'huggingface-hub', 'safetensors']:
    try: versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
        issues.append('Missing package: ' + package)
for line in (here / 'requirements.txt').read_text().splitlines():
    if not line or line.startswith('#'): continue
    name, constraints = re.match(r'([A-Za-z0-9-]+)(.*)', line).groups()
    actual = versions.get(name)
    if actual is None: continue
    # This local requirements file uses only stable numeric ==, >= and < bounds.
    version = tuple(int(x) for x in actual.split('+')[0].split('.')[:3])
    version = version + (0,) * (3 - len(version))
    for operator, value in re.findall(r'(==|>=|<)([0-9.]+)', constraints):
        wanted = tuple(int(x) for x in value.split('.'))
        wanted = wanted + (0,) * (3 - len(wanted))
        okay = version == wanted if operator == '==' else version >= wanted if operator == '>=' else version < wanted
        if not okay: issues.append('Installed version violates requirement: ' + line)
value = args.upstream or os.environ.get('I2IPR_UPSTREAM')
if value:
    root = Path(value).expanduser()
    pins = json.loads((here / 'upstream_sources.json').read_text())
    for name, expected in pins['files'].items():
        try: actual = hashlib.sha256((root / name).read_text(encoding='utf-8').encode()).hexdigest()
        except OSError: actual = None
        if actual != expected: issues.append('Missing or mismatched external source: ' + name)
else: issues.append('I2IPR_UPSTREAM is not configured')
cfg = json.loads((here / 'configs/train.example.json').read_text())
for name, expected in cfg['code_sha256'].items():
    if hashlib.sha256((here / name).read_bytes()).hexdigest() != expected:
        issues.append('Example source hash mismatch: ' + name)
print(json.dumps({'status': 'blocked' if issues else 'static_dependencies_checked',
                  'issues': issues, 'versions': versions, 'GPU_accessed': False,
                  'example_config_is_not_training_authorization': True,
                  'training_or_inference_executed': False}, indent=2))
raise SystemExit(1 if issues else 0)
