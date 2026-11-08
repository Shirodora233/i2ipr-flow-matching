"""Load a separately obtained, hash-pinned upstream implementation.

No upstream source, notebook execution, weight download or dataset acquisition
is performed by this adapter. Only the selected model class is evaluated.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from diffusers import UNet2DModel

value = os.environ.get('I2IPR_UPSTREAM')
if not value:
    raise RuntimeError('Set I2IPR_UPSTREAM to your separately obtained official source root.')
root = Path(value).expanduser().resolve(strict=True)
pins = json.loads(Path(__file__).with_name('upstream_sources.json').read_text(encoding='utf-8'))
for name, expected in pins['files'].items():
    candidate = (root / name).resolve(strict=True)
    if root not in candidate.parents:
        raise RuntimeError('External source path escapes the selected source root.')
    actual = hashlib.sha256(candidate.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
    if actual != expected:
        raise RuntimeError('External source hash differs from the recorded upstream snapshot: ' + name)

package_name = '_i2ipr_external_utils'
spec = importlib.util.spec_from_file_location(package_name, root / 'utils/__init__.py',
                                             submodule_search_locations=[str(root / 'utils')])
package = importlib.util.module_from_spec(spec)
sys.modules[package_name] = package
spec.loader.exec_module(package)
from _i2ipr_external_utils.utils import fft2d, ifft2d, zero_padding_twice, crop_center_half
from _i2ipr_external_utils.algorithms import apply_image_constraint_hio, get_algorithm


def build_model():
    """Same upstream raw UNet used by the experiment, without notebook loading."""
    tree = ast.parse((root / 'models/denoisers/unet_hf.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'UNet2DModel11Channels')
    cls.decorator_list = []
    ns = {'nn': nn, 'torch': torch, 'UNet2DModel': UNet2DModel}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 'external_UNet11', 'exec'), ns)
    return SimpleNamespace(denoiser=ns['UNet2DModel11Channels']().cuda())
