"""Single place that pins what each Modal serving app loads. Pure stdlib.

Lives inside the package so Modal containers (which only receive jev_inference/)
import the same values that the local deploy baked into the image environment.

To deploy a new adapter revision: edit the `jev-nyotti` entry (or add a new app entry
and a matching modal file), set `action_hold_margin` to the margin frozen on
validation, then `modal deploy`. `image_env` validates the entry through Settings
before any image is built. The values below are the currently deployed ones; see
ADAPTER_DEPLOYMENT.json.

Numeric CPU app (modal_action_cpu.py): the artifact file is supplied at deploy time via
JEV_NUMERIC_MODEL_FILE, but its SHA-256 and hold margin are pinned HERE; the deploy refuses a
file whose hash differs and the container re-verifies it at startup.
"""
from __future__ import annotations

import json

SERVING = {
    # modal_app.py: original base checkpoint, no adapter.
    'jev-qwen-35-4b': dict(adapter_id=None, adapter_revision=None, adapter_sha256=None, action_hold_margin=0.0),
    # modal_adapter.py: the verified jev-nyotti canary.
    'jev-nyotti': dict(
        adapter_id='guzus/jev-nyotti',
        adapter_revision='73867def94f8b062700ad3f8d63128b4e1c9b1d4',
        adapter_sha256='918fdcd054e3d77116ddb7b708cc7c2a24aa443696f4639bd103408777051831',
        action_hold_margin=0.0,
    ),
}

# Numeric policy apps: set both pins from `training/numeric_export.py` output before deploying.
NUMERIC_SERVING = {
    # ACTION_V7 (training/ACTION_V7.md; gate PASSED on unseen 2022-04..12): V6 HGB weights + decision rules.
    'jev-nyotti-action-cpu': dict(numeric_model_sha256='95531fcda384c095101051bf689947f46177a3a97e47806d3d16bb25f157d284',
                                  action_hold_margin={'flat': -1.5, 'position': -0.9}),
}
NUMERIC_MODEL_REMOTE_PATH = '/opt/model/numeric_policy.json'


def numeric_image_env(app_name: str) -> dict[str, str]:
    """Validated image environment for a numeric CPU app. Raises until both pins are set."""
    from .settings import Settings

    entry = NUMERIC_SERVING[app_name]
    if entry['numeric_model_sha256'] is None or entry['action_hold_margin'] is None:
        raise ValueError(f'{app_name}: pin numeric_model_sha256 and action_hold_margin in deployment.py first')
    Settings(api_key='deployment-config-validation-only-0000', device='cpu', action_policy='numeric',
             numeric_model_path=NUMERIC_MODEL_REMOTE_PATH, **entry)
    return {'ACTION_POLICY': 'numeric', 'INFERENCE_DEVICE': 'cpu', 'NUMERIC_MODEL_PATH': NUMERIC_MODEL_REMOTE_PATH,
            'NUMERIC_MODEL_SHA256': entry['numeric_model_sha256'],
            'ACTION_HOLD_MARGIN': json.dumps(entry['action_hold_margin'], sort_keys=True) if isinstance(entry['action_hold_margin'], dict) else repr(float(entry['action_hold_margin']))}


def image_env(app_name: str) -> dict[str, str]:
    """Validated Modal image environment for one serving app (no credentials)."""
    from .settings import Settings

    entry = SERVING[app_name]
    # Validate with a throwaway key: the real key is only in the Modal secret.
    Settings(api_key='deployment-config-validation-only-0000', **entry)
    env = {'ACTION_HOLD_MARGIN': repr(float(entry['action_hold_margin']))}
    if entry['adapter_id']:
        env.update(LORA_MODEL_ID=entry['adapter_id'], LORA_REVISION=entry['adapter_revision'])
        if entry['adapter_sha256']:
            env['LORA_SHA256'] = entry['adapter_sha256']
    return env
