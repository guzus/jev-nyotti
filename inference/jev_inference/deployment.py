"""Single place that pins what each Modal serving app loads. Pure stdlib.

Lives inside the package so Modal containers (which only receive jev_inference/)
import the same values that the local deploy baked into the image environment.

To deploy a new adapter revision: edit the `jev-nyotti` entry (or add a new app entry
and a matching modal file), set `action_hold_margin` to the margin frozen on
validation, then `modal deploy`. `image_env` validates the entry through Settings
before any image is built. The values below are the currently deployed ones; see
ADAPTER_DEPLOYMENT.json.
"""
from __future__ import annotations

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
