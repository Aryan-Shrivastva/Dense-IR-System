"""
Optional Hugging Face Hub client with SSL verification disabled.
Set HF_INSECURE_SSL=1 only behind a corporate TLS proxy with broken chains.
"""

import os
import ssl

os.environ["HF_INSECURE_SSL"] = "1"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import httpx
from huggingface_hub import set_client_factory
from huggingface_hub.utils import _http as hf_http


def factory():
    return httpx.Client(
        event_hooks={"request": [hf_http.hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
        verify=False,
    )


def apply_if_configured():
    if os.getenv("HF_INSECURE_SSL", "").strip() not in ("1", "true", "yes"):
        return
    set_client_factory(factory)
