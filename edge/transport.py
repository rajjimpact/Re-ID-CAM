"""
edge/transport.py — Payload delivery from edge to central.

In the single-machine demo (run_demo.py) the edge worker calls the central
engine's ingest() directly in-process — no network hop.

In a real distributed deployment each edge node calls http_post() to deliver
TrackPayload JSON to the central FastAPI /ingest endpoint.
"""
from __future__ import annotations
from typing import Any, Optional
import json

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class HttpSink:
    """
    Delivers TrackPayload dicts to a remote FastAPI /ingest endpoint via HTTP POST.
    Used only when edge and central run on different machines.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.ingest_url = f"{base_url.rstrip('/')}/ingest"

    def send(self, payload_dict: dict) -> bool:
        """
        POST the payload dict as JSON to /ingest.
        Returns True on HTTP 2xx, False otherwise.
        """
        if not _HAS_REQUESTS:
            print("[transport] 'requests' library not installed — cannot send HTTP payload.")
            return False
        try:
            resp = _requests.post(self.ingest_url, json=payload_dict, timeout=2.0)
            return resp.ok
        except Exception as e:
            print(f"[transport] HTTP send failed: {e}")
            return False


class InProcessSink:
    """
    Calls the central engine's ingest method directly in the same process.
    Zero serialisation overhead — used in run_demo.py.
    """

    def __init__(self, ingest_fn) -> None:
        self._ingest = ingest_fn

    def send(self, payload_dict: dict) -> bool:
        try:
            self._ingest(payload_dict)
            return True
        except Exception as e:
            print(f"[transport] In-process ingest failed: {e}")
            return False


def build_http_sink(base_url: str = "http://localhost:8000") -> HttpSink:
    return HttpSink(base_url=base_url)
