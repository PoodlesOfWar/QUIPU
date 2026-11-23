"""Speech-to-text service for edge devices: audio in, text out.  Nothing else.

``POST /transcribe`` (bearer token) takes a short audio clip and returns ``{"ok": true, "text": ...}``.
It has no access to QUIPU's ledger, observer or HubCore and cannot stage, authorise or realise anything:
a transcript is untrusted text the *client* then stages as a held proposal, like any other agent input.

faster-whisper is optional and loaded lazily (``QUIPU_STT_MODEL``, default ``base.en``, CPU int8).  Without it
the service answers 503 and says how to install it, rather than pretending.  Fails closed without
``QUIPU_STT_TOKEN``.  Clips are size-capped, written to a private temp file and deleted straight after.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import urlparse

log = logging.getLogger("quipu.voice_stt")

TOKEN_ENV = "QUIPU_STT_TOKEN"
MODEL_ENV = "QUIPU_STT_MODEL"
MAX_BYTES = int(os.environ.get("QUIPU_STT_MAX_BYTES", "") or 8 * 1024 * 1024)
MAX_TEXT = 2000
DEFAULT_PORT = 7120
AUDIO_TYPES = ("audio/", "application/octet-stream", "video/mp4", "video/webm")

Transcriber = Callable[[str], tuple[str, str]]          # path -> (text, model name)


class SttUnavailable(RuntimeError):
    pass


_model = None
_model_lock = threading.Lock()


def faster_whisper_transcriber(path: str) -> tuple[str, str]:
    global _model
    name = os.environ.get(MODEL_ENV, "").strip() or "base.en"
    with _model_lock:
        if _model is None:
            try:
                from faster_whisper import WhisperModel          # heavy, optional
            except ImportError:
                raise SttUnavailable("faster-whisper is not installed on the server "
                                     "(pip install 'faster-whisper>=1,<2')") from None
            _model = WhisperModel(name, device="cpu", compute_type="int8")
        model = _model
    segments, _info = model.transcribe(path, vad_filter=True, language="en" if name.endswith(".en") else None)
    return " ".join(s.text.strip() for s in segments).strip(), name


def _expected_token() -> Optional[str]:
    t = os.environ.get(TOKEN_ENV, "").strip()
    return t or None


class _Handler(BaseHTTPRequestHandler):
    server_version = "quipu-stt"
    sys_version = ""
    transcriber: Transcriber = staticmethod(faster_whisper_transcriber)
    lock = threading.Semaphore(1)                  # one transcription at a time: CPU-bound

    def log_message(self, fmt, *a):                 # no request lines (no tokens, no paths)
        log.debug(fmt, *a)

    def _send(self, code: int, obj: dict, extra: Optional[dict] = None) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._send(200, {"ok": True, "service": "quipu-stt",
                             "auth": "token" if _expected_token() else "closed (token unset)"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/transcribe":
            return self._send(404, {"error": "not found"})
        expected = _expected_token()
        if expected is None:
            return self._send(401, {"error": f"{TOKEN_ENV} unset: the operator must set a token"})
        auth = self.headers.get("Authorization") or ""
        got = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if not got or not hmac.compare_digest(got, expected):
            return self._send(401, {"error": "unauthorized"}, {"WWW-Authenticate": 'Bearer realm="quipu-stt"'})
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not ctype.startswith(AUDIO_TYPES):
            return self._send(415, {"error": "send audio (Content-Type: audio/*)"})
        try:
            n = int(self.headers.get("Content-Length") or "")
        except ValueError:
            return self._send(411, {"error": "Content-Length required"})
        if n <= 0:
            return self._send(400, {"error": "empty body"})
        if n > MAX_BYTES:
            return self._send(413, {"error": f"clip over {MAX_BYTES} bytes"})
        data = self.rfile.read(n)
        if len(data) != n:
            return self._send(400, {"error": "short body"})
        fd, path = tempfile.mkstemp(prefix="quipu-stt-", suffix=".audio")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            with self.lock:
                text, model = self.transcriber(path)
        except SttUnavailable as exc:
            return self._send(503, {"error": str(exc)})
        except Exception:
            log.exception("transcription failed")
            return self._send(422, {"error": "could not transcribe that clip"})
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        self._send(200, {"ok": True, "text": (text or "")[:MAX_TEXT], "model": model})


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                transcriber: Optional[Transcriber] = None) -> ThreadingHTTPServer:
    h = type("SttHandler", (_Handler,), {"transcriber": staticmethod(transcriber or faster_whisper_transcriber),
                                         "lock": threading.Semaphore(1)})
    return ThreadingHTTPServer((host, port), h)


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:     # pragma: no cover
    httpd = make_server(host, port)
    print(f"[quipu-stt] on http://{host}:{port}/transcribe  auth: "
          f"{'token' if _expected_token() else 'CLOSED until ' + TOKEN_ENV + ' is set'}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":                                                  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()
    serve(a.host, a.port)
