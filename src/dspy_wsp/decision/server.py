"""A local HTTP server for one decision backend, speaking TypeSafe's System One API.

`POST /v1/systemone` takes `{"model", "state", "questions"}` and returns
`{"model", "answers", "usage"}`. `GET /v1/models` lists the one model. The server
answers with the backend's current weights; it ignores the API key.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dspy_wsp.decision.schema import RequestError, parse_request

MAX_BODY_BYTES = 64 * 1024 * 1024


def _decompress(body: bytes, encoding: str | None) -> bytes:
    if not encoding or encoding == "identity":
        return body
    if encoding != "zstd":
        raise RequestError(f"unsupported Content-Encoding {encoding!r}")
    try:
        from compression.zstd import decompress
    except ImportError:
        try:
            from backports.zstd import decompress
        except ImportError:
            raise RequestError("zstd request bodies need Python 3.14+ or backports.zstd") from None
    return decompress(body)


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args):  # requests are recorded by the forward tool, not the console
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            return self._send(HTTPStatus.OK, {"object": "list", "data": [{"id": self.server.backend.model_id, "object": "model"}]})
        self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/systemone":
            return self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise RequestError("request body too large")
            body = json.loads(_decompress(self.rfile.read(length), self.headers.get("Content-Encoding")))
            state, questions = parse_request(body)
            out = self.server.backend.system_one(state, questions)
        except (RequestError, json.JSONDecodeError, UnicodeDecodeError) as e:
            return self._send(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - a model failure is the server's fault
            return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(e).__name__}: {e}"})
        self._send(HTTPStatus.OK, {"model": self.server.backend.model_id, **out})


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend):
        super().__init__(address, _Handler)
        self.backend = backend


class SystemOneServer:
    def __init__(self, backend, host: str = "127.0.0.1", port: int = 0):
        self.backend = backend
        self._httpd = _Server((host, port), backend)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> SystemOneServer:
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="wsp-systemone", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
