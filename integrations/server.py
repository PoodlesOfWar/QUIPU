"""Static server for the integrations site: stdlib, GET/HEAD only, strict headers, /health."""
from __future__ import annotations

import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
    "Cross-Origin-Resource-Policy": "same-origin", "Cache-Control": "no-cache",
}


class Handler(SimpleHTTPRequestHandler):
    server_version = "integrations"
    sys_version = ""

    def end_headers(self):
        for k, v in HEADERS.items():
            self.send_header(k, v)
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def _no(self):
        self.send_error(405)
        
    do_POST = do_PUT = do_DELETE = do_PATCH = lambda self: self._no()

    def list_directory(self, path):               # no directory listings
        self.send_error(404)
        return None

    def log_message(self, fmt, *args):
        sys.stderr.write("integrations: " + fmt % args + "\n")


def make_server(host="127.0.0.1", port=8090, root=SITE):
    return ThreadingHTTPServer((host, port), partial(Handler, directory=root))


if __name__ == "__main__":
    make_server(os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", "8090"))).serve_forever()
