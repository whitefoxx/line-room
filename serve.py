#!/usr/bin/env python3
"""Static server for local development, with caching turned off.

Any static server works — this one exists because Python's stock http.server
sends Last-Modified and no Cache-Control, so Chrome applies heuristic caching.
You edit index.html, reload, and the page silently runs the previous version,
which looks exactly like your edit being wrong. That cost three separate
debugging detours here. It also gets the .wasm MIME type right.

    python3 serve.py [port]        # default 8231
"""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",   # so instantiateStreaming works
        ".cact": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8231
    print(f"line-room on http://127.0.0.1:{port}/  (no-store)")
    ThreadingHTTPServer(("127.0.0.1", port), partial(NoCacheHandler)).serve_forever()
