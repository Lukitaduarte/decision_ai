"""A local Decision API provider for tests: `POST /v1/systemone` with `{"model", "state", "questions"}` and a bearer key,
answered by the Python reference implementation (dinah.py from the model folder).

    python tool/mock_decision_server.py --model-dir <folder with dinah.py> --port 8766 --key test-key
"""
from __future__ import annotations

import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--key", default="test-key"); ap.add_argument("--path", default="/v1/systemone")
    a = ap.parse_args()
    sys.path.insert(0, a.model_dir)
    from dinah import DinahONNX
    model = DinahONNX.from_pretrained(a.model_dir)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            if self.path != a.path:
                return self._send(404, {"error": "not found"})
            if self.headers.get("Authorization") != f"Bearer {a.key}":
                return self._send(401, {"error": "invalid api key"})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            keys = list(req["questions"])
            qs = [dict(req["questions"][k], state=req.get("state")) for k in keys]
            try:
                answers = model.predict(qs)
            except ValueError as e:
                return self._send(413, {"error": f"context window: {e}"})
            self._send(200, {"model": req.get("model", "dinah-0"), "answers": dict(zip(keys, answers)), "usage": {"cost": 0}})

        def log_message(self, *args):
            pass

    print(f"serving {a.path} on 127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
