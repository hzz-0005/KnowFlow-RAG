"""Compare deployed localhost assets to this checkout; UI reload is a separate step."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8080")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
frontend = root / "frontend"
html = (frontend / "index.html").read_text(encoding="utf-8")
app = (frontend / "app.js").read_text(encoding="utf-8")
app_url = re.search(r'<script src="([^"]+)" type="module">', html).group(1)
bundle_url = re.search(r'from "([^"]+)";', app).group(1)
resources = [
    ("/", "index.html"),
    (app_url, "app.js"),
    ("/styles.css", "styles.css"),
    (bundle_url, "dist/answer-view.js"),
]
opener = (
    build_opener(ProxyHandler({}))
    if urlparse(args.base_url).hostname in {"localhost", "127.0.0.1", "::1"}
    else build_opener()
)
rows = []
for url, path in resources:
    with opener.open(args.base_url.rstrip("/") + url, timeout=15) as response:
        served = response.read()
    local = (frontend / path).read_bytes()

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    rows.append(
        {
            "url": url,
            "file": path,
            "local_sha256": digest(local),
            "served_sha256": digest(served),
            "matches": local == served,
        }
    )
result = {
    "recorded_at": datetime.now(UTC).isoformat(),
    "resources": rows,
    "all_match": all(r["matches"] for r in rows),
    "browser_reload_verified": False,
    "note": "HTTP content check only; confirm actual browser script URL after reload separately.",
}
if args.output:
    if args.output.exists():
        raise SystemExit("Refusing to overwrite evidence")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
raise SystemExit(0 if result["all_match"] else 1)
