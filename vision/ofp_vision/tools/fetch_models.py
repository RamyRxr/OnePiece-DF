"""Download the MediaPipe .task model files.

    python -m ofp_vision.tools.fetch_models
"""

from __future__ import annotations

import urllib.request

from ..config import MODEL_FILES, MODEL_URLS, MODELS_DIR


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in MODEL_FILES.items():
        target = MODELS_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {filename} already present ({target.stat().st_size} bytes)")
            continue
        url = MODEL_URLS[name]
        print(f"[get ] {filename} <- {url}")
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            target.write_bytes(data)
            print(f"[ok  ] {filename} ({len(data)} bytes)")
        except Exception as exc:
            print(f"[fail] {filename}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
