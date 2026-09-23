"""
qc/output/html.py — Self-contained HTML report assembly.

Moved from report.render_report. Behaviour is unchanged; the only structural
difference is that the payload now comes from qc.payload.builder.build_payload
(which reads from qc.stats / qc.render) instead of being computed inline. That
removes the import of statistics helpers from III_QC_collage and breaks the
circular import.

The report embeds Plotly, the payload, CSS, JS and UTIF so the resulting HTML is
fully self-contained and can be shared by email without any external assets.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime
from pathlib import Path

from qc import profiles
from qc.payload.builder import build_payload

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _fetch_plotly_js() -> str:
    """Download Plotly JS once (cached next to this file) for embedding."""
    cache = Path(__file__).resolve().parent.parent / ".plotly_cache.js"
    if cache.exists():
        return cache.read_text()
    print("[html] Downloading Plotly JS for embedding (one-time)…")
    try:
        with urllib.request.urlopen(_PLOTLY_URL, timeout=30) as r:
            js = r.read().decode()
        cache.write_text(js)
        return js
    except Exception as e:
        print(f"[warn] Could not fetch Plotly: {e}. HTML will use CDN fallback.")
        return ('/* CDN fallback */\n'
                f'document.write(\'<script src="{_PLOTLY_URL}"></script>\');')


def _read_asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def render_report(cohort_name: str, plates_data: list[dict],
                  output_path: Path, profile: profiles.DatasetProfile,
                  web_scale: float = 0.2, payload: dict | None = None) -> dict:
    """Build the self-contained HTML QC report from assets + computed payload.

    Returns the payload dict so callers (e.g. the PDF stage) can reuse it without
    rebuilding — building the payload is expensive (it encodes every collage as
    base64). Pass a pre-built `payload` to skip construction entirely.

    Substitution order matters: the short text markers (__COHORT__,
    __GENERATED__) are replaced FIRST, while the template still contains only our
    own markup. The large injected blobs (Plotly, payload, JS) go LAST, so a
    literal like "__COHORT__" appearing inside injected data (e.g. a compound
    name) is treated as data and not accidentally substituted.
    """
    plotly_js = _fetch_plotly_js()
    qc_payload = payload if payload is not None else \
        build_payload(cohort_name, plates_data, profile)

    payload_js = "window.__QC__ = " + json.dumps(qc_payload) + ";"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = _read_asset("report.html")
    html = html.replace("__COHORT__",    cohort_name)
    html = html.replace("__GENERATED__", generated)
    html = html.replace("__CSS__",       _read_asset("report.css"))
    html = html.replace("__PLOTLY__",    plotly_js)
    html = html.replace("__PAYLOAD__",   payload_js)
    html = html.replace("__JS__",        _read_asset("report.js"))
    html = html.replace("__UTIF__",      _read_asset("utif.js"))

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"[html] Report written to {output_path}")
    return qc_payload
