"""
qc/pipeline.py — QCPipeline: end-to-end orchestrator.

Each stage is an explicit, ordered method so the pipeline can be driven from a notebook or 
test without argparse, and each stage is independently inspectable.

Flow
----
    load()   — TableStore (one read per table), site-level QC, platemap.
    render() — per plate: group images, build montages (parallel I/O), fit the
               ThresholdEngine, draw the collage + overview, aggregate MFI, and
               accumulate the per-plate payload dict.
    write()  — build window.__QC__ (payload/builder) and emit the HTML report.

Everything reads its dataset vocabulary from the active DatasetProfile: channel
order, filename regex and plate geometry all come from the .env-built profile,
so the same pipeline serves the internal and JUMP datasets unchanged.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from qc import profiles
from qc.profiles import DatasetProfile, profile_from_env
from qc.io.platemap import load_platemap
from qc.io.readers import TableStore
from qc.render.collage import (
    _aggregate_mfi_per_well, _aggregate_mfi_sites,
    _well_passes_all, _well_passes_group,
    render_plate_collage,
)
from qc.render.montage import build_montages_parallel
from qc.render.overview import _make_overview_grid, _well_montage_b64
from qc.stats.aggregate import collapse_sites_to_wells_with_sources
from qc.stats.engine import ThresholdEngine

# Plate id embedded in the image filename, e.g. "..._P25.tiff".
_PLATE_IN_NAME = re.compile(r"_(P\d+)")


@dataclass
class PipelineConfig:
    input_dir: Path
    output_path: Path
    profile: DatasetProfile = field(default_factory=profile_from_env)
    qc_tsv: Path | None = None
    platemap: Path | None = None
    sites_per_well: int = 9
    scale: float = 0.5
    workers: int = 8
    band_height: int = 280
    font_size: int = 18
    n_sigma: float = 3.0
    web_scale: float = 0.2
    emit_pdf: bool = True

    @property
    def cohort_name(self) -> str:
        return self.profile.cohort

    @property
    def plate_rows(self) -> int:
        return self.profile.plate_rows

    @property
    def plate_cols(self) -> int:
        return self.profile.plate_cols


class QCPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.profile = cfg.profile
        self.engine = ThresholdEngine(cfg.profile, n_sigma=cfg.n_sigma)
        self.output_path = Path(cfg.output_path)
        self.input_dir = Path(cfg.input_dir)

        self.store: TableStore | None = None
        self.qc_sites: dict = {}
        self.qc: dict = {}
        self.platemap: dict = {}
        self._mfi_sources: dict = {}
        self._channel_map: dict = {}
        self._mfi_img: dict = {}
        self._mfi_img_sites: dict = {}
        self._radius_all: dict = {}
        self._plates_data: list[dict] = []

    # ── Stage 1: load ────────────────────────────────────────────────────────
    def load(self) -> "QCPipeline":
        self.output_path.mkdir(parents=True, exist_ok=True)

        measurements = self.input_dir.resolve().parent / "Measurements"
        if not measurements.is_dir():
            measurements = self.input_dir / "Measurements"
        self.store = TableStore(measurements, self.profile)

        # Site-level QC (single read of the Image table), well level derived.
        self.qc_sites = self.store.qc_sites()
        if not self.qc_sites:
            print(f"[warn] No QC measurements found in {measurements}")
        self.qc, self.qc_worst_sites = collapse_sites_to_wells_with_sources(
            self.qc_sites, self.profile)

        # MFI (single read of Cells/Nuclei), image-level MFI, radius (all plates).
        self._mfi_sources, self._channel_map = self.store.mfi_sources()
        self._mfi_img = self.store.mfi_img()
        self._mfi_img_sites = self.store.mfi_img_sites()
        self._radius_all = self.store.radius_all_plates()

        # Platemap: explicit path, else first platemap_*.csv in input dir.
        self.platemap = load_platemap(self.cfg.platemap, self.profile)
        if not self.platemap:
            candidates = sorted(self.input_dir.glob("platemap_*.csv"))
            if candidates:
                print(f"[platemap] Auto-detected: {candidates[0]}")
                self.platemap = load_platemap(candidates[0], self.profile)
        return self

    # ── Stage 2: render ──────────────────────────────────────────────────────
    def render(self) -> "QCPipeline":
        plates = self._group_by_plate()
        for plate_name, files in sorted(plates.items()):
            print(f"\n[plate] {plate_name}")
            wells = self._group_well_imgs(files)
            montages = build_montages_parallel(
                wells, scale=self.cfg.scale,
                sites_per_well=self.cfg.sites_per_well, workers=self.cfg.workers)
            entry = self._render_plate(plate_name, montages)
            if entry is not None:
                self._plates_data.append(entry)

        if not self._plates_data:
            raise SystemExit(f"[ERROR] No plate processed from {self.input_dir}")
        return self

    # ── Stage 3: write ───────────────────────────────────────────────────────
    def write(self) -> None:
        from qc.output.html import render_report
        # Build the payload once (expensive: base64-encodes every collage) and
        # reuse it for both HTML and the PDF summary so they never diverge.
        payload = self._build_payload()

        html_name = f"{self.cfg.cohort_name}_QC_report.html"
        render_report(
            cohort_name=self.cfg.cohort_name,
            plates_data=self._plates_data,
            output_path=self.output_path / html_name,
            profile=self.profile,
            web_scale=self.cfg.web_scale,
            payload=payload,
        )
        if self.cfg.emit_pdf:
            from qc.output.pdf import summary_pdf
            pdf_name = f"{self.cfg.cohort_name}_QC_summary.pdf"
            try:
                summary_pdf(payload, self.output_path / pdf_name)
            except Exception as e:
                # PDF is a convenience artifact; never let it break the HTML run.
                print(f"[warn] PDF summary skipped: {e}")

        try:
            from qc.stats.site_flags import write_flagged_sites_parquet
            fq_name = f"{self.cfg.cohort_name}_flagged_sites.parquet"
            n = write_flagged_sites_parquet(
                self.qc_sites, self.profile, self.output_path / fq_name)
            print(f"[info] Flagged-sites parquet: {n} site(s) → {fq_name}")
        except Exception as e:
            print(f"[warn] Flagged-sites parquet skipped: {e}")

    def run(self) -> None:
        self.load().render().write()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _build_payload(self) -> dict:
        from qc.payload.builder import build_payload
        return build_payload(self.cfg.cohort_name, self._plates_data, self.profile)

    def _group_by_plate(self) -> dict:
        plates = defaultdict(list)
        for f in self.input_dir.glob("*.tiff"):
            m = _PLATE_IN_NAME.search(f.name)
            if m:
                plates[m.group(1)].append(f)
        return plates

    def _group_well_imgs(self, files: list) -> dict:
        wells = defaultdict(dict)
        skipped = 0
        for f in files:
            try:
                r, c, s = self.profile.parse_name(f.name)
            except ValueError as e:
                skipped += 1
                if skipped <= 5:
                    print(f"[warn] {e}", flush=True)
                continue
            wells[(r, c)][s] = f
        if skipped:
            print(f"[warn] {skipped} files could not be parsed in this group")
        return wells

    def _lookup_plate(self, mapping: dict, plate_name: str, label: str = "") -> dict:
        """Robust plate-key lookup across P-prefix / zero-pad / case variants,
        with a single-entry fallback. Ported from Collage._lookup_plate."""
        if not mapping:
            return {}

        def _norm(k: str) -> str:
            k = str(k).strip().lower()
            if k.startswith("p"):
                return f"p{k[1:].lstrip('0') or '0'}"
            return k.lstrip("0") or "0"

        norm_map = {_norm(k): k for k in mapping}
        for cand in (plate_name, plate_name.lstrip("Pp"),
                     f"P{plate_name.lstrip('Pp')}"):
            if cand in mapping:
                return mapping[cand]
            nk = _norm(cand)
            if nk in norm_map:
                return mapping[norm_map[nk]]
        if len(mapping) == 1:
            key = next(iter(mapping))
            print(f"  [warn] {label}: no key matched '{plate_name}'; using sole '{key}'.")
            return mapping[key]
        print(f"  [warn] {label}: no key matched '{plate_name}'. Keys: {list(mapping)[:10]}")
        return {}

    def _render_plate(self, plate_name: str, montages: dict) -> dict | None:
        if not montages:
            print(f"  No images for {plate_name}, skipping.")
            return None

        plate_qc = self._lookup_plate(self.qc, plate_name, "QC")
        plate_map = self._lookup_plate(self.platemap, plate_name, "platemap")
        plate_qc_sites = self._lookup_plate(self.qc_sites, plate_name, "QC-sites")
        plate_worst_sites = self._lookup_plate(
            self.qc_worst_sites or {}, plate_name, "QC-worst-sites")

        if plate_qc:
            self.engine.fit(plate_qc)

        collage = render_plate_collage(
            plate_name, montages, plate_qc, plate_map, self.engine,
            plate_rows=self.cfg.plate_rows, plate_cols=self.cfg.plate_cols,
            band_height=self.cfg.band_height, font_size=self.cfg.font_size)

        n_wells = len(plate_qc)
        n_pass = sum(1 for m in plate_qc.values()
                     if _well_passes_all(m, self.engine))
        n_illum = sum(1 for m in plate_qc.values()
                      if _well_passes_group(m, list(profiles.ILLUM_METRICS), self.engine))

        overview_arr, ov_cw, ov_ch = _make_overview_grid(
            montages, plate_qc, plate_map,
            plate_rows=self.cfg.plate_rows, plate_cols=self.cfg.plate_cols)

        # Flagged well thumbnails (absolute OR adaptive failure in any metric).
        flagged_b64 = {}
        metric_cols = self.profile.metric_cols()
        col_to_ch = self.profile.col_to_channel()
        for wl, m in plate_qc.items():
            flagged = any(
                self.engine.passes(m.get(col), mk, col_to_ch.get(col, "")) is False
                for mk, cols in metric_cols.items() for col in cols)
            if flagged:
                r_idx = ord(wl[0]) - ord("A") + 1
                c_idx = int(wl[1:])
                b64 = _well_montage_b64(montages, (r_idx, c_idx), scale_factor=0.33)
                if b64:
                    flagged_b64[wl] = b64

        from qc import schema
        plate_name_norm = schema.normalize_plate(plate_name)
        radius_data = self._radius_all.get(plate_name_norm, {})

        return {
            "name": plate_name,
            "plate_rows": self.cfg.plate_rows,
            "plate_cols": self.cfg.plate_cols,
            "collage_arr": collage,
            "overview_arr": overview_arr,
            "overview_cw": ov_cw,
            "overview_ch": ov_ch,
            "plate_qc": plate_qc,
            "plate_qc_sites": plate_qc_sites,
            "plate_worst_sites": plate_worst_sites,
            "plate_map": plate_map,
            "flagged_b64": flagged_b64,
            "engine_adaptive": self.engine._adaptive,
            "mfi_data": (_aggregate_mfi_per_well(self._mfi_sources, self._channel_map,
                                                 plate_name) if self._channel_map else {}),
            "mfi_img": self._mfi_img,
            "mfi_sites": (_aggregate_mfi_sites(self._mfi_sources, self._channel_map,
                                               plate_name) if self._channel_map else {}),
            "mfi_img_sites": self._mfi_img_sites,
            "radius_data": radius_data,
            "pass_rate": 100 * n_pass / n_wells if n_wells else 0,
            "n_wells": n_wells,
            "n_pass": n_pass,
            "n_illum_pass": n_illum,
        }