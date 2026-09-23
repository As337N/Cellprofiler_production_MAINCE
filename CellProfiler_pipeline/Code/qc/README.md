# qc — Cell Painting QC Report Pipeline

Generates a self-contained, interactive HTML QC report (plus an optional PDF
summary) from CellProfiler microscopy output. It reads per-image and per-object
measurements, computes quality statistics, renders plate collages, and assembles
everything into a single shareable HTML file.

The pipeline is dataset-agnostic: channel set, filename layout, and plate
geometry all come from the environment (`.env`), so the same code serves the
internal cohort and JUMP datasets unchanged.

---

## Quick start

Run from the parent of `qc/` (i.e. `Code/`), with the dataset variables loaded:

```bash
cd Code
set -a; source .env; set +a
python -m qc.cli -i "$IMAGES_DIR" -o "$OUTPUT_DIR" --platemap "$PLATEMAP" --no-pdf
```

Outputs, written to `OUTPUT_DIR`:

- `<COHORT>_QC_report.html` — the full interactive report (self-contained).
- `<COHORT>_QC_summary.pdf` — executive summary (omit with `--no-pdf`).

### Requirements

Python 3.8+, with `polars`, `numpy`, `pillow`, `tifffile`, `opencv-python`,
`scipy`, and `matplotlib` (the last only for the PDF). No `scikit-learn`
dependency — the PCA uses a numpy SVD.

---

## Configuration (`.env`)

The dataset profile is built from four environment variables:

| Variable          | Meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `COHORT`          | Cohort label; names the output files.                          |
| `CHANNEL_DICT`    | JSON `{channel: index}`, e.g. `{"Mito":"1","Hoechst":"6"}`. Sets channel order. |
| `FILENAME_REGEX`  | Regex with named groups `Row`, `Column`, `Field` to parse image filenames. |
| `WELLS_PER_PLATE` | `96` (8×12) or `384` (16×24). Sets plate geometry.             |

**Filename regex — important:** capture only up to the `Field`, and do **not**
anchor the end. Filenames carry a `_P<plate>` suffix before the extension
(`001002-4-001002004_P11.tiff`), so an end-anchored pattern (`…\.tiff$`) fails to
match. A prefix pattern is correct:

```
FILENAME_REGEX='^\d(?P<Row>\d{2})\d(?P<Column>\d{2})-(?P<Field>\d)-'
```

---

## CLI options

`-i/--input` and `-o/--output` are required; the rest have defaults.

| Flag            | Default | Purpose                                            |
|-----------------|---------|----------------------------------------------------|
| `-i, --input`   | —       | Directory of `.tiff` images.                       |
| `-o, --output`  | —       | Output directory.                                  |
| `--qc`          | auto    | Explicit QC measurements path.                     |
| `--platemap`    | auto    | Platemap CSV (else auto-detects `platemap_*.csv`). |
| `--sites`       | 9       | Sites per well.                                    |
| `--scale`       | 0.5     | Montage downscale factor.                          |
| `--workers`     | 8       | Parallel image-loading threads.                    |
| `--band-height` | 280     | Collage info-band height (px).                     |
| `--font`        | 18      | Collage font size.                                 |
| `--n-sigma`     | 3.0     | MAD sigma for adaptive thresholds.                 |
| `--web-scale`   | 0.2     | Web thumbnail scale.                               |
| `--no-pdf`      | off     | Skip the PDF summary.                              |

Cohort, channels, geometry, and filename parsing are **not** flags — they come
from the `.env`, so there is a single source of truth.

---

## Architecture

Data flows `load → render → write`, orchestrated by `QCPipeline` (`pipeline.py`).

```
qc/
├── cli.py            Entry point (python -m qc.cli).
├── pipeline.py       QCPipeline: load → render → write.
├── profiles.py       DatasetProfile from .env; shared constants (thresholds, colours).
├── schema.py         Canonical column names + plate-id normalisation.
├── io/
│   ├── readers.py    TableStore: one read per parquet table (Cells/Nuclei/Image); MFI, radius, qc_sites.
│   └── platemap.py   Platemap loading + geometry validation.
├── stats/
│   ├── expressions.py  Polars expressions + absolute pass/fail.
│   ├── engine.py       ThresholdEngine: absolute + adaptive MAD bounds.
│   ├── aggregate.py    Site → well collapse.
│   ├── scores.py       MAD scores, colour scales, red-well counts.
│   └── pca.py          PCA over well features → deviant-plate detection.
├── render/
│   ├── montage.py    Parallel image loading, well montages.
│   ├── overview.py   Plate overview grid.
│   └── collage.py    Full plate collage + per-well MFI aggregation.
├── payload/
│   └── builder.py    Assembles window.__QC__ (the single JSON the frontend reads).
├── output/
│   ├── html.py       render_report: self-contained HTML from assets + payload.
│   └── pdf.py         summary_pdf: matplotlib PdfPages executive summary.
└── assets/           report.html, report.css, report.js, utif.js (frontend, non-Python).
```

**Design principles.** Statistics run on Polars; each parquet table is read once
(`TableStore`). The MAD scale factor is exact (`1/norm.ppf(0.75)`), not the
truncated `1.4826`. New payload fields are additive so existing frontend
consumers never break. The HTML embeds Plotly, the payload, CSS, JS and the TIFF
decoder, so the report is fully self-contained and email-shareable.

---

## Report features

The HTML report has six collapsible sections. Click any section heading to
collapse or expand it (all open on load; state is not persisted).

### Cohort Summary
Per-plate table: well counts and red-well counts per metric family. The Slope and
MedianIntensity columns use the cohort MAD threshold (a well counts as red when
its cohort MAD z-score exceeds 3).

### Plate Browser
Plate collage with a clickable overview. Selecting a well shows its 3×3 site
montage; selecting a site shows an enlarged crop (see below).

- **Verdict-criteria toggles.** Four checkboxes (Slope, PctMax, PctMin, MedInt)
  control which metric families count toward each well's overall colour verdict
  (overview + montage). Disable a family that always fails in your dataset (e.g.
  PctMinimal) so it stops masking other error sources. The per-metric detail
  tables and per-family red-count columns are unaffected.
- **Single-site crop.** Clicking a site tile crops that site out of the montage
  and draws it enlarged, in its own panel between the montage and the plate
  overview, for eyeballing segmentation. It reuses the montage already loaded
  (no extra file weight); it does **not** overlay segmentation contours.

### QC Metrics
Per-channel heatmaps. **Slope and MedianIntensity are coloured by discrete MAD
bands** over the |cohort z-score|: green ≤1× (typical), amber 1–3× (moderate),
red >3× (strong outlier) — uniform across every channel. The colour reflects
*deviation from the cohort*, not the absolute metric value: a well that is
atypical in absolute terms but typical for the cohort reads green. Other metrics
(PctMax, PctMin) keep their fixed absolute colour scales. The compound selector
filters which wells are shown, in every heatmap.

### Cell Counts
Per-channel count heatmaps (Cells, Nuclei, artifacts).

### MFI (Median Fluorescence Intensity)
Per-channel boxplots by row and column, plus:

- **Download MFI CSV.** The button in the section header exports one row per
  site — `plate, well, site_index, <channel>_obj, <channel>_img` — aligned by the
  real ImageNumber, for the whole cohort. Object and image MFI line up per site.
- **Positional MFI drift.** Per-channel tables giving each row's and column's %
  drift of its median MFI versus two references: the current plate, and the whole
  cohort. Cells are coloured by |drift| on a continuous green→amber→red gradient.
  Drift is computed on the median (robust), and a channel with a very low
  reference median can show inflated percentages.

### Plate PCA (deviant-plate detection)
One PCA fitted on all wells (every numeric feature, z-scored), projecting wells
and per-plate centroids into the same 2-D space. A plate whose centroid sits far
from the cluster is a candidate outlier; each plate carries a *deviation* score
(distance from the global centroid in PC space). Well hover shows plate, well and
compound. **Hidden when there are fewer than 3 plates or 6 wells** — a PCA needs
enough points to be meaningful. Note that PC1+PC2 rarely capture all the
variance, so treat the plot as a screening tool, not a verdict.

### Image Viewer (optional)
Activates when a CSV + TIFFs are loaded in-browser.

---

## Notes & caveats

- **Robust metrics** are exactly two: `PowerLogLogSlope` and `MedianIntensity`.
  Only these use MAD-based colouring; PctMax/PctMin are judged on absolute
  saturation, where a fixed scale is correct.
- **Small datasets legitimately show less.** With <4 wells no robust z-scores are
  computed; with <3 plates the PCA hides; a well absent from Cells/Nuclei has no
  MFI. These are correct behaviours, not bugs — verify against the source parquet
  before treating an empty result as an error.
- **Removable legacy files.** `config.py` and `report.py` are superseded (config
  migrated into `profiles.py`; report split into `payload/builder.py` +
  `output/html.py`) and can be deleted once a real run is confirmed.
- See `MIGRATION.md` for the refactor history and `FILES.md` for the file
  inventory.