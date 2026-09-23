"""
qc/cli.py — Interfaz de línea de comandos. Parsea args y llama al pipeline.

El dataset (canales, regex, geometría, cohort) NO se pasa por CLI: viene del
.env vía profiles.profile_from_env(), que es como el orquestador ya opera.
Cambiar de interno a JUMP = descomentar el bloque correspondiente del .env,
sin tocar la invocación ni el código.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from qc.pipeline import PipelineConfig, QCPipeline
from qc.profiles import profile_from_env


def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(
        description="Cell Painting QC collage + HTML/PDF report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, type=Path)
    p.add_argument("-o", "--output", required=True, type=Path)
    p.add_argument("--qc", default=None, type=Path)
    p.add_argument("--platemap", default=None, type=Path)
    p.add_argument("--sites", type=int, default=9)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--band-height", type=int, default=280)
    p.add_argument("--font", type=int, default=18)
    p.add_argument("--n-sigma", type=float, default=3.0)
    p.add_argument("--web-scale", type=float, default=0.2)
    p.add_argument("--no-pdf", action="store_true", help="Omite el PDF resumen.")
    a = p.parse_args()

    # Dataset (canales/regex/geometría/cohort) desde el .env.
    profile = profile_from_env()

    return PipelineConfig(
        input_dir=a.input, output_path=a.output, profile=profile,
        qc_tsv=a.qc, platemap=a.platemap, sites_per_well=a.sites,
        scale=a.scale, workers=a.workers, band_height=a.band_height,
        font_size=a.font, n_sigma=a.n_sigma, web_scale=a.web_scale,
        emit_pdf=not a.no_pdf,
    )


def main() -> None:
    QCPipeline(parse_args()).run()


if __name__ == "__main__":
    main()
