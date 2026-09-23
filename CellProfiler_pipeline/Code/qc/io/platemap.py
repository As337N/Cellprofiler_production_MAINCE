"""
qc/io/platemap.py — Carga del platemap (por placa).

Cada placa (96 u 384) tiene su propio platemap independiente de N entradas. La
geometría (8×12 o 16×24) viene del perfil (WELLS_PER_PLATE del .env) y sirve
solo para validar: si un platemap trae más pozos de los que la placa admite,
casi seguro el .env tiene el WELLS_PER_PLATE equivocado para esa cohorte.

Firma dict {plate: {well: compound}} conservada (Fase 0). El cambio a DataFrame
long para join directo es Fase 3.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import polars as pl

from qc.profiles import DatasetProfile


def load_platemap(platemap_path, profile: DatasetProfile | None = None) -> dict:
    """Load platemap CSV -> {plate: {well: compound}}.

    Si se pasa `profile`, valida que ninguna placa exceda wells_per_plate y
    avisa (no aborta) ante desajustes de geometría.
    """
    if platemap_path is None:
        return {}
    platemap_path = Path(platemap_path)
    if not platemap_path.exists():
        print(f"[warn] Platemap not found: {platemap_path}")
        return {}

    df           = pl.read_csv(platemap_path)
    well_col     = next((c for c in df.columns if "Well" in c), None)
    compound_col = next((c for c in df.columns if "Compound" in c
                         or "Perturbation" in c), None)
    plate_col    = next((c for c in df.columns if "Plate" in c), None)
    if not well_col or not compound_col:
        print(f"[warn] Platemap missing Well/Compound column. Found: {list(df.columns)}")
        return {}

    result = defaultdict(dict)
    for row in df.iter_rows(named=True):
        well  = str(row[well_col]).strip().upper()
        cmpd  = str(row[compound_col]).strip()
        raw_p = str(row[plate_col]).strip() if plate_col else "Plate"
        plate = raw_p if raw_p.startswith("P") else f"P{raw_p}"
        result[plate][well] = cmpd

    n = sum(len(v) for v in result.values())
    print(f"[platemap] {n} well->compound mappings.")

    # Validación de geometría contra el perfil (WELLS_PER_PLATE del .env).
    if profile is not None:
        cap = profile.wells_per_plate
        for plate, wells in result.items():
            if len(wells) > cap:
                print(f"[warn] Platemap '{plate}' tiene {len(wells)} pozos > "
                      f"WELLS_PER_PLATE={cap}. ¿.env con geometría equivocada?")

    return dict(result)
