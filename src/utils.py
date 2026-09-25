from __future__ import annotations

import re
import unicodedata
from pathlib import Path

DATA_FILES = (
    "ALIMENTADOR_2024.xlsx", "ALIMENTADOR_2025.xlsx", "BALANCE_SED.xlsx",
    "CALIDAD_TENSION.xlsx", "FALLAS_SED.xlsx", "HISTORICO_CNR.xlsx",
    "POTENCIA_SED.xlsx",
)
MONTHS = ("ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", "JULIO",
          "AGOSTO", "SETIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE")


def normalized(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).upper())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def ensure_dirs(root: Path) -> None:
    for name in ("reports", "outputs", "models"):
        (root / name).mkdir(parents=True, exist_ok=True)
