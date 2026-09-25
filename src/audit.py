from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.labels import classify_events
from src.utils import DATA_FILES, ensure_dirs, normalized


def audit(root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    ensure_dirs(root)
    missing = [name for name in DATA_FILES if not (root / "data" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos de origen: {missing}")
    frames = {}
    summaries, dictionary, hashes = [], [], {}
    for name in DATA_FILES:
        path = root / "data" / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        book = pd.ExcelFile(path)
        if len(book.sheet_names) != 1:
            raise ValueError(f"{name}: se esperaba exactamente una hoja; hay {book.sheet_names}")
        data = pd.read_excel(book, sheet_name=book.sheet_names[0])
        frames[path.stem] = data
        summaries.append(dict(file=name, sheet=book.sheet_names[0], rows=len(data), columns=len(data.columns),
                              duplicate_rows=int(data.duplicated().sum()),
                              unique_supply=int(data.SUMINISTRO_ID.nunique()) if "SUMINISTRO_ID" in data else None,
                              unique_sed=int(data.SED_ID.nunique()) if "SED_ID" in data else None))
        for col in data:
            s = data[col]
            converted = pd.to_numeric(s, errors="coerce") if not pd.api.types.is_datetime64_any_dtype(s) else None
            num = converted if converted is not None and converted.notna().sum() >= .8*max(1,s.notna().sum()) else None
            if num is not None and num.notna().any():
                minimum,maximum=str(num.min()),str(num.max())
            else:
                minimum=str(s.dropna().astype(str).min()) if s.notna().any() else ""
                maximum=str(s.dropna().astype(str).max()) if s.notna().any() else ""
            dictionary.append(dict(file=name, column=col, dtype=str(s.dtype), nulls=int(s.isna().sum()),
                                   unique=int(s.nunique(dropna=True)), negative=int(num.lt(0).sum()) if num is not None else None,
                                   zero=int(num.eq(0).sum()) if num is not None else None,
                                   invalid_numeric=int((s.notna() & num.isna()).sum()) if num is not None else None,
                                   min=minimum,max=maximum))
    private = root / "reports/private"
    private.mkdir(exist_ok=True)
    (private / "source_sha256.txt").write_text("\n".join(f"{k} {v}" for k,v in hashes.items())+"\n",encoding="utf-8")
    pd.DataFrame(summaries).to_csv(root / "reports/file_summary.csv", index=False)
    pd.DataFrame(dictionary).to_csv(root / "reports/data_dictionary.csv", index=False)
    a, b, h = (frames[x] for x in ("ALIMENTADOR_2024", "ALIMENTADOR_2025", "HISTORICO_CNR"))
    ids24, ids25 = set(a.SUMINISTRO_ID), set(b.SUMINISTRO_ID)
    shared = ids24 & ids25
    old = a.set_index("SUMINISTRO_ID").loc[list(shared)]
    new = b.set_index("SUMINISTRO_ID").loc[list(shared)]
    lines = ["# Auditoría de datos", "", "Origen: los siete Excel locales. Ninguna fila individual se incluye en este reporte.", "",
             "| Archivo | Filas | Columnas | Duplicados de fila |", "|---|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['file']} | {s['rows']} | {s['columns']} | {s['duplicate_rows']} |")
    lines += ["", f"Suministros compartidos 2024–2025: {len(shared)}; nuevos en 2025: {len(ids25-ids24)}; ausentes en 2025: {len(ids24-ids25)}.",
              f"Cambio de SED en compartidos: {int(old.SED_ID.ne(new.SED_ID).sum())}; cambio de alimentador: {int(old.ALIMENTADOR_ID.ne(new.ALIMENTADOR_ID).sum())}.",
              f"Solapamiento entre HISTORICO_CNR y ALIMENTADOR_2025: {len(set(h.SUMINISTRO_ID)&ids25)}.", ""]
    for name in ("BALANCE_SED", "POTENCIA_SED", "FALLAS_SED", "CALIDAD_TENSION"):
        coverage = b.SED_ID.isin(frames[name].SED_ID).mean()
        lines.append(f"Cobertura de SED en {name}: {coverage:.1%} de suministros 2025.")
    lines += ["", "Las fechas de lectura y el periodo nominal no siempre coinciden. Se exige que ambos precedan el corte temporal.",
              "Los días facturados válidos se limitan a 1–45; los valores inválidos quedan como faltantes, nunca cero.",
              "Las pérdidas de SED son contexto de red; no constituyen una etiqueta individual.",
              "Los detalles de columnas, nulos, extremos y ceros están en data_dictionary.csv (local, ignorado por Git).", ""]
    (root / "reports/data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    labeled = classify_events(h)
    desc = next(c for c in h if "VULERACI" in normalized(c))
    typ = next(c for c in h if "TIPIFICACI" in normalized(c))
    labeled.groupby([desc, typ, "label_class"], dropna=False).size().rename("frequency").reset_index().to_csv(
        private / "label_catalog.csv", index=False)
    counts = labeled.label_class.value_counts().to_dict()
    (root / "reports/labels_analysis.md").write_text(
        "# Etiquetas observadas\n\n" + "\n".join(f"- {k}: {v}" for k,v in counts.items()) +
        "\n\nLa clase positiva requiere tipificación MANIPULACIÓN o CLANDESTINA, descripción explícita y fecha válida. "
        "Los casos ambiguos no se tratan como negativos. Se conserva sólo la primera intervención positiva por suministro. "
        "El catálogo de textos exactos está en reports/private/label_catalog.csv y se excluye de Git.\n", encoding="utf-8")
    return frames
