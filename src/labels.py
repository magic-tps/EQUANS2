from __future__ import annotations

import pandas as pd

from src.utils import normalized


def classify_events(history: pd.DataFrame) -> pd.DataFrame:
    result = history.copy()
    description_col = next(c for c in result if "VULERACI" in normalized(c))
    type_col = next(c for c in result if "TIPIFICACI" in normalized(c))
    date_col = next(c for c in result if "DIA INTERVENCI" in normalized(c))
    result["event_date"] = pd.to_datetime(result[date_col], errors="coerce")
    desc = result[description_col].map(normalized)
    typ = result[type_col].map(normalized)
    valid_type = typ.isin(("MANIPULACION", "CLANDESTINA"))
    explicit = desc.str.contains(
        r"\b(?:CLANDESTINA|MANIPULAD[OA]S?|CONEX(?:ION)? DIRECTA|LINEAS? DIRECTAS?|"
        r"CONEXION ADICIONAL|\bMANIP\b|\bDESCON\b|PUENTES? DE TENSION ABIERTOS|SELLOS VIOLADOS|HURTO|DESCONECTADO|PUENTEADO|"
        r"CONTRAFASE|CAPSULA PERFORADA|INVERSION EN CONEXIONADO|NO CORRESPONDE AL SISTEMA)\b",
        regex=True, na=False)
    result["label_class"] = "AMBIGUOUS"
    result.loc[valid_type & explicit & result.event_date.notna(), "label_class"] = "POSITIVE_CONFIRMED"
    result.loc[result.event_date.isna() | typ.eq("SUMINISTROS RETIRADOS"), "label_class"] = "INVALID"
    return result


def first_positive(history: pd.DataFrame) -> pd.DataFrame:
    labeled = classify_events(history)
    labeled = labeled[labeled.label_class.eq("POSITIVE_CONFIRMED")]
    return labeled.sort_values("event_date").drop_duplicates("SUMINISTRO_ID", keep="first")
