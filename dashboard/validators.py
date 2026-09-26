from __future__ import annotations

import pandas as pd

from src.preprocess import find_month_columns


def validate(frame: pd.DataFrame,allow_insufficient: bool=False) -> tuple[list[str],dict]:
    errors=[]
    warnings=[]
    if frame.empty: errors.append("El archivo está vacío")
    if "SUMINISTRO_ID" not in frame:
        errors.append("Falta SUMINISTRO_ID")
    else:
        if frame.SUMINISTRO_ID.isna().any():errors.append("Hay SUMINISTRO_ID vacíos")
        ids=frame.SUMINISTRO_ID.astype(str).str.strip()
        if ids.eq("").any():errors.append("Hay SUMINISTRO_ID vacíos")
        if ids.duplicated().any():errors.append("Hay SUMINISTRO_ID duplicados")
    months=find_month_columns(frame)
    complete=[]
    for n,fields in months.items():
        missing={"consumption","days","reading"}-set(fields)
        if missing:
            from src.utils import MONTHS
            labels={"consumption":"CONSUMO","days":"DIAS FACTURADO","reading":"FECHA LECTURA"}
            errors.append("Faltan: "+", ".join(f"{labels[k]} {MONTHS[n-1]}" for k in sorted(missing)))
        else:
            complete.append(n)
            consumption=pd.to_numeric(frame[fields["consumption"]],errors="coerce")
            days=pd.to_numeric(frame[fields["days"]],errors="coerce")
            reading=pd.to_datetime(frame[fields["reading"]],errors="coerce")
            invalid=int((consumption.lt(0)|(~days.between(1,45)&days.notna())|(reading.isna()&frame[fields["reading"]].notna())).sum())
            if invalid:warnings.append(f"Mes {n}: {invalid} filas con consumo negativo, días fuera de 1–45 o fecha inválida")
    if len(complete)<3:
        message="Se requieren al menos tres meses con consumo, días facturados y fecha de lectura para el score; los demás casos se derivan a completar datos"
        (warnings if allow_insufficient else errors).append(message)
    return errors,{"months":complete,"rows":len(frame),"columns":len(frame.columns),"warnings":warnings,
                   "supplies":frame.SUMINISTRO_ID.nunique() if "SUMINISTRO_ID" in frame else 0}
