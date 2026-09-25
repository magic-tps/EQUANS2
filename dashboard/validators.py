from __future__ import annotations

import pandas as pd

from src.preprocess import find_month_columns


def validate(frame: pd.DataFrame) -> tuple[list[str],dict]:
    errors=[]
    warnings=[]
    if frame.empty: errors.append("El archivo está vacío")
    if "SUMINISTRO_ID" not in frame:
        errors.append("Falta SUMINISTRO_ID")
    else:
        if frame.SUMINISTRO_ID.isna().any():errors.append("Hay SUMINISTRO_ID vacíos")
        if frame.SUMINISTRO_ID.duplicated().any():errors.append("Hay SUMINISTRO_ID duplicados")
    months=find_month_columns(frame)
    complete=[]
    for n,fields in months.items():
        missing={"consumption","days","reading"}-set(fields)
        if missing:errors.append(f"Mes {n}: faltan {', '.join(sorted(missing))}")
        else:
            complete.append(n)
            consumption=pd.to_numeric(frame[fields["consumption"]],errors="coerce")
            days=pd.to_numeric(frame[fields["days"]],errors="coerce")
            reading=pd.to_datetime(frame[fields["reading"]],errors="coerce")
            invalid=int((consumption.lt(0)|(~days.between(1,45)&days.notna())|(reading.isna()&frame[fields["reading"]].notna())).sum())
            if invalid:warnings.append(f"Mes {n}: {invalid} filas con consumo negativo, días fuera de 1–45 o fecha inválida")
    if len(complete)<3:errors.append("Se requieren al menos tres meses con consumo, días facturados y fecha de lectura")
    return errors,{"months":complete,"rows":len(frame),"columns":len(frame.columns),"warnings":warnings,
                   "supplies":frame.SUMINISTRO_ID.nunique() if "SUMINISTRO_ID" in frame else 0}
