from __future__ import annotations

from io import BytesIO

import pandas as pd

from dashboard.validators import validate
from src.features import build_features
from src.predict import rank,score_features
from src.preprocess import monthly_long


def read_upload(name: str, content: bytes) -> pd.DataFrame:
    if not content:
        raise ValueError("El archivo está vacío")
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(BytesIO(content))
    if name.lower().endswith(".csv"):
        return pd.read_csv(BytesIO(content))
    raise ValueError("Formato no compatible: use .xlsx o .csv")


def infer(bundle: dict, frame: pd.DataFrame, year: int, cutoff=None, progress=None) -> pd.DataFrame:
    errors,_=validate(frame)
    if errors:raise ValueError("No se puede ejecutar Volt Patrol. " + "; ".join(errors))
    base=frame.copy().reset_index(drop=True)
    base["SUMINISTRO_ID"]=base.SUMINISTRO_ID.astype(str).str.strip()
    monthly=monthly_long(base,year)
    if cutoff is None:
        available=monthly.loc[monthly.daily_kwh.notna(),"nominal_month"]
        cutoff=available.max()+pd.offsets.MonthBegin(1) if not available.empty else pd.Timestamp(year+1,1,1)
    base["cutoff"]=pd.Timestamp(cutoff)
    if progress:progress(.35,"Lecturas validadas · preparando ventanas temporales")
    features=build_features(base,monthly)
    if progress:progress(.65,"Características listas · ejecutando el modelo entrenado")
    scores=score_features(bundle,features)
    ranked=rank(base,scores,features.months_observed,
                recent_months=features.w3_available if bundle.get("pipeline_version",1)>=2 else None)
    if bundle.get("pipeline_version",1)>=2:
        from src.evidence import enrich_ranking
        ranked=enrich_ranking(ranked,base,features,bundle)
    if progress:progress(1.,"Ranking listo")
    return ranked


def export_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result=frame.copy()
    for column in result.select_dtypes(include=["object","str"]):
        result[column]=result[column].map(lambda value: "'"+value if isinstance(value,str) and value.lstrip().startswith(("=","+","-","@")) else value)
    return result


def to_csv(frame: pd.DataFrame) -> bytes:
    return export_safe(frame).to_csv(index=False).encode("utf-8-sig")


def to_xlsx(frame: pd.DataFrame) -> bytes:
    buffer=BytesIO()
    with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
        export_safe(frame).to_excel(writer,index=False)
    return buffer.getvalue()
