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


def infer(bundle: dict, frame: pd.DataFrame, year: int) -> pd.DataFrame:
    errors,_=validate(frame)
    if errors:raise ValueError("No se puede ejecutar Volt Patrol. " + "; ".join(errors))
    base=frame.copy()
    base["cutoff"]=pd.Timestamp(year+1,1,1)
    features=build_features(base,monthly_long(base,year))
    scores=score_features(bundle,features)
    return rank(base,scores,features.months_observed)


def to_xlsx(frame: pd.DataFrame) -> bytes:
    buffer=BytesIO()
    with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
        frame.to_excel(writer,index=False)
    return buffer.getvalue()
