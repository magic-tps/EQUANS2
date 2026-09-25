from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils import MONTHS, normalized


def find_month_columns(frame: pd.DataFrame) -> dict[int, dict[str, str]]:
    names = {normalized(c): c for c in frame.columns}
    result = {}
    for month_num, month in enumerate(MONTHS, 1):
        fields = {}
        for kind, pattern in (("consumption", "CONSUMO "),
                              ("days", "DIAS FACTURADO "),
                              ("reading", "FECHA LECTURA ")):
            key = pattern + month
            if kind == "reading" and key not in names:
                key = "FECHA TOMA LECTURA " + month
            if key in names:
                fields[kind] = names[key]
        if fields:
            result[month_num] = fields
    return result


def monthly_long(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """Preserve missing and invalid observations; no interpolation or zero filling."""
    parts = []
    for month, fields in find_month_columns(frame).items():
        if not all(k in fields for k in ("consumption", "days", "reading")):
            continue
        part = pd.DataFrame({
            "SUMINISTRO_ID": frame["SUMINISTRO_ID"].astype(str),
            "nominal_month": pd.Timestamp(year=year, month=month, day=1),
            "reading_date": pd.to_datetime(frame[fields["reading"]], errors="coerce"),
            "consumption_kwh": pd.to_numeric(frame[fields["consumption"]], errors="coerce"),
            "days_billed": pd.to_numeric(frame[fields["days"]], errors="coerce"),
        })
        plausible_date = part.reading_date.between(part.nominal_month-pd.Timedelta(days=65),
                                                   part.nominal_month+pd.Timedelta(days=65))
        good = part.consumption_kwh.ge(0) & part.days_billed.between(1, 45) & plausible_date
        part["daily_kwh"] = (part.consumption_kwh / part.days_billed).where(good)
        part["observation_status"] = np.select(
            [part.consumption_kwh.isna() | part.days_billed.isna() | part.reading_date.isna(),
             ~plausible_date,
             part.consumption_kwh.lt(0) | ~part.days_billed.between(1, 45),
             part.consumption_kwh.eq(0)],
            ["missing", "inconsistent_reading", "invalid", "observed_zero"], default="valid")
        parts.append(part)
    if not parts:
        raise ValueError("No se detectaron meses completos de consumo, días y lectura")
    return pd.concat(parts, ignore_index=True)


def before_cutoff(monthly: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    return monthly.loc[monthly.reading_date.lt(cutoff) & monthly.nominal_month.lt(cutoff)].copy()
