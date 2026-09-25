from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocess import monthly_long


def _window(values: np.ndarray, n: int, prefix: str) -> dict[str, float]:
    x = values[-n:]
    good = x[np.isfinite(x)]
    out = {f"{prefix}_available": len(good), f"{prefix}_zeros": int(np.sum(good == 0))}
    for key in ("mean", "median", "minimum", "maximum", "std", "cv", "slope", "last", "drop", "drop_relative", "zero_streak", "decrease_streak", "level_change", "robust_z"):
        out[f"{prefix}_{key}"] = np.nan
    if len(good) == 0:
        return out
    median = float(np.median(good))
    std = float(np.std(good))
    out.update({f"{prefix}_mean": float(np.mean(good)), f"{prefix}_median": median,
                f"{prefix}_minimum": float(np.min(good)), f"{prefix}_maximum": float(np.max(good)),
                f"{prefix}_std": std, f"{prefix}_cv": std / (np.mean(good)+1e-6),
                f"{prefix}_last": float(good[-1]),
                f"{prefix}_robust_z": float((good[-1]-median)/(1.4826*np.median(np.abs(good-median))+1e-6))})
    if len(good) > 1:
        dif = np.diff(good)
        out[f"{prefix}_slope"] = float(np.polyfit(np.arange(len(good)),good,1)[0])
        out[f"{prefix}_drop"] = float(np.min(dif))
        out[f"{prefix}_drop_relative"] = float(np.min(dif/(good[:-1]+1e-6)))
        out[f"{prefix}_level_change"] = float(np.mean(good[-max(1,len(good)//2):])-np.mean(good[:max(1,len(good)//2)]))
        streak = 0
        for d in dif[::-1]:
            if d < 0: streak += 1
            else: break
        out[f"{prefix}_decrease_streak"] = streak
    streak = 0
    for value in good[::-1]:
        if value == 0: streak += 1
        else: break
    out[f"{prefix}_zero_streak"] = streak
    return out


def consumption_features(base: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Each row's own cutoff controls its observations, including duplicate IDs at different cuts."""
    series = {key: group.sort_values("nominal_month") for key,group in monthly.groupby("SUMINISTRO_ID",sort=False)}
    records = []
    for row in base.itertuples(index=False):
        ident = getattr(row, "SUMINISTRO_ID")
        cutoff = pd.Timestamp(getattr(row, "cutoff"))
        m = series.get(ident)
        if m is None:
            x = np.array([], dtype=float)
        else:
            available = m[m.nominal_month.lt(cutoff) & m.reading_date.lt(cutoff)]
            x = available.daily_kwh.to_numpy(dtype=float)[-12:]
        record = {"SUMINISTRO_ID":ident, "months_observed":int(np.isfinite(x).sum()),
                  "months_missing_or_invalid":int(np.isnan(x).sum())}
        for n in (3,6,12):
            record.update(_window(x,n,f"w{n}"))
        records.append(record)
    return pd.DataFrame(records, index=base.index)


def _asof_sed(base: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    b = balance.copy()
    b["period"] = pd.to_datetime(b["PERIODO BALANCE"],errors="coerce")
    b = b.dropna(subset=["SED_ID","period"])
    cols = {"E. DISTRIBUIDA MES MWh":"sed_distributed_mwh",
            "E. FACTURADA MES MWh":"sed_billed_mwh",
            "PERDIDA  MES MWh":"sed_loss_mwh",
            "PERDIDA MES %":"sed_loss_fraction",
            "PROYEC_PERDIDAS NO TECNICAS MES MWh":"sed_projected_nontech_mwh",
            "CANT CLIENTES TOTALES APROX":"sed_clients"}
    b = b[["SED_ID","period",*cols]].rename(columns=cols).sort_values("period")
    left = base[["SED_ID","cutoff"]].copy()
    left["_idx"] = np.arange(len(left))
    left = left.sort_values("cutoff")
    result = pd.merge_asof(left,b,left_on="cutoff",right_on="period",by="SED_ID",direction="backward",allow_exact_matches=False)
    return result.sort_values("_idx")[[*cols.values()]].set_axis(base.index)


def _annual_sed(base: pd.DataFrame, data: pd.DataFrame, year_col: str, value_cols: dict[str,str]) -> pd.DataFrame:
    frame = data.copy()
    frame["year"] = pd.to_numeric(frame[year_col],errors="coerce")
    frame = frame.dropna(subset=["SED_ID","year"])
    # Annual rows cannot be known in full until the following year.
    frame["available"] = pd.to_datetime((frame.year.astype(int)+1).astype(str)+"-01-01")
    for col in value_cols:
        frame[col] = pd.to_numeric(frame[col],errors="coerce")
    frame = frame.groupby(["SED_ID","available"],as_index=False)[list(value_cols)].mean()
    frame = frame.rename(columns=value_cols).sort_values("available")
    left = base[["SED_ID","cutoff"]].copy()
    left["_idx"] = np.arange(len(left))
    merged = pd.merge_asof(left.sort_values("cutoff"),frame,left_on="cutoff",right_on="available",by="SED_ID",direction="backward",allow_exact_matches=True)
    return merged.sort_values("_idx")[list(value_cols.values())].set_axis(base.index)


def _peer_features(feature: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out mean; avoids using the customer's own value as its reference."""
    x = feature["w6_median"]
    group = base["SED_ID"].fillna("UNKNOWN")
    count = x.groupby(group).transform("count")
    total = x.groupby(group).transform("sum")
    peer_mean = ((total-x)/(count-1)).where(count.gt(1))
    return pd.DataFrame({"peer_count":count-1,"peer_ratio":x/(peer_mean+1e-6),
                         "peer_difference":x-peer_mean},index=base.index)


def build_features(base: pd.DataFrame, monthly: pd.DataFrame, sed_files: dict[str,pd.DataFrame]|None=None) -> pd.DataFrame:
    base = base.reset_index(drop=True).copy()
    base = base.rename(columns={"TIPO ACOMETIDA":"ACOMETIDA AEREA/SUBTERRANEA",
                                "POTENCIA CONTRATADA":"POTENCIA USUARIO CONTRATADO"})
    base["cutoff"] = pd.to_datetime(base.cutoff)
    f = consumption_features(base,monthly)
    for col in ("TARIFA","ACOMETIDA AEREA/SUBTERRANEA","TIPO CONEXIONADO"):
        if col in base:
            f["category_"+col] = base[col].fillna("UNKNOWN").astype(str).to_numpy()
    if "POTENCIA USUARIO CONTRATADO" in base:
        f["contracted_power"] = pd.to_numeric(base["POTENCIA USUARIO CONTRATADO"],errors="coerce").to_numpy()
    f = pd.concat([f,_peer_features(f,base)],axis=1)
    if sed_files:
        if "BALANCE_SED" in sed_files:
            f = pd.concat([f,_asof_sed(base,sed_files["BALANCE_SED"])],axis=1)
        if "POTENCIA_SED" in sed_files:
            d=sed_files["POTENCIA_SED"]
            f=pd.concat([f,_annual_sed(base,d,d.columns[-1],{"POTENCIA [KVA]":"sed_power_kva","Clientes BT Cantidad Referencial":"sed_clients_reference"})],axis=1)
        if "FALLAS_SED" in sed_files:
            d=sed_files["FALLAS_SED"]
            f=pd.concat([f,_annual_sed(base,d,d.columns[-1],{"Cant_Fallas":"sed_failures"})],axis=1)
    return f.drop(columns=["SUMINISTRO_ID"])
