from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocess import monthly_long


def _window(values: np.ndarray, n: int, prefix: str) -> dict[str, float]:
    x = values[-n:]
    good = x[np.isfinite(x)]
    out = {f"{prefix}_available": len(good), f"{prefix}_zeros": int(np.sum(good == 0))}
    for key in ("mean", "median", "minimum", "maximum", "std", "cv", "slope", "last", "drop", "drop_relative", "zero_streak", "decrease_streak", "level_change", "robust_z", "relative_slope", "last_ratio", "relative_change", "zero_fraction"):
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
    out[f"{prefix}_last"] = float(x[-1])
    if len(good) > 1:
        # Missing calendar months must not become adjacent measurements.
        dif = np.diff(x)
        adjacent = np.isfinite(dif)
        out[f"{prefix}_slope"] = float(np.polyfit(np.flatnonzero(np.isfinite(x)),good,1)[0])
        if adjacent.any():
            out[f"{prefix}_drop"] = float(np.min(dif[adjacent]))
            out[f"{prefix}_drop_relative"] = float(np.min((dif/(np.abs(x[:-1])+1))[adjacent]))
        midpoint=max(1,len(x)//2)
        if np.isfinite(x[:midpoint]).any() and np.isfinite(x[midpoint:]).any():
            out[f"{prefix}_level_change"] = float(np.nanmean(x[midpoint:])-np.nanmean(x[:midpoint]))
        streak = 0
        for d in dif[::-1]:
            if d < 0: streak += 1
            else: break
        out[f"{prefix}_decrease_streak"] = streak
    streak = 0
    for value in x[::-1]:
        if value == 0: streak += 1
        else: break
    out[f"{prefix}_zero_streak"] = streak
    out[f"{prefix}_relative_slope"] = out[f"{prefix}_slope"]/(median+1)
    out[f"{prefix}_last_ratio"] = out[f"{prefix}_last"]/(median+1)
    out[f"{prefix}_relative_change"] = out[f"{prefix}_level_change"]/(median+1)
    out[f"{prefix}_zero_fraction"] = float(np.mean(good == 0))
    out[f"{prefix}_robust_z"] = float(np.clip(out[f"{prefix}_robust_z"],-30,30))
    return out


def _consumption_features_reference(base: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Each row's own cutoff controls its observations, including duplicate IDs at different cuts."""
    series = {key: group.sort_values("nominal_month") for key,group in monthly.groupby("SUMINISTRO_ID",sort=False)}
    records = []
    for row in base.itertuples(index=False):
        ident = getattr(row, "SUMINISTRO_ID")
        cutoff = pd.Timestamp(getattr(row, "cutoff"))
        m = series.get(ident)
        end = cutoff.to_period("M").start_time
        grid = pd.date_range(end=end-pd.offsets.MonthBegin(1),periods=12,freq="MS")
        x = np.full(12,np.nan)
        if m is not None:
            available = m[m.nominal_month.lt(end) & m.reading_date.lt(cutoff)]
            if available.nominal_month.duplicated().any():
                raise ValueError("Hay lecturas duplicadas del mismo suministro y mes")
            x = available.set_index("nominal_month").daily_kwh.reindex(grid).to_numpy(dtype=float)
        record = {"SUMINISTRO_ID":ident, "months_observed":int(np.isfinite(x).sum()),
                  "months_missing_or_invalid":int(np.isnan(x).sum())}
        for n in (3,6,12):
            record.update(_window(x,n,f"w{n}"))
        recent=x[-3:]; previous=x[-6:-3]
        record["recent_baseline_ratio"] = float(np.nanmean(recent)/(np.nanmean(previous)+1)) if np.isfinite(recent).any() and np.isfinite(previous).any() else np.nan
        records.append(record)
    return pd.DataFrame(records, index=base.index)


def _matrix_window(values: np.ndarray, n: int, prefix: str) -> dict:
    """Same definitions as _window, evaluated over a batch of supplies."""
    import warnings
    x=values[:,-n:];valid=np.isfinite(x);count=valid.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        mean=np.nanmean(x,axis=1);median=np.nanmedian(x,axis=1);std=np.nanstd(x,axis=1)
        last_position=np.where(valid,np.arange(n),-1).max(axis=1)
        last_good=x[np.arange(len(x)),np.maximum(last_position,0)]
        mad=np.nanmedian(np.abs(x-median[:,None]),axis=1)
        position=np.arange(n,dtype=float)[None,:]
        xm=np.sum(np.where(valid,position,0),axis=1)/np.maximum(count,1)
        numerator=np.nansum((position-xm[:,None])*(x-mean[:,None]),axis=1)
        denominator=np.sum(np.where(valid,(position-xm[:,None])**2,0),axis=1)
        slope=np.divide(numerator,denominator,out=np.full(len(x),np.nan),where=count>1)
        # Preserve legacy floating-point signs near zero: existing trees can have
        # split borders there. Only these exceptional rows need the reference fit.
        near_zero=(count>1)&(np.abs(slope)<1e-12*np.maximum(1,np.abs(mean)))
        for i in np.flatnonzero(near_zero):
            slope[i]=np.polyfit(np.flatnonzero(valid[i]),x[i,valid[i]],1)[0]
        dif=np.diff(x,axis=1)
        level=np.nanmean(x[:,max(1,n//2):],axis=1)-np.nanmean(x[:,:max(1,n//2)],axis=1)
        result={"available":count,"zeros":np.sum(x==0,axis=1),"mean":mean,"median":median,
                "minimum":np.nanmin(x,axis=1),"maximum":np.nanmax(x,axis=1),"std":std,
                "cv":std/(mean+1e-6),"last":x[:,-1],"slope":slope,
                "drop":np.nanmin(dif,axis=1),"drop_relative":np.nanmin(dif/(np.abs(x[:,:-1])+1),axis=1),
                "zero_streak":np.cumprod(x[:,::-1]==0,axis=1).sum(axis=1).astype(float),
                "decrease_streak":np.cumprod(dif[:,::-1]<0,axis=1).sum(axis=1).astype(float),
                "level_change":level,"robust_z":np.clip((last_good-median)/(1.4826*mad+1e-6),-30,30),
                "relative_slope":slope/(median+1),"last_ratio":x[:,-1]/(median+1),
                "relative_change":level/(median+1),"zero_fraction":np.sum(x==0,axis=1)/np.maximum(count,1)}
        result["decrease_streak"][count<2]=np.nan
        for name,value in result.items():
            if name not in ("available","zeros"):value[count==0]=np.nan
    return {f"{prefix}_{name}":value for name,value in result.items()}


def consumption_features(base: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Vectorized calendar lookup. Cuts, gaps and invalid readings retain their meaning."""
    if base.empty:return pd.DataFrame(index=base.index)
    cuts=pd.to_datetime(base["cutoff"])
    if cuts.isna().any():raise ValueError("Hay fechas de corte inválidas")
    months=cuts.dt.year.to_numpy()*12+cuts.dt.month.to_numpy()-1
    grid=months[:,None]+np.arange(-12,0)[None,:]
    lookup=monthly.copy()
    lookup["_month"]=lookup.nominal_month.dt.year*12+lookup.nominal_month.dt.month-1
    if lookup.duplicated(["SUMINISTRO_ID","_month"]).any():
        raise ValueError("Hay lecturas duplicadas del mismo suministro y mes")
    index=pd.MultiIndex.from_arrays([np.repeat(base.SUMINISTRO_ID.astype(str).to_numpy(),12),grid.ravel()],names=["SUMINISTRO_ID","_month"])
    selected=lookup.set_index(["SUMINISTRO_ID","_month"])[["daily_kwh","reading_date"]].reindex(index)
    eligible=selected.reading_date.to_numpy(dtype="datetime64[ns]")<np.repeat(cuts.to_numpy(dtype="datetime64[ns]"),12)
    x=np.where(eligible,selected.daily_kwh.to_numpy(dtype=float),np.nan).reshape(-1,12)
    records={"SUMINISTRO_ID":base.SUMINISTRO_ID.to_numpy(),"months_observed":np.isfinite(x).sum(axis=1),
             "months_missing_or_invalid":np.isnan(x).sum(axis=1)}
    for n in (3,6,12):records.update(_matrix_window(x,n,f"w{n}"))
    recent=x[:,-3:];previous=x[:,-6:-3]
    recent_count=np.isfinite(recent).sum(axis=1);previous_count=np.isfinite(previous).sum(axis=1)
    ratio=(np.nansum(recent,axis=1)/np.maximum(recent_count,1))/(np.nansum(previous,axis=1)/np.maximum(previous_count,1)+1)
    records["recent_baseline_ratio"]=np.where((recent_count>0)&(previous_count>0),ratio,np.nan)
    return pd.DataFrame(records,index=base.index)


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
    if "SED_ID" not in base:
        return pd.DataFrame(index=base.index)
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
