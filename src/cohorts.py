"""Deterministic customer-disjoint temporal partitions, before any model fitting."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from src.features import consumption_features
from src.labels import first_positive
from src.preprocess import monthly_long


def supply_partition(supply: str) -> str:
    bucket=int(hashlib.sha256(str(supply).encode()).hexdigest()[:8],16)%100
    return "train" if bucket<60 else "selection" if bucket<80 else "test"


def make_cohorts(files: dict, seed: int=42) -> tuple[pd.DataFrame,pd.DataFrame]:
    from src.methods import method_family
    events=first_positive(files["HISTORICO_CNR"]).copy()
    known=set(events.SUMINISTRO_ID.astype(str))
    rows=[]
    for year,group in events.groupby(events.event_date.dt.year,sort=True):
        base=group.reset_index(drop=True).copy()
        base["cutoff"]=base.event_date
        feature=consumption_features(base,monthly_long(base,int(year)))
        feature["year"]=int(year)
        feature["event_date"]=base.event_date
        feature["SED_ID"]=base.SED_ID.to_numpy()
        feature["label"]=1
        feature["method_family"]=method_family(base).to_numpy()
        feature["split"]=np.where(base.event_date.lt("2025-01-01"),"train",
                                    np.where(base.event_date.lt("2025-07-01"),"selection","test"))
        rows.append(feature)
    positives=pd.concat(rows,ignore_index=True)
    urows=[]
    for year in (2024,2025):
        data=files[f"ALIMENTADOR_{year}"]
        base=data[~data.SUMINISTRO_ID.astype(str).isin(known)].reset_index(drop=True).copy()
        base["split"]=base.SUMINISTRO_ID.map(supply_partition)
        rng=np.random.default_rng(seed+year)
        base["cutoff"]=pd.NaT
        for split in ("train","selection","test"):
            pool=events[events.event_date.dt.year.eq(year)]
            if year==2025 and split!="train":
                pool=pool[pool.event_date.lt("2025-07-01") if split=="selection" else pool.event_date.ge("2025-07-01")]
            mask=base.split.eq(split)
            base.loc[mask,"cutoff"]=pd.to_datetime(rng.choice(pool.event_date.to_numpy(),size=int(mask.sum())))
        feature=consumption_features(base,monthly_long(base,year))
        feature["year"]=year
        feature["event_date"]=base.cutoff
        feature["SED_ID"]=base.SED_ID.to_numpy()
        feature["label"]=0
        feature["method_family"]="SIN_ETIQUETA"
        feature["split"]=base.split.to_numpy()
        urows.append(feature)
    unlabeled=pd.concat(urows,ignore_index=True)
    # The three latest complete calendar months must be observed for this model.
    p=positives[positives.w3_available.eq(3)].copy()
    u=unlabeled[unlabeled.w3_available.eq(3)].copy()
    evaluation=pd.concat([p,u[(u.year.eq(2024)&u.split.eq("train")) | (u.year.eq(2025)&u.split.ne("train"))]],ignore_index=True)
    assert_disjoint(evaluation)
    # Final refit can use all eligible data, with one U snapshot per customer.
    final_u=u.sort_values(["year","event_date"]).drop_duplicates("SUMINISTRO_ID",keep="last")
    final=pd.concat([p,final_u],ignore_index=True)
    return evaluation,final


def assert_disjoint(frame: pd.DataFrame) -> None:
    sets={split:set(group.SUMINISTRO_ID) for split,group in frame.groupby("split")}
    for first in sets:
        for second in sets:
            if first<second and sets[first]&sets[second]:
                raise ValueError(f"Suministros repetidos entre {first} y {second}")
    if frame.SUMINISTRO_ID.duplicated().any():
        raise ValueError("Cada suministro debe aparecer una sola vez en la evaluación")


def feature_sets(frame: pd.DataFrame) -> dict[str,list[str]]:
    raw=[c for c in frame if c.startswith("w3_") and c!="w3_available"]
    relative=["w3_"+c for c in ("cv","relative_slope","last_ratio","relative_change","drop_relative","zero_fraction","decrease_streak","zero_streak","robust_z")]
    all_windows=[c for c in frame if c.startswith(("w3_","w6_","w12_","months_"))]
    return {"consumo_3_meses":raw,"patron_relativo_3_meses":relative,"multiventana_control":all_windows}
