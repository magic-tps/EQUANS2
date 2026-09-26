"""Descriptive drift and observed outcomes. No automatic training or risk calibration."""
from __future__ import annotations
import numpy as np
import pandas as pd


def drift(reference,current):
    records=[]
    for col in ("priority_score","recent_daily_kwh","recent_change_pct","model_spread"):
        if col not in reference or col not in current:continue
        first=pd.to_numeric(reference[col],errors="coerce").dropna().to_numpy()
        second=pd.to_numeric(current[col],errors="coerce").dropna().to_numpy()
        if len(first)<20 or len(second)<20:continue
        bins=np.unique(np.quantile(first,np.linspace(0,1,11)))
        if len(bins)<3:continue
        bins[0]=-np.inf;bins[-1]=np.inf
        a=(np.histogram(first,bins)[0]+.5)/(len(first)+.5*(len(bins)-1))
        b=(np.histogram(second,bins)[0]+.5)/(len(second)+.5*(len(bins)-1))
        psi=float(np.sum((b-a)*np.log(b/a)))
        records.append({"variable":col,"psi":psi,"reference_n":len(first),"current_n":len(second),
                        "action":"Revisar distribución" if psi>=.2 else "Sin alerta por este umbral"})
    return pd.DataFrame(records)


def evaluate_top(ranking,truth,k=76):
    if not {"SUMINISTRO_ID","HURTO_REAL"}<=set(truth):raise ValueError("Faltan SUMINISTRO_ID y/o HURTO_REAL")
    truth=truth.copy();truth["SUMINISTRO_ID"]=truth.SUMINISTRO_ID.astype(str).str.strip()
    if truth.SUMINISTRO_ID.duplicated().any():raise ValueError("Etiquetas duplicadas por suministro")
    labels=pd.to_numeric(truth.HURTO_REAL,errors="coerce")
    if labels.isna().any() or not labels.isin([0,1]).all():raise ValueError("HURTO_REAL debe ser 0 o 1; no convierte desconocidos en negativos")
    truth["HURTO_REAL"]=labels.astype(int)
    top=ranking.head(k)[["SUMINISTRO_ID"]].merge(truth,on="SUMINISTRO_ID",how="left",validate="one_to_one")
    resolved=int(top.HURTO_REAL.notna().sum());confirmed=int(top.HURTO_REAL.sum())
    return {"top_k":k,"listed":len(top),"resolved":resolved,"confirmed":confirmed,
            "precision_at_k":confirmed/k if len(top)==k and resolved==k else None,
            "confirmation_among_resolved":confirmed/resolved if resolved else None,
            "coverage":resolved/k,"labels_outside_ranking":int((~truth.SUMINISTRO_ID.isin(ranking.SUMINISTRO_ID)).sum())}


def scenario(visits,hit_rate,cost_per_visit,recovery_per_hit,operation_cost):
    if visits<0 or not 0<=hit_rate<=1 or min(cost_per_visit,recovery_per_hit,operation_cost)<0:raise ValueError("Supuestos inválidos")
    cost=visits*cost_per_visit+operation_cost
    return {"expected_hits":visits*hit_rate,"cost":cost,"expected_recovery":visits*hit_rate*recovery_per_hit,
            "expected_net":visits*hit_rate*recovery_per_hit-cost,
            "break_even_hit_rate":cost/(visits*recovery_per_hit) if visits>0 and recovery_per_hit>0 else None}
