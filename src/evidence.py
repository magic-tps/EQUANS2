"""Observed consumption evidence and optional, validated method hypotheses."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_LABELS={"mean":"Media diaria","median":"Mediana diaria","minimum":"Mínimo diario",
    "maximum":"Máximo diario","std":"Variabilidad","cv":"Variabilidad relativa","slope":"Tendencia",
    "last":"Última lectura diaria","drop":"Mayor caída","drop_relative":"Caída relativa",
    "zeros":"Meses en cero","zero_streak":"Racha de ceros","decrease_streak":"Racha descendente",
    "level_change":"Cambio de nivel","robust_z":"Desviación reciente","relative_slope":"Tendencia relativa",
    "last_ratio":"Último consumo / mediana","relative_change":"Cambio relativo","zero_fraction":"Proporción de ceros",
    "available":"Lecturas disponibles"}


def feature_label(name):
    if name.startswith(("w3_","w6_","w12_")):
        window,feature=name.split("_",1)
        return f"{FEATURE_LABELS.get(feature,feature)} · {window[1:]} meses"
    return {"months_observed":"Meses válidos","months_missing_or_invalid":"Meses sin lectura válida"}.get(name,name)


def evidence_text(row):
    signals=[]
    change=row.get("recent_change_pct",np.nan)
    if pd.notna(change) and change<=-20:signals.append(f"Consumo reciente {abs(change):.0f}% menor que los 3 meses previos")
    zeros=row.get("w3_zeros",0)
    if zeros>0:signals.append(f"{int(zeros)} de los últimos 3 meses con consumo cero")
    if row.get("w3_decrease_streak",0)>=2:signals.append("Descenso en dos meses consecutivos")
    if row.get("w3_cv",0)>.7:signals.append("Variación elevada entre lecturas recientes")
    if not signals:signals.append("Prioridad por el patrón combinado de consumo")
    return " · ".join(signals)


def enrich_ranking(ranking,base,features,bundle):
    meta=base.reset_index(drop=True).copy()
    feature=features.reset_index(drop=True).copy()
    cols=[c for c in ("SUMINISTRO_ID","SED_ID","TARIFA","TIPO CONEXIONADO","ALIMENTADOR_ID","POTENCIA USUARIO CONTRATADO") if c in meta]
    details=meta[cols].copy()
    details["months_observed"]=feature.months_observed
    details["recent_months"]=feature.w3_available
    details["recent_daily_kwh"]=feature.w3_mean
    details["recent_zeros"]=feature.w3_zeros
    # A percentage needs an actual positive previous baseline, no artificial denominator.
    previous=(feature.w6_mean*feature.w6_available-feature.w3_mean*feature.w3_available)/(feature.w6_available-feature.w3_available).replace(0,np.nan)
    change=(feature.w3_mean/previous-1)*100
    details["recent_change_pct"]=change.where(previous.gt(0)&(feature.w6_available-feature.w3_available).ge(2))
    feature["recent_change_pct"]=details.recent_change_pct
    details["priority_signals"]=feature.apply(evidence_text,axis=1)
    result=ranking.merge(details,on="SUMINISTRO_ID",how="left",validate="one_to_one")
    result["inspection_round"]=np.select([~result.valid_prediction,result["rank"].le(76),result["rank"].le(200)],
        ["Sin evaluación","Primera ronda","Siguiente ronda"],default="Seguimiento")
    result["support_status"]=np.where(result.get("out_of_range_fraction",pd.Series(0,index=result.index)).gt(.2),
        "Fuera del rango histórico","Dentro del rango histórico")
    result.loc[~result.valid_prediction,"support_status"]="Datos insuficientes"
    result.loc[~result.valid_prediction,"priority_signals"]="Faltan tres meses recientes completos y válidos"
    result["method_hypothesis"]="No determinable con estos datos"
    method=bundle.get("method_classifier")
    if method and method["summary"].get("enabled",False):
        transformed=method["preprocessor"].transform(feature[method["feature_columns"]])
        probs=method["model"].predict_proba(transformed);order=probs.argmax(axis=1)
        labels=np.asarray(method["model"].classes_)[order]
        hypotheses={}
        for i,label in enumerate(labels):
            if label in method["summary"]["qualified_families"] and probs[i,order[i]]>=method["summary"]["thresholds"][label]:
                hypotheses[str(meta.SUMINISTRO_ID.iloc[i])]=f"Hipótesis: {label}"
        supported=result.valid_prediction & result.support_status.eq("Dentro del rango histórico")
        result.loc[supported,"method_hypothesis"]=result.loc[supported,"SUMINISTRO_ID"].map(hypotheses).fillna("No determinable con estos datos")
    return result
