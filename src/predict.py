from __future__ import annotations

import numpy as np
import pandas as pd


def score_features(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    missing = set(bundle["feature_columns"])-set(features.columns)
    if missing:
        raise ValueError(f"Faltan variables del modelo: {sorted(missing)}")
    x = bundle["preprocessor"].transform(features[bundle["feature_columns"]])
    cat_members=np.asarray([model.predict_proba(x)[:,1] for model in bundle["catboost"]])
    lgb_members=np.asarray([model.predict_proba(x)[:,1] for model in bundle["lightgbm"]])
    cat = cat_members.mean(axis=0)
    lgb = lgb_members.mean(axis=0)
    weight = bundle["catboost_weight"]
    members=weight*cat_members+(1-weight)*lgb_members
    result=pd.DataFrame({"score_catboost":cat,"score_lightgbm":lgb,
                        "priority_score":weight*cat+(1-weight)*lgb,
                        "model_spread":members.std(axis=0),"seed_min":members.min(axis=0),
                        "seed_max":members.max(axis=0)},index=features.index)
    if "support_lower" in bundle:
        raw=features[bundle["feature_columns"]].to_numpy(dtype=float)
        outside=(raw<bundle["support_lower"])|(raw>bundle["support_upper"])
        result["out_of_range_fraction"]=outside.sum(axis=1)/np.maximum(np.isfinite(raw).sum(axis=1),1)
    return result


def rank(frame: pd.DataFrame, scores: pd.DataFrame, months_observed: pd.Series, recent_months: pd.Series|None=None) -> pd.DataFrame:
    result = frame[["SUMINISTRO_ID"]].reset_index(drop=True).copy()
    result = pd.concat([result,scores.reset_index(drop=True)],axis=1)
    result["data_status"] = np.select([months_observed.ge(6),months_observed.ge(3)],
                                      ["complete","partial"],default="insufficient")
    result["valid_prediction"] = result.data_status.ne("insufficient")
    if recent_months is not None:
        result["valid_prediction"] &= recent_months.to_numpy()>=3
        result.loc[~result.valid_prediction,"data_status"]="insufficient"
    numeric_scores=[c for c in scores if c!="out_of_range_fraction"]
    result.loc[~result.valid_prediction,numeric_scores] = np.nan
    result = result.sort_values(["priority_score","SUMINISTRO_ID"],ascending=[False,True],na_position="last",kind="stable").reset_index(drop=True)
    result["rank"] = np.arange(1,len(result)+1)
    result.loc[~result.valid_prediction,"rank"] = np.nan
    return result
