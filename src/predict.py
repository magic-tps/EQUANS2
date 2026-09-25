from __future__ import annotations

import numpy as np
import pandas as pd


def score_features(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    missing = set(bundle["feature_columns"])-set(features.columns)
    if missing:
        raise ValueError(f"Faltan variables del modelo: {sorted(missing)}")
    x = bundle["preprocessor"].transform(features[bundle["feature_columns"]])
    cat = np.mean([model.predict_proba(x)[:,1] for model in bundle["catboost"]],axis=0)
    lgb = np.mean([model.predict_proba(x)[:,1] for model in bundle["lightgbm"]],axis=0)
    weight = bundle["catboost_weight"]
    return pd.DataFrame({"score_catboost":cat,"score_lightgbm":lgb,
                         "priority_score":weight*cat+(1-weight)*lgb},index=features.index)


def rank(frame: pd.DataFrame, scores: pd.DataFrame, months_observed: pd.Series) -> pd.DataFrame:
    result = frame[["SUMINISTRO_ID"]].reset_index(drop=True).copy()
    result = pd.concat([result,scores.reset_index(drop=True)],axis=1)
    result["data_status"] = np.select([months_observed.ge(6),months_observed.ge(3)],
                                      ["complete","partial"],default="insufficient")
    result["valid_prediction"] = result.data_status.ne("insufficient")
    result.loc[~result.valid_prediction,["priority_score","score_catboost","score_lightgbm"]] = np.nan
    result = result.sort_values("priority_score",ascending=False,na_position="last",kind="stable").reset_index(drop=True)
    result["rank"] = np.arange(1,len(result)+1)
    result.loc[~result.valid_prediction,"rank"] = np.nan
    return result[["SUMINISTRO_ID","rank","priority_score","score_catboost","score_lightgbm","data_status","valid_prediction"]]
