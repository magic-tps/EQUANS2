"""One unique supply per delivery, plus year-specific alternatives and an audit."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from dashboard.inference import infer
from src.artifacts import resolve
from src.features import consumption_features
from src.preprocess import monthly_long


def build_delivery(root):
    root=Path(root);bundle=joblib.load(resolve(root,"models/final_model.joblib"))
    rows=[];ids={};shap=[]
    for year in (2024,2025):
        source=resolve(root,f"data/ALIMENTADOR_{year}.xlsx")
        raw=pd.read_excel(source) if source.suffix==".xlsx" else pd.read_csv(source)
        ids[year]=set(raw.SUMINISTRO_ID)
        result=infer(bundle,raw,year,cutoff=f"{year+1}-01-01")
        result["source_year"]=year
        result["available_years"]=str(year)
        rows.append(result)
        base=raw.copy();base["cutoff"]=pd.Timestamp(year+1,1,1)
        features=consumption_features(base,monthly_long(raw,year))
        transformed=bundle["preprocessor"].transform(features[bundle["feature_columns"]])
        # Explain the component that actually carries the larger ensemble weight.
        if bundle["catboost_weight"]>=.5:
            values=np.mean([m.get_feature_importance(Pool(transformed),type="ShapValues")[:,:-1] for m in bundle["catboost"]],axis=0)
            component="CatBoost";weight=bundle["catboost_weight"]
        else:
            values=np.mean([m.booster_.predict(transformed,pred_contrib=True)[:,:-1] for m in bundle["lightgbm"]],axis=0)
            component="LightGBM";weight=1-bundle["catboost_weight"]
        shap.append(pd.DataFrame(values,columns=bundle["feature_columns"]).assign(SUMINISTRO_ID=raw.SUMINISTRO_ID.to_numpy(),source_year=year))
    union=pd.concat(rows,ignore_index=True).sort_values("source_year").drop_duplicates("SUMINISTRO_ID",keep="last")
    union["available_years"]=union.SUMINISTRO_ID.map(lambda x:"2024,2025" if x in ids[2024]&ids[2025] else "2024" if x in ids[2024] else "2025")
    union=union.sort_values(["priority_score","SUMINISTRO_ID"],ascending=[False,True],na_position="last",kind="stable").reset_index(drop=True)
    union["position"]=np.arange(1,len(union)+1)
    union["rank"]=union.position.where(union.valid_prediction)
    union["inspection_round"]=np.where(~union.valid_prediction,"Sin evaluación",np.where(union.position<=76,"Primera ronda",np.where(union.position<=200,"Siguiente ronda","Seguimiento")))
    assert len(union)==len(ids[2024]|ids[2025]) and union.SUMINISTRO_ID.is_unique
    destination=root/"outputs/delivery";destination.mkdir(parents=True,exist_ok=True)
    union.to_csv(destination/"RANKING_AUDITABLE.csv",index=False,lineterminator="\n")
    union[["SUMINISTRO_ID"]].to_csv(destination/"ENTREGA_RANKING.csv",index=False,lineterminator="\n")
    union.head(76)[["SUMINISTRO_ID"]].to_csv(destination/"ENTREGA_TOP_76.csv",index=False,lineterminator="\n")
    for year,result in zip((2024,2025),rows):result[["SUMINISTRO_ID"]].to_csv(destination/f"ALTERNATIVA_{year}.csv",index=False,lineterminator="\n")
    combined_shap=pd.concat(shap).sort_values("source_year").drop_duplicates("SUMINISTRO_ID",keep="last").drop(columns="source_year")
    combined_shap.to_csv(destination/"shap_values.csv",index=False,lineterminator="\n")
    # Refresh the existing 2025 explanations without changing any fitted model or score.
    shap[-1].drop(columns="source_year").to_csv(root/"outputs/shap_values.csv",index=False,lineterminator="\n")
    valid_ids=set(rows[-1].loc[rows[-1].valid_prediction,"SUMINISTRO_ID"])
    imp=shap[-1].set_index("SUMINISTRO_ID").drop(columns="source_year").loc[list(valid_ids)].abs().mean()
    imp.rename("mean_abs_shap").rename_axis("feature").sort_values(ascending=False).to_csv(root/"reports/feature_importance.csv")
    report={"scope":"Unión 2024 y 2025; una fila por suministro; se usa el año más reciente disponible",
            "template_status":"Sin plantilla oficial proporcionada; columna única SUMINISTRO_ID en orden de prioridad",
            "supplies":len(union),"valid_scores":int(union.valid_prediction.sum()),"unknown_scores":int((~union.valid_prediction).sum()),
            "shared_between_years":len(ids[2024]&ids[2025]),"only_2024":len(ids[2024]-ids[2025]),"only_2025":len(ids[2025]-ids[2024]),
            "known_blind_thefts":76,"official_precision_at_76":None,"model_version":bundle["version"],"trained_at":bundle["trained_at"],
            "ordering":"Score relativo descendente; empate por ID. Sin score al final, sin afirmar un orden de riesgo entre ellos.",
            "no_history_policy":"Verificación exploratoria por SED y rotación mensual; no se inventa probabilidad",
            "shap_component":component,"shap_component_weight":weight,
            "metric_scope":"Las métricas publicadas corresponden a la prueba histórica; no a las etiquetas del conjunto ciego"}
    (root/"reports/delivery_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)
    return report


if __name__=="__main__":build_delivery(Path(__file__).resolve().parents[1])
