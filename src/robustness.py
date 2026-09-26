"""Fixed-model diagnostic audits: held-out SED, later periods and simple rules."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from src.experiments import fit_bags,bag_predictions
from src.validation import observable_metrics
from dashboard.inference import infer
from src.preprocess import find_month_columns


def group_fold(value):return int(hashlib.sha256(str(value).encode()).hexdigest()[:8],16)%3


def fixed_fit(train,cols,seed=910):
    # Fixed audit configuration. No search on the retained SED or test periods.
    prep=SimpleImputer(strategy="median",keep_empty_features=True)
    x=prep.fit_transform(train[cols]);y=train.label.to_numpy()
    config={"iterations":200,"depth":4,"learning_rate":.04,"l2_leaf_reg":5,"ratio_u":1.0}
    models=fit_bags(x[y==1],x[y==0],"catboost",config,seed,3)
    return prep,models


def evaluate(train,test,cols):
    if min(train.label.value_counts().reindex([0,1],fill_value=0))<30 or test.label.nunique()!=2:raise ValueError("Cohorte insuficiente")
    assert not set(train.SUMINISTRO_ID)&set(test.SUMINISTRO_ID)
    prep,models=fixed_fit(train,cols)
    prediction=bag_predictions(models,prep.transform(test[cols]))
    metrics=observable_metrics(test.label,prediction,test.SUMINISTRO_ID)
    return {key:metrics[key] for key in ("hits_76","recall_known_76","average_precision_observable","lift_observable_76")}|{
        "train_supplies":len(train),"test_supplies":len(test),"test_known_positives":int(test.label.sum()),"test_unlabeled":int(test.label.eq(0).sum())}


def audit(root):
    root=Path(root);bundle=joblib.load(root/"models/final_model.joblib")
    cohort=pd.read_csv(root/"reports/private/cohort_features.csv",parse_dates=["event_date"])
    cols=bundle["feature_columns"]
    train=cohort[cohort.split.eq("train")];test=cohort[cohort.split.eq("test")]
    geo=[]
    for fold in range(3):
        training=train[train.SED_ID.map(group_fold).ne(fold)]
        holdout=test[test.SED_ID.map(group_fold).eq(fold)]
        overlap=len(set(training.SED_ID)&set(holdout.SED_ID));assert overlap==0
        result={"fold":fold,"sed_overlap":overlap,"retained_seds":holdout.SED_ID.nunique(),**evaluate(training,holdout,cols)}
        geo.append(result);print("SED audit",result,flush=True)
    pd.DataFrame(geo).to_csv(root/"reports/geographic_validation.csv",index=False)
    rolling=[]
    for start,end in (("2025-07-01","2025-10-01"),("2025-10-01","2026-01-01")):
        prior=cohort[cohort.event_date.lt(start)]
        later=cohort[cohort.event_date.ge(start)&cohort.event_date.lt(end)&cohort.split.eq("test")]
        result={"start":start,"end_exclusive":end,**evaluate(prior,later,cols)}
        rolling.append(result);print("Rolling audit",result,flush=True)
    pd.DataFrame(rolling).to_csv(root/"reports/rolling_validation.csv",index=False)
    frozen=pd.read_csv(root/"reports/private/test_predictions.csv").set_index("SUMINISTRO_ID")
    rules={"Modelo elegido (prueba original)":frozen.loc[test.SUMINISTRO_ID,"priority_score"].to_numpy(),
           "Mayor caída relativa reciente":-test.w3_drop_relative.to_numpy(),
           "Mayor cantidad de ceros recientes":test.w3_zeros.to_numpy(),
           "Mayor variabilidad relativa":test.w3_cv.to_numpy()}
    comparison=[]
    for name,pred in rules.items():
        m=observable_metrics(test.label,np.nan_to_num(pred,nan=-1e30),test.SUMINISTRO_ID)
        comparison.append({"reference":name,"hits_76":m["hits_76"],"ap_observable":m["average_precision_observable"],"lift_observable_76":m["lift_observable_76"]})
    comparison.append({"reference":"Selección aleatoria (esperanza)","hits_76":76*float(test.label.mean()),"ap_observable":None,"lift_observable_76":1.0})
    pd.DataFrame(comparison).to_csv(root/"reports/operational_baselines.csv",index=False)
    raw=pd.read_csv(root/"runtime/data/ALIMENTADOR_2025.csv").head(100)
    cases=[];monthly=find_month_columns(raw)
    for history in (0,1,2,3):
        allowed=set(range(13-history,13));frame=raw.copy()
        for month,fields in monthly.items():
            if month not in allowed:frame=frame.drop(columns=list(fields.values()))
        result=infer(bundle,frame,2025,cutoff="2026-01-01")
        assert set(result.SUMINISTRO_ID)==set(raw.SUMINISTRO_ID)
        if history<3:assert result.priority_score.isna().all() and not result.valid_prediction.any()
        cases.append({"available_calendar_months":history,"input_supplies":len(frame),"output_supplies":len(result),"scores":int(result.valid_prediction.sum()),
                      "policy":"Exploración / completar datos" if history<3 else "Modelo donde las lecturas son válidas"})
    cold={"cases":cases,"test_type":"Masked-history compatibility test; not detection accuracy",
          "limitation":"Sin historial no hay evidencia temporal para este modelo. Se implementa cobertura operativa y exploración, no una probabilidad ficticia. Un predictor alternativo necesita metadatos fechados y validación propia."}
    (root/"reports/cold_start_validation.json").write_text(json.dumps(cold,ensure_ascii=False,indent=2),encoding="utf-8")
    summary={"model_version":bundle["version"],"trained_at":bundle["trained_at"],"geographic_folds":3,"total_sed_overlap":sum(r["sed_overlap"] for r in geo),
             "rolling_periods":2,"audit_configuration":{"model":"CatBoost PU","bags":3,"iterations":200,"depth":4,"learning_rate":.04,"ratio_u":1.0},
             "limitation":"Auditorías complementarias sobre las cohortes disponibles; no se usan para volver a elegir el modelo ni son una nueva prueba ciega. Hay validación entre SED. U procede de un único alimentador: no puede demostrarse transferencia entre alimentadores completos con ambas clases. Se requieren inspecciones y población sin etiqueta de otras zonas.",
             "baseline_scope":"Reglas de referencia; el procedimiento operativo actual de EQUANS no fue proporcionado",
             "official_precision":None}
    (root/"reports/robustness_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary


if __name__=="__main__":print(audit(Path(__file__).resolve().parents[1]))
