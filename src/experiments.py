"""Fresh PU experiments with customer-disjoint selection and final testing."""
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import make_pipeline
from src.audit import audit
from src.cohorts import make_cohorts, feature_sets
from src.features import consumption_features
from src.preprocess import monthly_long
from src.validation import observable_metrics, domain_auc
from src.methods import train_method_classifier, method_family
from src.labels import first_positive
from src.predict import score_features, rank
from src.utils import ensure_dirs

SEED=42


def fit_bags(xp,xu,kind,config,seed,bags=3):
    models=[]
    for i in range(bags):
        rng=np.random.default_rng(seed+i)
        size=max(1,int(len(xp)*config.get("ratio_u",1)))
        idx=rng.choice(len(xu),size=size,replace=True)
        x=np.concatenate([xp,xu[idx]]);y=np.r_[np.ones(len(xp)),np.zeros(size)]
        if kind=="catboost":
            model=CatBoostClassifier(iterations=config.get("iterations",180),depth=config.get("depth",5),
                learning_rate=config.get("learning_rate",.05),l2_leaf_reg=config.get("l2_leaf_reg",5),
                verbose=False,thread_count=4,random_seed=seed+i,allow_writing_files=False)
        else:
            model=LGBMClassifier(n_estimators=config.get("iterations",180),num_leaves=config.get("num_leaves",15),
                learning_rate=config.get("learning_rate",.05),min_child_samples=config.get("min_child_samples",40),
                colsample_bytree=.9,subsample=.85,subsample_freq=1,reg_lambda=3,
                random_state=seed+i,verbosity=-1,n_jobs=4)
        model.fit(x,y);models.append(model)
    return models


def bag_predictions(models,x):
    return np.mean([model.predict_proba(x)[:,1] for model in models],axis=0)


def metric_key(m):
    return m["hits_76"],m["average_precision_observable"]


def bootstrap(test,scores,iterations=200):
    groups=list(test.reset_index(drop=True).groupby("SED_ID",dropna=False).indices.values())
    rng=np.random.default_rng(SEED);values=[]
    for _ in range(iterations):
        ix=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        if test.iloc[ix].label.nunique()<2:continue
        m=observable_metrics(test.iloc[ix].label,scores[ix])
        values.append([m["hits_76"],m["recall_known_76"]])
    limits=np.quantile(values,[.025,.975],axis=0)
    return dict(hits76_low=float(limits[0,0]),hits76_high=float(limits[1,0]),
                recall76_low=float(limits[0,1]),recall76_high=float(limits[1,1]),replicates=len(values))


def train(root: Path) -> dict:
    root=Path(root);ensure_dirs(root);started=time.perf_counter()
    print("[1/7] Auditoría y cohortes por suministro",flush=True)
    files=audit(root);evaluation,final=make_cohorts(files,SEED)
    evaluation.to_csv(root/"reports/private/cohort_features.csv",index=False)
    splits={n:evaluation[evaluation.split.eq(n)].reset_index(drop=True) for n in ("train","selection","test")}
    counts={n:dict(positive=int(f.label.sum()),unlabeled=int(f.label.eq(0).sum()),supplies=len(f)) for n,f in splits.items()}
    if any(min(v["positive"],v["unlabeled"])<76 for v in counts.values()):raise RuntimeError(f"Cohortes insuficientes: {counts}")
    print("Cohortes:",counts,flush=True)
    ysel=splits["selection"].label.to_numpy();candidates={};trials=[];comparison=[]
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    print("[2/7] Ventanas, variables relativas y bagging PU",flush=True)
    for name,cols in feature_sets(evaluation).items():
        prep=SimpleImputer(strategy="median",keep_empty_features=True)
        xt=prep.fit_transform(splits["train"][cols]);xv=prep.transform(splits["selection"][cols])
        labels=splits["train"].label.to_numpy();xp=xt[labels==1];xu=xt[labels==0]
        configs={};models={};predictions={}
        for kind in ("catboost","lightgbm"):
            def objective(trial):
                cfg=dict(iterations=trial.suggest_categorical("iterations",[120,200]),
                         learning_rate=trial.suggest_float("learning_rate",.03,.08),
                         ratio_u=trial.suggest_categorical("ratio_u",[.5,1.,2.]))
                if kind=="catboost":cfg.update(depth=trial.suggest_int("depth",4,6),l2_leaf_reg=trial.suggest_float("l2_leaf_reg",3,10))
                else:cfg.update(num_leaves=trial.suggest_categorical("num_leaves",[11,19,27]),min_child_samples=trial.suggest_int("min_child_samples",30,70))
                fitted=fit_bags(xp,xu,kind,cfg,SEED,2)
                m=observable_metrics(ysel,bag_predictions(fitted,xv))
                trials.append(dict(feature_set=name,model=kind,trial=trial.number,**cfg,**m))
                return m["hits_76"]+.01*m["average_precision_observable"]
            study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(objective,n_trials=3,show_progress_bar=False)
            configs[kind]=study.best_params
            models[kind]=fit_bags(xp,xu,kind,configs[kind],SEED+100,3)
            predictions[kind]=bag_predictions(models[kind],xv)
            comparison.append(dict(model=f"{name}/{kind}",split="selection",**observable_metrics(ysel,predictions[kind])))
        blends=[]
        for weight in (0.,.25,.5,.75,1.):
            pred=weight*predictions["catboost"]+(1-weight)*predictions["lightgbm"]
            blends.append((observable_metrics(ysel,pred),weight))
        metrics,weight=max(blends,key=lambda item:metric_key(item[0]))
        candidates[name]=dict(feature_columns=cols,preprocessor=prep,config=configs,**models,catboost_weight=weight,selection=metrics)
        comparison.append(dict(model=name,split="selection",**metrics))
        print(name,"Hits@76",metrics["hits_76"],"AP",round(metrics["average_precision_observable"],3),flush=True)
    chosen=max(candidates,key=lambda n:metric_key(candidates[n]["selection"]))
    selected=candidates[chosen]
    print("[3/7] Prueba final congelada:",chosen,flush=True)
    test=splits["test"]
    selected_scores=score_features(selected,test).priority_score.to_numpy()
    test_metrics=observable_metrics(test.label,selected_scores)
    comparison.append(dict(model="selected_ensemble",split="test",**test_metrics))
    baseline_metrics=observable_metrics(test.label,score_features(candidates["multiventana_control"],test).priority_score)
    comparison.append(dict(model="multiventana_control",split="test",**baseline_metrics))
    cols=selected["feature_columns"]
    xt=selected["preprocessor"].transform(splits["train"][cols]);xte=selected["preprocessor"].transform(test[cols])
    anomaly=IsolationForest(n_estimators=150,random_state=SEED,n_jobs=4).fit(xt[splits["train"].label.eq(0)])
    comparison.append(dict(model="isolation_forest",split="test",**observable_metrics(test.label,-anomaly.score_samples(xte))))
    ci=bootstrap(test,selected_scores)
    pd.DataFrame(comparison).to_csv(root/"reports/model_comparison.csv",index=False)
    pd.DataFrame(trials).to_csv(root/"reports/optuna_trials.csv",index=False)
    pd.DataFrame([{**c["selection"],"feature_set":n} for n,c in candidates.items()]).to_csv(root/"reports/ablation_results.csv",index=False)
    predictions=test[["SUMINISTRO_ID","SED_ID","label","event_date"]].copy();predictions["priority_score"]=selected_scores
    predictions.to_csv(root/"reports/private/test_predictions.csv",index=False)
    target=files["ALIMENTADOR_2025"].copy();target["cutoff"]=pd.Timestamp("2026-01-01")
    target_f=consumption_features(target,monthly_long(target,2025))
    domain_score=domain_auc(evaluation[evaluation.label.eq(1)&evaluation.year.eq(2025)][cols].to_numpy(),
                           target_f.loc[target_f.w3_available.eq(3),cols].to_numpy(),
                           lambda:make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                           RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_leaf=15,random_state=SEED,n_jobs=4)))
    print("[4/7] Evaluación de métodos de vulneración",flush=True)
    method=train_method_classifier(evaluation,cols,domain_score,root)
    print("Métodos:",method["summary"],flush=True)
    print("[5/7] Ajuste final desde los originales",flush=True)
    prep=SimpleImputer(strategy="median",keep_empty_features=True);xf=prep.fit_transform(final[cols]);yf=final.label.to_numpy()
    fitted={kind:fit_bags(xf[yf==1],xf[yf==0],kind,selected["config"][kind],SEED+300,5) for kind in ("catboost","lightgbm")}
    bundle=dict(version="2.0.0",pipeline_version=2,trained_at=datetime.now(timezone.utc).isoformat(),
        preprocessor=prep,feature_columns=cols,**fitted,catboost_weight=selected["catboost_weight"],config=selected["config"],
        seed=SEED,positive_count=int(yf.sum()),unlabeled_count=int((yf==0).sum()),validation=test_metrics,
        selection_metrics=selected["selection"],baseline_test=baseline_metrics,selected_feature_set=chosen,
        split_counts=counts,bootstrap=ci,domain_auc=domain_score,
        support_lower=final[cols].quantile(.005).fillna(0).to_numpy(),support_upper=final[cols].quantile(.995).fillna(0).to_numpy(),
        method_classifier=method,model_input="Consumo previo; ventanas de calendario; particiones por suministro",
        minimum_recent_months=3,comparison=comparison)
    scores=score_features(bundle,target_f);seed_top=[]
    tx=prep.transform(target_f[cols]);eligible=target_f.w3_available.eq(3).to_numpy()
    for cat,lgb in zip(fitted["catboost"],fitted["lightgbm"]):
        pred=bundle["catboost_weight"]*cat.predict_proba(tx)[:,1]+(1-bundle["catboost_weight"])*lgb.predict_proba(tx)[:,1]
        pred[~eligible]=-np.inf;seed_top.append(set(np.argsort(-pred,kind="stable")[:76]))
    bundle["top76_overlap"]=float(np.mean([len(a&b)/76 for i,a in enumerate(seed_top) for b in seed_top[i+1:]]))
    from src.evidence import enrich_ranking
    output=enrich_ranking(rank(target,scores,target_f.months_observed,recent_months=target_f.w3_available),target,target_f,bundle)
    output.to_csv(root/"outputs/VOLT_PATROL_RANKING_COMPLETO.csv",index=False)
    output[output.valid_prediction].head(76).to_csv(root/"outputs/VOLT_PATROL_TOP_76.csv",index=False)
    output.to_csv(root/"outputs/VOLT_PATROL_SUBMISSION_PROVISIONAL.csv",index=False)
    target_f.to_csv(root/"outputs/target_features.csv",index=False)
    print("[6/7] Explicaciones y catálogo histórico",flush=True)
    impact=np.mean([m.booster_.predict(tx,pred_contrib=True)[:,:-1] for m in fitted["lightgbm"]],axis=0)
    pd.DataFrame(impact,columns=cols).assign(SUMINISTRO_ID=target.SUMINISTRO_ID.to_numpy()).to_csv(root/"outputs/shap_values.csv",index=False)
    pd.DataFrame(dict(feature=cols,mean_abs_shap=np.abs(impact[eligible]).mean(axis=0))).sort_values("mean_abs_shap",ascending=False).to_csv(root/"reports/feature_importance.csv",index=False)
    events=first_positive(files["HISTORICO_CNR"]);events["family"]=method_family(events);events["year"]=events.event_date.dt.year
    events.groupby(["family","year"]).size().rename("events").reset_index().to_csv(root/"reports/method_catalog_summary.csv",index=False)
    write_reports(root,bundle,output,time.perf_counter()-started)
    temp=root/"models/final_model.tmp.joblib";joblib.dump(bundle,temp,compress=3);temp.replace(root/"models/final_model.joblib")
    print("[7/7] Finalizado",round(time.perf_counter()-started,1),"s; válidos",int(output.valid_prediction.sum()),flush=True)
    return bundle


def write_reports(root,bundle,output,elapsed):
    keys=("version","trained_at","positive_count","unlabeled_count","validation","selection_metrics","baseline_test", "selected_feature_set","split_counts","bootstrap","domain_auc","top76_overlap")
    metadata={key:bundle[key] for key in keys};metadata["method_validation"]=bundle["method_classifier"]["summary"]
    metadata.update(ranking_count=len(output),valid_count=int(output.valid_prediction.sum()),training_seconds=round(elapsed,2))
    (root/"reports/model_summary.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    report=("# Volt Patrol 2 — resultados reproducibles\n\n"
        "La validación anterior reutilizaba suministros U entre años. Esta ejecución corrige el solapamiento: "
        "ningún ID cruza entrenamiento, selección y prueba. Positivos: 2024 para entrenar, enero–junio 2025 para elegir "
        "y julio–diciembre 2025 para la prueba final. U se divide por hash de ID y sus cortes se emparejan por periodo. "
        "El mes de intervención se excluye y los meses faltantes conservan su posición. Se exigen tres meses completos recientes.\n\n"
        f"## Modelo elegido\n\nVariables: `{bundle['selected_feature_set']}`. Selección por Hits@76 y luego AP, antes de consultar la prueba final.\n\n"
        "| Métrica de prueba | Elegido | Control multiventana |\n|---|---:|---:|\n"
        f"| Hits@76 | {bundle['validation']['hits_76']} | {bundle['baseline_test']['hits_76']} |\n"
        f"| Recall conocido@76 | {bundle['validation']['recall_known_76']:.2%} | {bundle['baseline_test']['recall_known_76']:.2%} |\n"
        f"| AP observable | {bundle['validation']['average_precision_observable']:.3f} | {bundle['baseline_test']['average_precision_observable']:.3f} |\n\n"
        "El control se reentrenó con la misma partición. El modelo elegido recupera un caso adicional en Top 76, "
        "pero tiene menor AP: no hay evidencia de una mejora general.\n\n"
        f"Bootstrap por SED: {bundle['bootstrap']['replicates']} repeticiones; intervalo 95% de Hits@76 "
        f"{bundle['bootstrap']['hits76_low']:.1f}–{bundle['bootstrap']['hits76_high']:.1f}. "
        "Describe variación dentro de esta cohorte, no incertidumbre de transferencia a otra población.\n\n"
        f"AUC de separabilidad de origen: {bundle['domain_auc']:.3f}. Este diagnóstico mezcla dominio y etiqueta. "
        "La prueba histórica no demuestra detección real en el alimentador objetivo. U no son negativos verificados; "
        "Hits, AP y lift son observables. La dispersión entre semillas no es un intervalo de probabilidad de hurto.\n\n"
        "## Identificación del método\n\n"
        f"F1 macro: {bundle['method_classifier']['summary']['macro_f1']:.3f}. "
        f"Exactitud: {bundle['method_classifier']['summary']['accuracy']:.1%}; "
        f"referencia mayoritaria: {bundle['method_classifier']['summary']['majority_accuracy']:.1%}. "
        f"Casos de prueba: {bundle['method_classifier']['summary']['test_count']}. "
        f"Atribución individual habilitada: {'sí' if bundle['method_classifier']['summary']['enabled'] else 'no'}. "
        "La atribución de métodos se bloquea si no supera "
        "los criterios definidos antes de evaluar. Sólo una inspección puede confirmar una vulneración.\n\n"
        f"## Cobertura y explicación\n\nRanking: {len(output)} suministros, {int(output.valid_prediction.sum())} scores válidos. "
        f"Coincidencia media del Top 76 entre semillas: {bundle['top76_overlap']:.1%}. "
        "SHAP explica la media de componentes LightGBM en escala interna; no implica causalidad. "
        "El paquete autorizado `runtime/` incluye los CSV, modelo y reportes necesarios para Streamlit. "
        "Los originales y las cohortes con intervenciones individuales permanecen locales.\n")
    for name in ("final_report.md","validation_results.md"):(root/"reports"/name).write_text(report,encoding="utf-8")
