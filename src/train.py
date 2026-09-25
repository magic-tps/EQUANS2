from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from src.audit import audit
from src.features import build_features
from src.labels import first_positive
from src.predict import rank, score_features
from src.preprocess import monthly_long
from src.utils import ensure_dirs
from src.validation import observable_metrics, domain_auc

SEED = 42


def _positive_features(history: pd.DataFrame, sed_files: dict) -> tuple[pd.DataFrame,pd.DataFrame]:
    events = first_positive(history).copy()
    events["cutoff"] = events.event_date
    frames = []
    for year in (2024,2025):
        base = events[events.event_date.dt.year.eq(year)].reset_index(drop=True)
        monthly = monthly_long(base,year)
        f = build_features(base,monthly,sed_files)
        f["year"] = year
        frames.append(f)
    return pd.concat(frames,ignore_index=True), events


def _u_features(data: pd.DataFrame, year: int, dates: pd.Series, sed_files: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED+year)
    base = data.copy().reset_index(drop=True)
    pool = pd.to_datetime(dates).dt.dayofyear.to_numpy()
    sampled = rng.choice(pool,size=len(base),replace=True)
    base["cutoff"] = pd.Timestamp(year,1,1)+pd.to_timedelta(sampled-1,unit="D")
    return build_features(base,monthly_long(base,year),sed_files)


def _preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    cat = list(features.select_dtypes(include=["str","object","category"]).columns)
    num = [c for c in features if c not in cat]
    return ColumnTransformer([
        ("numeric",SimpleImputer(strategy="median",keep_empty_features=True),num),
        ("category",Pipeline([("fill",SimpleImputer(strategy="constant",fill_value="UNKNOWN")),
                              ("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat),
    ],sparse_threshold=0)


def _fit_bags(xp, xu, kind: str, config: dict, seed: int, bags: int=2):
    models=[]
    for i in range(bags):
        rng=np.random.default_rng(seed+i)
        take=min(len(xu),max(len(xp),2*len(xp)))
        idx=rng.choice(len(xu),size=take,replace=take>len(xu))
        x=np.concatenate([xp,xu[idx]])
        y=np.r_[np.ones(len(xp)),np.zeros(len(idx))]
        if kind=="catboost":
            m=CatBoostClassifier(iterations=config.get("iterations",160),depth=config.get("depth",5),
                                 learning_rate=config.get("learning_rate",.05),
                                 l2_leaf_reg=config.get("l2_leaf_reg",3),verbose=False,thread_count=4,
                                 random_seed=seed+i,allow_writing_files=False)
        else:
            m=LGBMClassifier(n_estimators=config.get("iterations",160),num_leaves=config.get("num_leaves",15),
                             max_depth=config.get("max_depth",-1),learning_rate=config.get("learning_rate",.05),
                             min_child_samples=config.get("min_child_samples",30),
                             colsample_bytree=config.get("feature_fraction",.9),
                             subsample=config.get("bagging_fraction",.9),subsample_freq=1,
                             random_state=seed+i,verbosity=-1,n_jobs=4)
        m.fit(x,y)
        models.append(m)
    return models


def _predict_bags(models,x):
    return np.mean([m.predict_proba(x)[:,1] for m in models],axis=0)


def train(root: Path) -> dict:
    root=Path(root)
    ensure_dirs(root)
    start=time.perf_counter()
    files=audit(root)
    sed={key:files[key] for key in ("BALANCE_SED","POTENCIA_SED","FALLAS_SED")}
    history=files["HISTORICO_CNR"]
    events=first_positive(history)
    positives,_=_positive_features(history,sed)
    uparts=[]
    for year in (2024,2025):
        data=files[f"ALIMENTADOR_{year}"]
        dates=events.loc[events.event_date.dt.year.eq(year),"event_date"]
        u=_u_features(data,year,dates,sed)
        u["year"]=year
        uparts.append(u)
    unlabeled=pd.concat(uparts,ignore_index=True)
    # The intervention workbook does not timestamp technical metadata. Use only
    # pre-intervention consumption signals in the predictive experiment.
    feature_columns=[c for c in positives if c.startswith("w") or c.startswith("months_")]
    p=positives[positives.months_observed.ge(3)].copy()
    u=unlabeled[unlabeled.months_observed.ge(3)].copy()
    p24=p[p.year.eq(2024)];p25=p[p.year.eq(2025)]
    u24=u[u.year.eq(2024)];u25=u[u.year.eq(2025)]
    if min(len(p24),len(p25),len(u24),len(u25))<76:
        raise RuntimeError("No hay suficientes ejemplos elegibles para validar Hits@76")
    prep=_preprocessor(pd.concat([p24[feature_columns],u24[feature_columns]]))
    xtrain=prep.fit_transform(pd.concat([p24[feature_columns],u24[feature_columns]],ignore_index=True))
    xp=xtrain[:len(p24)];xu=xtrain[len(p24):]
    xval=prep.transform(pd.concat([p25[feature_columns],u25[feature_columns]],ignore_index=True))
    yval=np.r_[np.ones(len(p25)),np.zeros(len(u25))]
    trials=[];configs={}
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    for kind in ("catboost","lightgbm"):
        def objective(trial):
            cfg={"iterations":trial.suggest_int("iterations",100,220,step=60),
                 "learning_rate":trial.suggest_float("learning_rate",.03,.1)}
            if kind=="catboost":
                cfg.update(depth=trial.suggest_int("depth",4,6),l2_leaf_reg=trial.suggest_float("l2_leaf_reg",2,6))
            else:
                cfg.update(num_leaves=trial.suggest_int("num_leaves",11,27,step=8),
                           min_child_samples=trial.suggest_int("min_child_samples",20,50,step=15),
                           feature_fraction=.9,bagging_fraction=.9)
            models=_fit_bags(xp,xu,kind,cfg,SEED,2)
            pred=_predict_bags(models,xval)
            metrics=observable_metrics(yval,pred)
            trials.append(dict(model=kind,trial=trial.number,**cfg,**metrics))
            return metrics["hits_76"]+metrics["average_precision_observable"]*.01
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective,n_trials=4,show_progress_bar=False)
        configs[kind]=study.best_params
    pd.DataFrame(trials).to_csv(root/"reports/optuna_trials.csv",index=False)
    val_models={k:_fit_bags(xp,xu,k,configs[k],SEED+100,3) for k in configs}
    predictions={k:_predict_bags(m,xval) for k,m in val_models.items()}
    comparison=[]
    for kind,s in predictions.items():
        comparison.append(dict(model=kind,**observable_metrics(yval,s)))
    anomaly=IsolationForest(n_estimators=120,random_state=SEED,n_jobs=4)
    anomaly.fit(xu)
    comparison.append(dict(model="isolation_forest_signal",**observable_metrics(yval,-anomaly.score_samples(xval))))
    scaler=RobustScaler().fit(xtrain)
    xp_scaled=np.clip(scaler.transform(xp),-10,10)
    val_scaled=np.clip(scaler.transform(xval),-10,10)
    center=np.median(xp_scaled,axis=0)
    comparison.append(dict(model="positive_euclidean_centroid",**observable_metrics(yval,-np.linalg.norm(val_scaled-center,axis=1))))
    comparison.append(dict(model="positive_manhattan_centroid",**observable_metrics(yval,-np.abs(val_scaled-center).sum(axis=1))))
    cosine=(val_scaled@center)/(np.linalg.norm(val_scaled,axis=1)*np.linalg.norm(center)+1e-9)
    comparison.append(dict(model="positive_cosine_centroid",**observable_metrics(yval,cosine)))
    pca=PCA(n_components=min(8,xtrain.shape[1],len(xp)-1),random_state=SEED).fit(np.clip(scaler.transform(xtrain),-10,10))
    comparison.append(dict(model="positive_pca_centroid",**observable_metrics(yval,-np.linalg.norm(pca.transform(val_scaled)-pca.transform(xp_scaled).mean(axis=0),axis=1))))
    recent=pd.concat([p25,u25],ignore_index=True)["w6_level_change"].fillna(0).to_numpy()
    comparison.append(dict(model="recent_level_drop_signal",**observable_metrics(yval,-recent)))
    best_weight=0.5; best_key=(-1,-1)
    for weight in np.linspace(0,1,11):
        s=weight*predictions["catboost"]+(1-weight)*predictions["lightgbm"]
        m=observable_metrics(yval,s)
        key=(m["hits_76"],m["average_precision_observable"])
        if key>best_key:best_key=key;best_weight=float(weight)
    blended=best_weight*predictions["catboost"]+(1-best_weight)*predictions["lightgbm"]
    selected=observable_metrics(yval,blended)
    comparison.append(dict(model="selected_ensemble",**selected))
    pd.DataFrame(comparison).to_csv(root/"reports/model_comparison.csv",index=False)
    # Seed stability is measured on the same held-out ranking.
    top_sets=[]
    for model in val_models["lightgbm"]:
        top_sets.append(set(np.argsort(-model.predict_proba(xval)[:,1])[:76]))
    overlap=np.mean([len(top_sets[i]&top_sets[j])/76 for i in range(len(top_sets)) for j in range(i+1,len(top_sets))])
    # Simple block ablation using the selected tree family and the same temporal holdout.
    ablations=[]
    for label,prefixes in (("3_months",("w3_","months_")),("3_and_6_months",("w3_","w6_","months_")),
                           ("3_6_12_months",("w3_","w6_","w12_","months_"))):
        cols=[c for c in feature_columns if c.startswith(prefixes)]
        pp=_preprocessor(pd.concat([p24[cols],u24[cols]]))
        xt=pp.fit_transform(pd.concat([p24[cols],u24[cols]],ignore_index=True))
        xv=pp.transform(pd.concat([p25[cols],u25[cols]],ignore_index=True))
        model=_fit_bags(xt[:len(p24)],xt[len(p24):],"lightgbm",configs["lightgbm"],SEED+100,2)
        ablations.append(dict(block=label,**observable_metrics(yval,_predict_bags(model,xv))))
    pd.DataFrame(ablations).to_csv(root/"reports/ablation_results.csv",index=False)
    # Frozen architecture, then refit using every eligible confirmed event and U.
    full=pd.concat([p[feature_columns],u[feature_columns]],ignore_index=True)
    finalprep=_preprocessor(full)
    xfull=finalprep.fit_transform(full)
    final_models={k:_fit_bags(xfull[:len(p)],xfull[len(p):],k,configs[k],SEED+300,3) for k in configs}
    bundle={"preprocessor":finalprep,"feature_columns":feature_columns,
            "catboost":final_models["catboost"],"lightgbm":final_models["lightgbm"],
            "catboost_weight":best_weight,"config":configs,"seed":SEED,"version":"0.1.0",
            "trained_at":datetime.now(timezone.utc).isoformat(),"positive_count":len(p),
            "validation":selected,"top76_overlap":float(overlap),
            "model_input":"pre-cut consumption only"}
    joblib.dump(bundle,root/"models/final_model.joblib",compress=3)
    target=files["ALIMENTADOR_2025"].copy()
    target["cutoff"]=pd.Timestamp("2026-01-01")
    m25=monthly_long(target,2025)
    target_f=build_features(target,m25,sed)
    domain_x=finalprep.transform(p25[feature_columns])
    target_x=finalprep.transform(target_f[feature_columns])
    domain_score=domain_auc(domain_x,target_x,lambda:RandomForestClassifier(n_estimators=100,max_depth=8,
                                                                              random_state=SEED,n_jobs=4))
    scores=score_features(bundle,target_f)
    ranking=rank(target,scores,target_f.months_observed)
    ranking["priority_signals"]="Consumo previo a la fecha de corte; ver explicación del modelo"
    ranking.to_csv(root/"outputs/VOLT_PATROL_RANKING_COMPLETO.csv",index=False)
    ranking.head(76).to_csv(root/"outputs/VOLT_PATROL_TOP_76.csv",index=False)
    ranking.loc[ranking.valid_prediction,["SUMINISTRO_ID","rank"]].to_csv(root/"outputs/VOLT_PATROL_SUBMISSION_PROVISIONAL.csv",index=False)
    # LightGBM's tree SHAP contributions are model-specific, not causal effects.
    names=finalprep.get_feature_names_out()
    contributions=final_models["lightgbm"][0].booster_.predict(target_x,pred_contrib=True)
    impact=contributions[:,:-1]
    explanation=[]
    for i,ident in enumerate(target.SUMINISTRO_ID):
        positive=np.argsort(-impact[i])[:3]
        negative=np.argsort(impact[i])[:3]
        explanation.append(dict(SUMINISTRO_ID=ident,
                                top_positive="; ".join(f"{names[j]}:{impact[i,j]:.3g}" for j in positive if impact[i,j]>0),
                                top_negative="; ".join(f"{names[j]}:{impact[i,j]:.3g}" for j in negative if impact[i,j]<0)))
    explanations=pd.DataFrame(explanation)
    explanations.to_csv(root/"outputs/shap_values.csv",index=False)
    ranking["priority_signals"] = "LightGBM: " + ranking.SUMINISTRO_ID.map(explanations.set_index("SUMINISTRO_ID").top_positive).fillna("")
    ranking.loc[~ranking.valid_prediction,"priority_signals"]=""
    ranking.to_csv(root/"outputs/VOLT_PATROL_RANKING_COMPLETO.csv",index=False)
    ranking.head(76).to_csv(root/"outputs/VOLT_PATROL_TOP_76.csv",index=False)
    importance=pd.DataFrame({"feature":names,"mean_abs_shap":np.abs(impact).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
    importance.to_csv(root/"reports/feature_importance.csv",index=False)
    table=pd.DataFrame(comparison)
    comparison_text="| "+" | ".join(table.columns)+" |\n|"+"|".join("---" for _ in table.columns)+"|\n"
    comparison_text+="\n".join("| "+" | ".join(str(v) for v in row)+" |" for row in table.itertuples(index=False,name=None))
    (root/"reports/validation_results.md").write_text(
        "# Validación observable\n\nEntrenamiento: eventos 2024 y población no etiquetada 2024. "
        "Validación: eventos confirmados 2025 y población no etiquetada 2025, sin compartir suministros. "
        "Corte individual anterior a la intervención. La población U puede contener positivos ocultos. "
        "Las métricas miden recuperación conocida y no precisión real de hurto. "
        "Los positivos históricos pertenecen a otros alimentadores; la extrapolación al alimentador objetivo no está verificada.\n\n"
        +comparison_text+f"\n\nSolapamiento medio Top 76 entre semillas LightGBM: {overlap:.3f}. "
        "La búsqueda de hiperparámetros usa cuatro ensayos por familia sobre una única cohorte; sus estimaciones pueden ser optimistas.\n",encoding="utf-8")
    (root/"reports/domain_analysis.md").write_text(
        f"# Cambio de dominio\n\nAUC de un clasificador de dominio entre positivos 2025 y el alimentador objetivo: {domain_score:.3f}. "
        "Las muestras históricas se cortan antes de la intervención y el objetivo al cierre de 2025; "
        "parte de esta separación puede ser la longitud distinta del historial. "
        "No se aplica adaptación de dominio sin evidencia de mejora en validación.\n",encoding="utf-8")
    (root/"reports/feature_analysis.md").write_text(
        "# Variables\n\nEl modelo usa ventanas de consumo diario de 3, 6 y 12 meses, faltantes y ceros. "
        "Se excluye metadata técnica histórica sin fecha de observación verificable. "
        "La ablación local de ventanas está en `ablation_results.csv`; las contribuciones SHAP del componente LightGBM "
        "están en `outputs/shap_values.csv`. SHAP no implica causalidad.\n",encoding="utf-8")
    (root/"reports/final_report.md").write_text(
        f"# Volt Patrol — informe técnico local\n\n## Datos y etiquetas\n\n"
        f"Se auditaron los siete Excel originales. Hay {len(target)} suministros únicos en 2025. "
        f"Positivos elegibles con al menos tres meses válidos: {len(p)}; población sin etiqueta elegible: {len(u)}. "
        "Se usó la primera intervención positiva por suministro. Los resultados de intervención no entraron como predictores. "
        "Las fechas de lectura y el periodo nominal deben preceder el corte; lecturas incoherentes y días inválidos permanecen faltantes.\n\n"
        "## Experimentos\n\n"
        f"Entrenamiento temporal: {len(p24)} positivos y {len(u24)} U de 2024. "
        f"Validación: {len(p25)} positivos y {len(u25)} U de 2025. "
        "Se compararon CatBoost PU, LightGBM PU, Isolation Forest, similitudes con positivos y una señal de caída reciente. "
        f"El ensamble seleccionado usa peso CatBoost {best_weight:.1f} y LightGBM {1-best_weight:.1f}. "
        f"Hits@76 observable: {selected['hits_76']}; RecallKnown@76: {selected['recall_known_76']:.3f}; "
        f"Average Precision observable: {selected['average_precision_observable']:.3f}; "
        f"solapamiento Top 76 entre semillas LightGBM: {overlap:.3f}. "
        "La selección y los ocho ensayos Optuna comparten una sola cohorte de validación; pueden sobreajustar ese holdout.\n\n"
        "## Ranking y explicación\n\n"
        f"El modelo {bundle['version']} se entrenó {bundle['trained_at']} y cubre {len(target)} suministros; "
        f"{int(ranking.valid_prediction.sum())} tienen score válido y {int((~ranking.valid_prediction).sum())} quedan insuficientes. "
        "El bundle incluye preprocesador, orden de columnas, modelos, configuración y semillas. "
        "Se generaron ranking completo, TOP 76 y contribuciones SHAP del componente LightGBM. "
        f"Duración total de entrenamiento y generación: {time.perf_counter()-start:.1f} s.\n\n"
        "## Límites de validez\n\n"
        f"El clasificador de dominio obtuvo AUC {domain_score:.3f}: los positivos históricos y el objetivo se distinguen fuertemente. "
        "Los positivos históricos proceden de otros alimentadores y no hay negativos confirmados ni resultados conocidos en el alimentador objetivo. "
        "Por ello Hits@K no estima recuperación real en el objetivo; el score es prioridad relativa, no probabilidad calibrada. "
        "No se hizo bootstrap entre cohortes independientes porque sólo hay una cohorte de validación anual. "
        "No se hizo holdout geográfico comparable porque la población U proviene de un único alimentador. "
        "No se entrenó un ranker sin grupos y relevancias observadas coherentes. "
        "No se estimó un prior de clase PU fiable ni se aplicó nnPU con esa prevalencia desconocida. "
        "XGBoost no estaba instalado en el entorno inicial; LOF de alta dimensión y change points formales no se seleccionaron sin validación pertinente. "
        "La metadata técnica histórica sin fecha propia de observación se excluyó del modelo para reducir fuga temporal.\n\n"
        "## Privacidad\n\nLos Excel, rankings, reportes y modelo son archivos locales ignorados por Git; "
        "el repositorio público sólo incluye código y documentación.\n",encoding="utf-8")
    return bundle
