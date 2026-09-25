"""Historical method labels and an abstaining experimental family classifier.

No method label is inferred directly from a consumption rule. Labels come only
from the inspected intervention catalog; classification remains conditional on
an intervention being positive and cannot establish a current violation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

from src.utils import normalized

DIRECT="Conexión directa / puenteo"
CLANDESTINE="Conexión clandestina"
ELECTRONIC="Manipulación electrónica"
MECHANICAL="Manipulación mecánica"
CIRCUIT="Alteración del circuito de medición"
REPLACEMENT="Sustitución de medidor"
CATALOG={
    "CONEX DIRECTA 2 LINEAS":DIRECT,
    "LINEA DIRECTA EN LA BORNERA DEL MEDIDOR":DIRECT,
    "TRES LINEAS DIRECTAS A LA BORNERA DEL MEDIDOR":DIRECT,
    "CONEXION CLANDESTINA AL CABLE DE ACOMETIDA AREA":CLANDESTINE,
    "CONEXION CLANDESTINA AL CABLE DE ACOMETIDA SUBTERRANEO":CLANDESTINE,
    "CONEXION CLANDESTINA AL CABLE MATRIZ AEREO":CLANDESTINE,
    "CONEXION CLANDESTINA AL CABLE MATRIZ SUBTERRANEO":CLANDESTINE,
    "TARJETA ELECTRONICA MANIPULADA SELLOS VIOLADOS":ELECTRONIC,
    "MEDIDOR CON DISPLAY MANIPULADO ROTO / APAGADO O DANADO":ELECTRONIC,
    "ENGRANAJES DE MEDIDOR MANIPULADOS":MECHANICAL,
    "CONTOMETRO DESCON O MANIP A TRAVES DE INTERRUPTOR":MECHANICAL,
    "NUMERADOR MANIPULADO A TRAVES CAPSULA PERFORADA":MECHANICAL,
    "DISCO TRABADO A TRAVES DE CAPSULA PERFORADA C IV":MECHANICAL,
    "BOBINA MANIPULADA INTERNA Y/O EXTERNAMENTE":CIRCUIT,
    "CONEXION ADICIONAL AL CIRCUITO DE CORRIENTE EN F1 F2 F3":CIRCUIT,
    "HURTO A TRAVES DE LAS FASES":CIRCUIT,
    "MEDIDOR CONECTADO EN CONTRAFASE POR VULNERACION":CIRCUIT,
    "PUENTES DE TENSION ABIERTOS":CIRCUIT,
    "REDUCTOR MANIPULADO PUENTEADO INVERTIDO DESCONECTADO":CIRCUIT,
    "INVERSION EN CONEXIONADO CIRC DE CORRIENTE Y TENSION EN BORNERA DEL MED (MEDICION INDIRECTA) C IV":CIRCUIT,
    "MEDIDOR CAMBIADO NO CORRESPONDE AL SISTEMA":REPLACEMENT,
}


def method_family(history: pd.DataFrame) -> pd.Series:
    column=next(c for c in history if "VULERACI" in normalized(c))
    return history[column].map(normalized).map(CATALOG).fillna("Sin clasificación suficiente")


def wilson_lower(successes: int, total: int) -> float:
    if total==0:return 0.0
    z=1.96; p=successes/total
    return float((p+z*z/(2*total)-z*np.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total))


def train_method_classifier(evaluation: pd.DataFrame, feature_columns: list[str], domain_auc: float, root) -> dict:
    positive=evaluation[evaluation.label.eq(1)].copy()
    training=positive[positive.split.eq("train")]
    counts=training.method_family.value_counts()
    classes=sorted(counts[counts.ge(80)].index)
    supported=positive[positive.method_family.isin(classes)]
    parts={key:supported[supported.split.eq(key)] for key in ("train","selection","test")}
    prep=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),RobustScaler())
    xtrain=prep.fit_transform(parts["train"][feature_columns])
    xsel=prep.transform(parts["selection"][feature_columns]); xtest=prep.transform(parts["test"][feature_columns])
    ytrain=parts["train"].method_family.to_numpy(); ysel=parts["selection"].method_family.to_numpy(); ytest=parts["test"].method_family.to_numpy()
    candidates={
        "logistica":LogisticRegression(max_iter=800,C=.2,class_weight="balanced",random_state=42),
        "catboost":CatBoostClassifier(iterations=200,depth=4,learning_rate=.05,l2_leaf_reg=8,
                                      loss_function="MultiClass",auto_class_weights="Balanced",verbose=False,
                                      random_seed=42,thread_count=4,allow_writing_files=False),
    }
    records=[]; best=None; best_value=-1
    for name,model in candidates.items():
        model.fit(xtrain,ytrain)
        pred=np.asarray(model.predict(xsel)).ravel()
        value=f1_score(ysel,pred,average="macro",zero_division=0)
        records.append({"model":name,"split":"selection","macro_f1":value,"accuracy":accuracy_score(ysel,pred)})
        if value>best_value:best_value=value;best=name
    model=candidates[best]
    prediction=np.asarray(model.predict(xtest)).ravel()
    majority=pd.Series(ytrain).value_counts().index[0]
    majority_pred=np.full(len(ytest),majority,dtype=object)
    metrics={"model":best,"split":"test","macro_f1":float(f1_score(ytest,prediction,average="macro",zero_division=0)),
             "accuracy":float(accuracy_score(ytest,prediction)),"test_count":len(ytest),
             "majority_accuracy":float(accuracy_score(ytest,majority_pred)),
             "majority_macro_f1":float(f1_score(ytest,majority_pred,average="macro",zero_division=0))}
    records.append(metrics)
    label_order=list(model.classes_)
    probabilities=model.predict_proba(xsel)
    top=probabilities.argmax(axis=1)
    thresholds={}
    for j,label in enumerate(label_order):
        for threshold in (.65,.75,.85,.9,.95):
            mask=(top==j)&(probabilities[:,j]>=threshold)
            if mask.sum()>=30 and wilson_lower(int((ysel[mask]==label).sum()),int(mask.sum()))>=.75:
                thresholds[str(label)]=threshold;break
    # Test verifies frozen thresholds. It does not tune them.
    test_prob=model.predict_proba(xtest); test_top=test_prob.argmax(axis=1)
    qualified=[]
    for j,label in enumerate(label_order):
        threshold=thresholds.get(str(label),2.0)
        mask=(test_top==j)&(test_prob[:,j]>=threshold)
        if mask.sum()>=30 and wilson_lower(int((ytest[mask]==label).sum()),int(mask.sum()))>=.60:
            qualified.append(str(label))
    enabled=bool(qualified) and domain_auc<=.80 and metrics["macro_f1"]>=.40 and metrics["accuracy"]>=metrics["majority_accuracy"]+.05
    reason="No se alcanzaron los criterios de validación y transferencia al alimentador objetivo."
    if enabled:reason="Se permiten hipótesis sólo para familias y umbrales verificados; requieren inspección."
    precision,recall,f1,support=precision_recall_fscore_support(ytest,prediction,labels=label_order,zero_division=0)
    pd.DataFrame({"family":label_order,"precision":precision,"recall":recall,"f1":f1,"support":support}).to_csv(root/"reports/method_class_metrics.csv",index=False)
    pd.DataFrame(confusion_matrix(ytest,prediction,labels=label_order),index=label_order,columns=label_order).to_csv(root/"reports/method_confusion.csv",index_label="observed_family")
    pd.DataFrame(records).to_csv(root/"reports/method_experiments.csv",index=False)
    summary={**metrics,"enabled":enabled,"reason":reason,"domain_auc":domain_auc,
             "supported_families":classes,"excluded_rare_events":int((~positive.method_family.isin(classes)).sum()),
             "qualified_families":qualified,"thresholds":thresholds}
    # Retain the evaluated classifier with its own training transform; no post-gate refit.
    return {"model":model,"preprocessor":prep,"feature_columns":feature_columns,"summary":summary}
