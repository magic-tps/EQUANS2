from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

KS = (10,25,50,76,100,200,500)


def observable_metrics(y, scores) -> dict[str,float]:
    """Known positives versus unlabeled records, never verified specificity."""
    y = np.asarray(y,dtype=int)
    scores = np.asarray(scores,dtype=float)
    order = np.argsort(-scores,kind="stable")
    ranked = y[order]
    result = {"average_precision_observable":float(average_precision_score(y,scores))}
    for k in KS:
        top = ranked[:min(k,len(y))]
        hits = int(top.sum())
        result[f"hits_{k}"] = hits
        result[f"recall_known_{k}"] = hits/max(1,int(y.sum()))
        result[f"lift_observable_{k}"] = (hits/max(1,len(top)))/(y.mean()+1e-12)
    return result


def domain_auc(train_features, target_features, estimator) -> float:
    """Measure separability between two feature domains on a held-out split."""
    from sklearn.model_selection import train_test_split
    x = np.concatenate([train_features,target_features])
    y = np.r_[np.zeros(len(train_features)),np.ones(len(target_features))]
    xtr,xte,ytr,yte = train_test_split(x,y,test_size=.25,random_state=73,stratify=y)
    model = estimator()
    model.fit(xtr,ytr)
    return float(roc_auc_score(yte,model.predict_proba(xte)[:,1]))
