"""Build the explicitly authorized, self-contained dashboard assets."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.cohorts import assert_disjoint

DATA_NAMES=("ALIMENTADOR_2024","ALIMENTADOR_2025","BALANCE_SED",
            "POTENCIA_SED","FALLAS_SED","CALIDAD_TENSION")
COPIES=(
    "models/final_model.joblib",
    "outputs/VOLT_PATROL_RANKING_COMPLETO.csv","outputs/VOLT_PATROL_TOP_76.csv",
    "outputs/shap_values.csv",
    "reports/model_summary.json","reports/model_comparison.csv",
    "reports/feature_importance.csv","reports/file_summary.csv","reports/data_dictionary.csv",
    "reports/method_catalog_summary.csv","reports/method_confusion.csv",
    "reports/method_class_metrics.csv","reports/method_experiments.csv",
    "reports/final_report.md",
    "reports/delivery_manifest.json","reports/scale_benchmark.json","reports/robustness_summary.json",
    "reports/geographic_validation.csv","reports/rolling_validation.csv","reports/operational_baselines.csv",
    "reports/cold_start_validation.json",
    "outputs/delivery/RANKING_AUDITABLE.csv","outputs/delivery/ENTREGA_RANKING.csv","outputs/delivery/ENTREGA_TOP_76.csv",
    "outputs/delivery/ALTERNATIVA_2024.csv","outputs/delivery/ALTERNATIVA_2025.csv","outputs/delivery/shap_values.csv",
)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(root: Path) -> dict:
    root=Path(root)
    destination=root/"runtime"
    bundle=joblib.load(root/"models/final_model.joblib")
    summary=json.loads((root/"reports/model_summary.json").read_text(encoding="utf-8"))
    if bundle["trained_at"]!=summary["trained_at"]:
        raise ValueError("El modelo y sus reportes no pertenecen al mismo entrenamiento")
    for name in ("delivery_manifest.json","robustness_summary.json","scale_benchmark.json"):
        report=json.loads((root/"reports"/name).read_text(encoding="utf-8"))
        if report.get("trained_at")!=bundle["trained_at"] or report.get("model_version")!=bundle["version"]:
            raise ValueError(f"Regenera {name}: pertenece a otra versión del modelo")
    cohort=pd.read_csv(root/"reports/private/cohort_features.csv")
    assert_disjoint(cohort)
    ranking=pd.read_csv(root/"outputs/VOLT_PATROL_RANKING_COMPLETO.csv")
    target=pd.read_excel(root/"data/ALIMENTADOR_2025.xlsx")
    if set(ranking.SUMINISTRO_ID)!=set(target.SUMINISTRO_ID) or len(ranking)!=len(target):
        raise ValueError("El ranking no conserva toda la población objetivo")
    if len(ranking)!=summary["ranking_count"] or ranking.valid_prediction.sum()!=summary["valid_count"]:
        raise ValueError("El ranking no coincide con el reporte")
    published=[]
    sources={}
    for name in DATA_NAMES:
        source=root/"data"/f"{name}.xlsx"
        sources[source.name]=digest(source)
        output=destination/"data"/f"{name}.csv"
        output.parent.mkdir(parents=True,exist_ok=True)
        pd.read_excel(source).to_csv(output,index=False,encoding="utf-8",date_format="%Y-%m-%d")
        published.append(output)
    for relative in COPIES:
        output=destination/relative
        output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,output)
        published.append(output)
    # Keep individual historical labels and cohort IDs local; publish sufficient aggregates.
    predictions=pd.read_csv(root/"reports/private/test_predictions.csv")
    predictions["_tie"]=predictions.SUMINISTRO_ID.map(lambda value:hashlib.sha256(str(value).encode()).hexdigest())
    predictions=predictions.sort_values(["priority_score","_tie"],ascending=[False,True],kind="stable")
    n=min(500,len(predictions));positions=np.arange(1,n+1)
    curve=pd.DataFrame({"position":positions,"known_hits":predictions.label.cumsum().iloc[:n].to_numpy(),
                        "random_reference":positions*predictions.label.mean()})
    output=destination/"reports/recovery_curve.csv"
    curve.to_csv(output,index=False);published.append(output)
    counts={name:{"positive":int(group.label.sum()),"unlabeled":int(group.label.eq(0).sum()),"supplies":len(group)}
            for name,group in cohort.groupby("split")}
    if counts!=bundle["split_counts"]:raise ValueError("La auditoría de particiones no coincide con el modelo")
    output=destination/"reports/split_integrity.json"
    output.write_text(json.dumps({"trained_at":bundle["trained_at"],"overlapping_supplies":0,
                                 "duplicate_supplies":0,"counts":counts},indent=2),encoding="utf-8")
    published.append(output)
    # Stable text bytes on both Windows and Linux, including copied local reports.
    for path in published:
        if path.suffix!=".joblib":
            path.write_bytes(path.read_bytes().replace(b"\r\n",b"\n"))
    manifest={"model_version":bundle["version"],"trained_at":bundle["trained_at"],
              "python":platform.python_version(),
              "dependencies":{name:version(name) for name in ("pandas","numpy","scikit-learn","catboost","lightgbm","joblib","streamlit","plotly","openpyxl")},
              "source_sha256":sources,
              "files":{str(path.relative_to(destination)).replace('\\','/'):{"bytes":path.stat().st_size,"sha256":digest(path)} for path in sorted(published)}}
    (destination/"manifest.json").write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode("utf-8"))
    return {"files":len(published),"megabytes":round(sum(p.stat().st_size for p in published)/1024**2,2),
            "ranking_count":len(ranking),"version":bundle["version"]}


if __name__=="__main__":print(package(Path(__file__).resolve().parents[1]))
