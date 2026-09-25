from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
import psutil

from dashboard.inference import infer


def benchmark(root: Path) -> dict:
    root=Path(root)
    process=psutil.Process()
    rss=lambda:process.memory_info().rss/1024**2
    before=rss()
    begin=time.perf_counter()
    bundle=joblib.load(root/"models/final_model.joblib")
    load_s=time.perf_counter()-begin
    after_load=rss()
    data=pd.read_excel(root/"data/ALIMENTADOR_2025.xlsx")
    before_infer=rss()
    begin=time.perf_counter()
    ranked=infer(bundle,data,2025)
    inference_s=time.perf_counter()-begin
    after_infer=rss()
    result={"rows":len(data),"valid_predictions":int(ranked.valid_prediction.sum()),
            "bundle_mb":(root/"models/final_model.joblib").stat().st_size/1024**2,
            "load_seconds":load_s,"inference_seconds":inference_s,
            "rss_before_mb":before,"rss_after_load_mb":after_load,
            "rss_before_inference_mb":before_infer,"rss_after_inference_mb":after_infer}
    lines=["# Prueba local de inferencia", "", "Misma ruta `dashboard.inference.infer` que usa Streamlit.", ""]
    lines += [f"- {key}: {value:.3f}" if isinstance(value,float) else f"- {key}: {value}" for key,value in result.items()]
    lines += ["", "RSS medido en este equipo; el pico transitorio puede ser mayor. No es una garantía de límites de Streamlit Community Cloud."]
    (root/"reports/inference_tests.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


if __name__=="__main__":
    print(benchmark(Path(__file__).resolve().parents[1]))
