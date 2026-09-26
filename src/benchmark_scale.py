"""Measure real end-to-end throughput using explicitly replicated load data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import threading
import time
import joblib
import numpy as np
import pandas as pd
import psutil
from src.artifacts import resolve
from src.batch import score_csv


def benchmark(root,rows=1_000_000,batch_size=10000):
    root=Path(root);work=root/"reports/private/scale";work.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(root/"runtime/data/ALIMENTADOR_2025.csv")
    input_path=work/"load_fixture.csv"
    with input_path.open("w",encoding="utf-8",newline="") as stream:
        for start in range(0,rows,batch_size):
            size=min(batch_size,rows-start)
            frame=raw.iloc[np.arange(start,start+size)%len(raw)].reset_index(drop=True).copy()
            frame["SUMINISTRO_ID"]=[f"LOAD_{i:09d}" for i in range(start,start+size)]
            frame.to_csv(stream,index=False,header=start==0,lineterminator="\n")
    print("Load fixture ready",rows,flush=True)
    process=psutil.Process();peak=[process.memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.02):peak[0]=max(peak[0],process.memory_info().rss)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        begin=time.perf_counter();bundle=joblib.load(resolve(root,"models/final_model.joblib"));load=time.perf_counter()-begin
        result=score_csv(input_path,work/"ranked.csv",bundle,2025,"2026-01-01",batch_size,
                         progress=lambda n:print(f"Scale: {n}/{rows}",flush=True) if n%100000==0 else None)
    finally:stop.set();thread.join()
    result.update(test_type="synthetic_load_replicating_real_rows_not_accuracy",rows=rows,batch_size=batch_size,
                  input_bytes=input_path.stat().st_size,peak_rss_mb=round(peak[0]/1024**2,2),load_seconds=round(load,3),
                  python=platform.python_version(),platform=platform.platform(),cpu_count=psutil.cpu_count(),
                  ram_gb=round(psutil.virtual_memory().total/1024**3,2),model_version=bundle["version"],trained_at=bundle["trained_at"])
    # Independently check the whole exported ordering without loading it into RAM.
    total=0;previous=float("inf");unknown=False;valid_count=0
    for chunk in pd.read_csv(work/"ranked.csv",chunksize=batch_size):
        scores=chunk.priority_score.dropna().to_numpy()
        assert not (unknown and len(scores)),"Valid score after unknown row"
        if len(scores):assert scores[0]<=previous+1e-12 and np.all(np.diff(scores)<=1e-12);previous=scores[-1]
        if chunk.priority_score.isna().any():unknown=True
        assert chunk.position.tolist()==list(range(total+1,total+len(chunk)+1))
        total+=len(chunk);valid_count+=len(scores)
    assert total==rows and valid_count==result["valid_scores"]
    result["global_order_verified"]=True;result["identity_uniqueness_enforced"]=True
    (root/"reports/scale_benchmark.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--rows",type=int,default=1_000_000);parser.add_argument("--batch-size",type=int,default=10000)
    args=parser.parse_args();benchmark(Path(__file__).resolve().parents[1],args.rows,args.batch_size)
