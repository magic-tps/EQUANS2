"""Bounded-memory CSV inference with a disk-backed global ranking, no training."""
from __future__ import annotations
import argparse
from contextlib import closing
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import time

import joblib
import pandas as pd
from dashboard.inference import infer,export_safe
from src.artifacts import resolve


def rank_batches(bundle,frames,output,year,cutoff,progress=None):
    """Input batches must be unique by supply across the entire run.

    A common explicit cutoff prevents a batch's contents changing its feature window.
    Temporary SQLite only holds intermediate data and is removed on exit or failure.
    """
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();count=0;valid=0;columns=None
    with tempfile.TemporaryDirectory(prefix="volt_batch_",dir=output.parent) as directory:
        work=Path(directory)
        with closing(sqlite3.connect(work/"ranking.sqlite")) as connection:
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("CREATE TABLE scored (supply TEXT PRIMARY KEY, score REAL, payload TEXT NOT NULL)")
            for frame in frames:
                if frame.empty:continue
                result=infer(bundle,frame,year,cutoff=cutoff)
                if columns is None:columns=["position"]+result.columns.tolist()
                elif set(columns)-{"position"}!=set(result.columns):
                    raise ValueError("Los lotes deben conservar las mismas columnas")
                rows=json.loads(result.to_json(orient="records",double_precision=15))
                records=[(str(row["SUMINISTRO_ID"]),row["priority_score"],json.dumps(row,ensure_ascii=False,allow_nan=False)) for row in rows]
                try:connection.executemany("INSERT INTO scored VALUES (?,?,?)",records)
                except sqlite3.IntegrityError as exc:raise ValueError("SUMINISTRO_ID repetido entre lotes") from exc
                connection.commit();count+=len(result);valid+=int(result.valid_prediction.sum())
                if progress:progress(count)
            if columns is None:raise ValueError("El archivo está vacío")
            connection.execute("CREATE INDEX priority_order ON scored (score DESC,supply ASC)")
            with (work/"ranking.csv").open("w",encoding="utf-8-sig",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
                for position,(payload,) in enumerate(connection.execute("SELECT payload FROM scored ORDER BY score DESC,supply ASC"),1):
                    row=json.loads(payload);row["position"]=position
                    row["rank"]=position if row["valid_prediction"] else None
                    if "inspection_round" in row:
                        row["inspection_round"]="Sin evaluación" if not row["valid_prediction"] else "Primera ronda" if position<=76 else "Siguiente ronda" if position<=200 else "Seguimiento"
                    for key,value in row.items():
                        if isinstance(value,str) and value.lstrip().startswith(("=","+","-","@")):row[key]="'"+value
                    writer.writerow(row)
        (work/"ranking.csv").replace(output)
    return {"supplies":count,"valid_scores":valid,"seconds":round(time.perf_counter()-started,3),"output_bytes":output.stat().st_size,
            "batch_cutoff":str(pd.Timestamp(cutoff).date()),"global_order":"priority_score DESC, SUMINISTRO_ID ASC; unknown scores last"}


def score_csv(input_path,output,bundle,year,cutoff,batch_size=10000,progress=None):
    if Path(input_path).resolve()==Path(output).resolve():raise ValueError("La salida debe ser distinta del archivo original")
    if batch_size<1:raise ValueError("El tamaño de lote debe ser positivo")
    with pd.read_csv(input_path,chunksize=batch_size,dtype={"SUMINISTRO_ID":"str"}) as batches:
        return rank_batches(bundle,batches,output,year,cutoff,progress)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input");parser.add_argument("output")
    parser.add_argument("--year",type=int,required=True)
    parser.add_argument("--cutoff",required=True)
    parser.add_argument("--batch-size",type=int,default=10000)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=score_csv(args.input,args.output,joblib.load(resolve(root,"models/final_model.joblib")),args.year,args.cutoff,args.batch_size,
                     progress=lambda n:print(f"Procesados {n:,} suministros",flush=True))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
