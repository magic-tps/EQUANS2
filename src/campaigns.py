"""Portable monthly campaigns: durable user-owned JSON, no shared upload storage."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import re
import numpy as np
import pandas as pd

OUTCOMES=("Pendiente","Hurto confirmado","Sin hurto verificado","No concluyente","No visitado")
STATUSES=("Pendiente","Asignado","Visitado","Revisar datos")


def now():return datetime.now(timezone.utc).isoformat()


def empty_portfolio():return {"schema_version":1,"campaigns":[]}


def exploration_queue(ranking,month):
    """Reproducible rotation across SED for missing-history visits, never a theft score."""
    queue=ranking[~ranking.valid_prediction].copy()
    if queue.empty:return queue
    queue["_tie"]=queue.SUMINISTRO_ID.map(lambda x:hashlib.sha256(f"{month}|{x}".encode()).hexdigest())
    queue["_sed"]=queue.get("SED_ID",pd.Series("Sin SED",index=queue.index)).fillna("Sin SED")
    queue=queue.sort_values("_tie")
    queue["_turn"]=queue.groupby("_sed").cumcount()
    return queue.sort_values(["_turn","_tie"]).drop(columns=["_tie","_sed","_turn"])


def create_campaign(portfolio,ranking,month,capacity,exploration,version,source):
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])",month):raise ValueError("Mes inválido; usa AAAA-MM")
    if not 1<=capacity<=10000 or not 0<=exploration<=capacity:raise ValueError("Capacidad o exploración inválida")
    if any(c["month"]==month and c["source"]==source for c in portfolio["campaigns"]):
        raise ValueError("Ya existe una campaña para este mes y población")
    risk=ranking[ranking.valid_prediction].head(capacity-exploration).copy();risk["purpose"]="Inspección por prioridad"
    cold=exploration_queue(ranking,month).head(exploration).copy();cold["purpose"]="Exploración sin historial suficiente"
    selected=pd.concat([risk,cold],ignore_index=True)
    rows=[]
    for record in json.loads(selected.to_json(orient="records",double_precision=15)):
        rows.append({key:record.get(key) for key in ("SUMINISTRO_ID","SED_ID","priority_score","rank","purpose","data_status","priority_signals")}|{
            "status":"Pendiente","outcome":"Pendiente","assignee":"","note":"","evidence":"","visit_date":"",
            "inspection_cost":None,"recovered_kwh":None,"recovered_amount":None})
    campaign={"id":hashlib.sha256(f"{month}|{source}|{now()}".encode()).hexdigest()[:16],"month":month,"source":source,
              "created_at":now(),"model_version":version,"capacity":capacity,"exploration_slots":exploration,
              "entries":rows,"changes":[]}
    portfolio["campaigns"].append(campaign)
    return campaign


def update_entries(campaign,edited):
    rows=json.loads(edited.to_json(orient="records"))
    current={r["SUMINISTRO_ID"]:r for r in campaign["entries"]}
    if len(rows)!=len(current) or {r["SUMINISTRO_ID"] for r in rows}!=set(current):raise ValueError("No se pueden cambiar los suministros de la campaña")
    pending=[]
    for row in rows:
        if row["status"] not in STATUSES or row["outcome"] not in OUTCOMES:raise ValueError("Estado o resultado desconocido")
        if row["outcome"] in ("Hurto confirmado","Sin hurto verificado") and (not str(row.get("evidence") or "").strip() or not str(row.get("visit_date") or "").strip()):
            raise ValueError("Los resultados concluyentes necesitan fecha de visita y referencia de evidencia")
        if row.get("visit_date") and pd.isna(pd.to_datetime(row["visit_date"],errors="coerce")):raise ValueError("Fecha de visita inválida")
        for key in ("inspection_cost","recovered_kwh","recovered_amount"):
            if row.get(key) is not None and (not isinstance(row[key],(int,float)) or not np.isfinite(row[key]) or row[key]<0):raise ValueError("Costos y recuperos deben ser no negativos")
        old=current[row["SUMINISTRO_ID"]]
        fields=("status","outcome","assignee","note","evidence","visit_date","inspection_cost","recovered_kwh","recovered_amount")
        changes={key:{"before":old.get(key),"after":row.get(key)} for key in fields if old.get(key)!=row.get(key)}
        if changes:pending.append((old,{key:row.get(key) for key in fields},changes))
    for old,values,changes in pending:
        old.update(values);campaign["changes"].append({"at":now(),"supply":old["SUMINISTRO_ID"],"fields":changes})


def dumps(portfolio):return json.dumps(portfolio,ensure_ascii=False,allow_nan=False,indent=2).encode("utf-8")


def loads(content):
    if len(content)>10*1024**2:raise ValueError("La cartera supera 10 MB")
    try:data=json.loads(content)
    except (ValueError,UnicodeError) as exc:raise ValueError("Cartera JSON inválida") from exc
    if not isinstance(data,dict) or data.get("schema_version")!=1 or not isinstance(data.get("campaigns"),list):raise ValueError("Formato de cartera no compatible")
    identifiers=set()
    for campaign in data["campaigns"]:
        if not isinstance(campaign,dict) or not {"id","month","source","model_version","entries","changes"}<=set(campaign):raise ValueError("Campaña incompleta")
        if campaign["id"] in identifiers:raise ValueError("Campañas duplicadas")
        identifiers.add(campaign["id"])
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])",campaign["month"]):raise ValueError("Mes inválido")
        entries=campaign["entries"]
        if not isinstance(entries,list) or len(entries)>10000:raise ValueError("Lista de campaña inválida")
        if entries:
            required={"SUMINISTRO_ID","status","outcome","purpose","priority_score","evidence","visit_date"}
            if any(not isinstance(r,dict) or not required<=set(r) for r in entries):raise ValueError("Fila de campaña incompleta")
            if len({r["SUMINISTRO_ID"] for r in entries})!=len(entries):raise ValueError("Suministros duplicados")
            update_entries(campaign,pd.DataFrame(entries))
    return data


def campaign_metrics(campaign):
    entries=campaign["entries"]
    positives=sum(r["outcome"]=="Hurto confirmado" for r in entries)
    negatives=sum(r["outcome"]=="Sin hurto verificado" for r in entries)
    resolved=positives+negatives
    visited=[r for r in entries if r["status"]=="Visitado" or r["outcome"] in ("Hurto confirmado","Sin hurto verificado","No concluyente")]
    all_cost=bool(visited) and all(r.get("inspection_cost") is not None for r in visited)
    all_recovery=bool(visited) and all(r.get("recovered_amount") is not None for r in visited)
    cost=sum(r["inspection_cost"] for r in visited) if all_cost else None
    amount=sum(r["recovered_amount"] for r in visited) if all_recovery else None
    return {"planned":len(entries),"visited":len(visited),"resolved":resolved,"confirmed":positives,
            "confirmed_fraction":positives/resolved if resolved else None,"resolution_coverage":resolved/max(1,len(entries)),
            "inspection_cost":cost,"recovered_amount":amount,"net_observed":amount-cost if amount is not None and cost is not None else None,
            "cost_per_confirmed":cost/positives if cost is not None and positives else None}
