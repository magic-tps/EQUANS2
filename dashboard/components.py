from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.inference import to_csv,to_xlsx


def badge(label: str, value: object) -> None:
    st.metric(label, value if value is not None else "No disponible")


def number(value,decimals=0):
    value=pd.to_numeric(value,errors="coerce")
    if value is None or pd.isna(value):return "—"
    return f"{value:,.{decimals}f}".replace(","," ")


def table_frame(ranking):
    columns=[c for c in ["rank","SUMINISTRO_ID","SED_ID","priority_score","inspection_round","data_status","priority_signals"] if c in ranking]
    view=ranking[columns].copy()
    if "priority_score" in view:view["priority_score"]*=100
    if "data_status" in view:view["data_status"]=view.data_status.map({"complete":"Completo","partial":"Parcial","insufficient":"Insuficiente"})
    return view


def columns():
    return {"rank":st.column_config.NumberColumn("Posición",format="%d",width="small"),
        "SUMINISTRO_ID":st.column_config.TextColumn("Suministro",width="medium"),
        "SED_ID":st.column_config.TextColumn("SED",width="medium"),
        "priority_score":st.column_config.ProgressColumn("Índice /100",min_value=0,max_value=100,format="%.1f",width="medium"),
        "inspection_round":st.column_config.TextColumn("Ronda"),
        "data_status":st.column_config.TextColumn("Datos",width="small"),
        "priority_signals":st.column_config.TextColumn("Señales observadas",width="large")}


def ranking_table(ranking,height=330):
    st.dataframe(table_frame(ranking),hide_index=True,width="stretch",height=height,column_config=columns())


def downloads(frame,key,label="ranking"):
    a,b=st.columns(2)
    with a:st.download_button("Descargar CSV",to_csv(frame),file_name=f"{label}.csv",mime="text/csv",key=f"{key}_csv",width="stretch",on_click="ignore")
    with b:st.download_button("Descargar Excel",to_xlsx(frame),file_name=f"{label}.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key=f"{key}_xlsx",width="stretch",on_click="ignore")


def navigate(page,supply=None):
    st.session_state["navigation"]=page
    if supply is not None:st.session_state["investigate_supply"]=str(supply)
