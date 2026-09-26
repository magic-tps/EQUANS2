from __future__ import annotations
from datetime import date
import pandas as pd
import streamlit as st
from dashboard import data,theme
from dashboard.components import downloads,number
from src.campaigns import empty_portfolio,create_campaign,update_entries,dumps,loads,campaign_metrics,OUTCOMES,STATUSES


def render(ranking,bundle,budget):
    theme.heading("09","Cada mes, una campaña trazable.","Programa visitas, registra evidencia y conserva el historial en una cartera portátil que puedes abrir en otra sesión.")
    st.session_state.setdefault("campaign_portfolio",empty_portfolio())
    portfolio=st.session_state.campaign_portfolio
    with st.expander("Abrir una cartera guardada"):
        archive=st.file_uploader("Cartera de campañas (.json)",type=["json"],key="campaign_import")
        if archive is not None and st.button("Restaurar cartera",key="restore_campaign"):
            try:st.session_state.campaign_portfolio=loads(archive.getvalue());st.rerun()
            except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
    create,follow,history=st.tabs(["Programar mes","Registrar visitas","Resultados y evolución"])
    with create:
        population=data.frame("outputs/delivery/RANKING_AUDITABLE.csv")
        if population.empty:population=ranking
        source=st.selectbox("Población de la campaña",["Unión 2024–2025","Alimentador 2025"])
        selected=population if source=="Unión 2024–2025" else ranking
        a,b,c=st.columns(3)
        with a:month=st.date_input("Mes de campaña",value=date.today(),key="campaign_month").strftime("%Y-%m")
        with b:capacity=int(st.number_input("Visitas disponibles",min_value=1,max_value=1000,value=budget))
        with c:explore=int(st.number_input("Visitas exploratorias sin historial",min_value=0,max_value=capacity,value=0))
        st.caption(f"{capacity-explore} cupos para prioridad del modelo y {explore} para completar datos o explorar. La exploración rota por SED; no atribuye riesgo y no modifica el Top 76 de entrega.")
        if st.button("Crear campaña mensual",type="primary"):
            try:
                campaign=create_campaign(portfolio,selected,month,capacity,explore,bundle["version"],source)
                st.success(f"Campaña creada con {len(campaign['entries'])} suministros. Guarda la cartera para conservarla.")
            except ValueError as exc:st.error(str(exc))
    with follow:
        if not portfolio["campaigns"]:st.info("Crea una campaña o abre una cartera guardada.")
        else:
            choices={f"{c['month']} · {c['source']}":c for c in portfolio["campaigns"]}
            selected_name=st.selectbox("Campaña",list(choices));campaign=choices[selected_name]
            original=pd.DataFrame(campaign["entries"])
            if original.empty:st.info("No hay suministros disponibles para los cupos solicitados.")
            else:
                st.caption(f"Modelo {campaign['model_version']} · selección conservada desde {campaign['created_at'][:10]}")
                mutable=["status","outcome","assignee","visit_date","evidence","note","inspection_cost","recovered_kwh","recovered_amount"]
                visible=["SUMINISTRO_ID","status","outcome","visit_date","evidence","assignee","note","inspection_cost","recovered_kwh","recovered_amount"]
                edited=st.data_editor(original[visible],width="stretch",hide_index=True,height=430,key=f"visits_{campaign['id']}",
                    disabled=["SUMINISTRO_ID"],column_config={
                    "SUMINISTRO_ID":st.column_config.TextColumn("Suministro"),"SED_ID":st.column_config.TextColumn("SED"),
                    "status":st.column_config.SelectboxColumn("Estado",options=STATUSES),"outcome":st.column_config.SelectboxColumn("Resultado",options=OUTCOMES),
                    "assignee":st.column_config.TextColumn("Responsable"),"visit_date":st.column_config.TextColumn("Fecha AAAA-MM-DD"),
                    "evidence":st.column_config.TextColumn("Referencia de evidencia"),"note":st.column_config.TextColumn("Nota"),
                    "inspection_cost":st.column_config.NumberColumn("Costo de visita",min_value=0),
                    "recovered_kwh":st.column_config.NumberColumn("kWh recuperados",min_value=0),
                    "recovered_amount":st.column_config.NumberColumn("Importe recuperado",min_value=0)})
                if st.button("Aplicar resultados",type="primary"):
                    try:update_entries(campaign,edited);st.success("Resultados aplicados. Descarga la cartera actualizada para conservarlos.")
                    except ValueError as exc:st.error(str(exc))
                st.caption("Un resultado concluyente necesita fecha y referencia de evidencia. Las anotaciones no reentrenan el modelo. Usa una misma moneda para costos y recuperos.")
                with st.expander("Ver prioridad y motivo de selección"):
                    st.dataframe(original[["SUMINISTRO_ID","SED_ID","rank","priority_score","purpose","priority_signals"]].rename(columns={
                        "SUMINISTRO_ID":"Suministro","SED_ID":"SED","rank":"Posición","priority_score":"Índice relativo","purpose":"Motivo de visita","priority_signals":"Señales"}),hide_index=True,width="stretch")
                downloads(pd.DataFrame(campaign["entries"]),f"campaign_{campaign['id']}",f"campana_{campaign['month']}")
                with st.expander("Registro de modificaciones"):
                    if campaign["changes"]:st.json(campaign["changes"])
                    else:st.caption("Todavía no hay modificaciones registradas.")
    with history:
        if portfolio["campaigns"]:
            metrics=pd.DataFrame([{"month":c["month"],"source":c["source"],**campaign_metrics(c)} for c in portfolio["campaigns"]])
            st.dataframe(metrics,hide_index=True,width="stretch")
            st.caption("La tasa se calcula sólo sobre visitas concluyentes y se muestra junto a su cobertura. No es la precisión oficial del conjunto ciego. El beneficio observado se calcula únicamente cuando todas las visitas tienen costo e importe registrados.")
        else:st.info("Los resultados aparecerán cuando registres campañas.")
    if portfolio["campaigns"]:
        st.download_button("Guardar cartera completa (.json)",dumps(portfolio),file_name="volt_patrol_campanas.json",mime="application/json",type="primary",on_click="ignore")
        st.caption("Guarda este archivo antes de salir. Para continuar otro día, usa «Abrir una cartera guardada». El servidor no guarda una copia compartida.")
