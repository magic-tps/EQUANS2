from __future__ import annotations
from html import escape
import numpy as np
import pandas as pd
import streamlit as st
from dashboard import data,charts,theme
from dashboard.components import number,ranking_table,table_frame,columns,downloads,navigate


def overview(ranking,bundle,budget):
    theme.heading("01","Una inspección mejor informada.","Prioriza suministros, revisa señales y prepara la siguiente ronda con evidencia de consumo.")
    st.caption("Esta vista operativa muestra el alimentador 2025. «Entrega al jurado» reúne 2024 y 2025 en una sola lista sin duplicados.")
    valid=ranking[ranking.valid_prediction]
    top=valid.head(budget)
    cols=st.columns(4)
    for col,label,value,helptext in zip(cols,["Suministros analizados","Con score disponible","Primera ronda","SED en la ronda"],
        [number(len(ranking)),number(len(valid)),number(len(top)),number(top.SED_ID.nunique()) if "SED_ID" in top else "—"],
        ["Población completa del alimentador 2025","Se exigen tres meses recientes válidos","Capacidad de inspección elegida","Agrupación operativa por SED; no es una ruta geográfica"]):
        with col:theme.metric(label,value,helptext)
    drops=int(top.get("recent_change_pct",pd.Series(dtype=float)).le(-20).sum())
    zeros=int(top.get("recent_zeros",pd.Series(dtype=float)).gt(0).sum())
    theme.brief(f"Tu siguiente ronda: {len(top)} suministros priorizados",
        f"En esta selección, {drops} muestran una caída reciente de al menos 20% y {zeros} tienen meses con consumo cero. Estas señales requieren verificación de campo.")
    left,right=st.columns([1.75,1],gap="large")
    with left:
        st.subheader("Primeros en la lista")
        ranking_table(top.drop(columns=["priority_signals","data_status"],errors="ignore").head(7),height=297)
        st.button("Preparar ronda de inspección  →",type="primary",on_click=navigate,args=("Priorizar inspecciones",),key="overview_plan")
    with right:
        st.subheader("Lectura operativa")
        theme.signal(f"{number(len(ranking)-len(valid))} suministros necesitan completar datos antes de recibir un score.")
        theme.signal(f"{bundle.get('top76_overlap',0):.0%} de coincidencia media del Top 76 entre semillas del modelo.")
        method=bundle.get("method_classifier",{}).get("summary",{})
        theme.note("Método de vulneración: no determinable para casos nuevos con la validación disponible. El catálogo histórico conserva los hallazgos documentados."
            if not method.get("enabled") else "Las hipótesis de método se muestran sólo cuando superan los criterios de validación; una inspección debe confirmarlas.")
        st.button("Explorar métodos observados  →",on_click=navigate,args=("Métodos observados",),key="overview_methods")
    st.divider()
    left,right=st.columns(2,gap="large")
    with left:
        st.subheader("Cómo se distribuye la prioridad")
        charts.draw(charts.score_histogram(ranking),"overview_hist")
        st.caption("Índice relativo del modelo. No equivale a una probabilidad de hurto.")
    with right:
        st.subheader("Concentración de la ronda por SED")
        chart=charts.sed_concentration(ranking,budget)
        if chart is not None:charts.draw(chart,"overview_sed")
        st.caption("Permite agrupar inspecciones sin atribuir las pérdidas de la SED a un cliente.")
    with st.expander("Ver el patrón mensual de los primeros 16 suministros"):
        monthly=data.monthly()
        if not monthly.empty:
            charts.draw(charts.heatmap(ranking,monthly),"overview_heatmap")
            st.caption("Cada celda compara el consumo diario del mes con la mediana personal. Las celdas vacías no representan consumo cero.")


def planner(ranking,bundle,budget):
    theme.heading("02","Del ranking al trabajo de campo.","Filtra la población, selecciona casos y lleva un plan de inspección descargable. Las anotaciones permanecen en esta sesión.")
    a,b,c=st.columns([1.4,1.3,1])
    with a:query=st.text_input("Buscar suministro",placeholder="Escribe parte del identificador")
    with b:seds=st.multiselect("SED",sorted(ranking.get("SED_ID",pd.Series(dtype=str)).dropna().unique()))
    with c:state=st.selectbox("Disponibilidad",["Con score","Todos","Datos insuficientes"])
    view=ranking.copy()
    if query:view=view[view.SUMINISTRO_ID.str.contains(query,case=False,regex=False)]
    if seds:view=view[view.SED_ID.isin(seds)]
    if state=="Con score":view=view[view.valid_prediction]
    if state=="Datos insuficientes":view=view[~view.valid_prediction]
    st.caption(f"{number(len(view))} suministros cumplen los filtros · posiciones respecto al ranking completo")
    tab1,tab2=st.tabs(["Seleccionar suministros","Mi plan de inspección"])
    st.session_state.setdefault("inspection_ids",[])
    st.session_state.setdefault("inspection_annotations",{})
    with tab1:
        n=st.select_slider("Filas a mostrar",options=[10,25,50,76,100,200,500],value=76)
        selected_view=table_frame(view.head(n)).reset_index(drop=True)
        selected_view.insert(0,"Seleccionar",selected_view.SUMINISTRO_ID.isin(st.session_state.inspection_ids))
        edited=st.data_editor(selected_view,hide_index=True,width="stretch",height=440,
            column_config={**columns(),"Seleccionar":st.column_config.CheckboxColumn("Añadir",width="small")},
            disabled=[col for col in selected_view if col!="Seleccionar"],key="planner_editor")
        a,b=st.columns(2)
        with a:
            if st.button("Añadir seleccionados al plan",type="primary",width="stretch"):
                allowed=set(ranking.loc[ranking.valid_prediction,"SUMINISTRO_ID"])
                ids=edited.loc[edited.Seleccionar,"SUMINISTRO_ID"]
                additions=[x for x in ids if x in allowed]
                st.session_state.inspection_ids=list(dict.fromkeys(st.session_state.inspection_ids+additions))
                st.success(f"El plan contiene {len(st.session_state.inspection_ids)} suministros.")
        with b:
            if st.button(f"Usar los primeros {budget} de este filtro",width="stretch"):
                additions=view[view.valid_prediction].head(budget).SUMINISTRO_ID.tolist()
                st.session_state.inspection_ids=list(dict.fromkeys(st.session_state.inspection_ids+additions))
                st.success(f"El plan contiene {len(st.session_state.inspection_ids)} suministros.")
        with st.expander("Descargar el ranking filtrado completo"):
            downloads(view,"filtered","ranking_filtrado")
    with tab2:
        plan=ranking[ranking.SUMINISTRO_ID.isin(st.session_state.inspection_ids)].copy()
        if plan.empty:
            st.info("Selecciona suministros en la primera pestaña para preparar tu plan.")
        else:
            notes=st.session_state.inspection_annotations
            editable=plan[[c for c in ["SUMINISTRO_ID","SED_ID","rank"] if c in plan]].copy()
            editable["Estado"]=[notes.get(x,{}).get("Estado","Pendiente") for x in plan.SUMINISTRO_ID]
            editable["Nota de campo"]=[notes.get(x,{}).get("Nota de campo","") for x in plan.SUMINISTRO_ID]
            changed=st.data_editor(editable,hide_index=True,width="stretch",key="plan_notes",disabled=[c for c in editable if c not in ["Estado","Nota de campo"]],
                column_config={"Estado":st.column_config.SelectboxColumn(options=["Pendiente","Asignado","Visitado","Revisar datos"]),"Nota de campo":st.column_config.TextColumn(width="large")})
            for _,row in changed.iterrows():notes[row.SUMINISTRO_ID]={"Estado":row["Estado"],"Nota de campo":row["Nota de campo"]}
            output=plan.merge(changed[["SUMINISTRO_ID","Estado","Nota de campo"]],on="SUMINISTRO_ID",validate="one_to_one")
            st.caption("Las notas no cambian el ranking ni se convierten en etiquetas de entrenamiento.")
            downloads(output,"inspection_plan","plan_inspeccion")
            if st.button("Vaciar plan",key="clear_plan"):
                st.session_state.inspection_ids=[];st.session_state.inspection_annotations={};st.rerun()


def investigate(ranking,bundle,budget):
    theme.heading("03","Cada suministro, con contexto.","Revisa el comportamiento, la calidad de las lecturas y la explicación del modelo antes de decidir una inspección.")
    choices=ranking.SUMINISTRO_ID.tolist()
    if st.session_state.get("investigate_supply") not in choices:st.session_state["investigate_supply"]=choices[0]
    supply=st.selectbox("Suministro",choices,key="investigate_supply")
    row=ranking.set_index("SUMINISTRO_ID").loc[supply]
    a,b,c,d=st.columns(4)
    with a:theme.metric("Posición",f"#{int(row['rank'])}" if pd.notna(row["rank"]) else "Sin score")
    with b:theme.metric("Índice /100",number(row.priority_score*100,1),"Prioridad relativa; no es probabilidad de hurto")
    with c:theme.metric("Consumo reciente",number(row.get("recent_daily_kwh"),2)+" kWh/día")
    with d:theme.metric("Cambio reciente",number(row.get("recent_change_pct"),1)+"%" if pd.notna(row.get("recent_change_pct")) else "No disponible","Últimos 3 meses frente a los 3 anteriores")
    st.caption(" · ".join(f"{name}: {row.get(key,'No disponible')}" for key,name in [("SED_ID","SED"),("TARIFA","Tarifa"),("TIPO CONEXIONADO","Conexión"),("months_observed","Meses válidos")]))
    for evidence in str(row.get("priority_signals","")).split(" · "):
        if evidence:theme.signal(evidence)
    if not row.valid_prediction:st.warning("Este suministro se conserva en la población. No tiene los tres meses recientes válidos que requiere el modelo.")
    monthly=data.monthly();long=monthly[monthly.SUMINISTRO_ID.eq(supply)].sort_values("nominal_month") if not monthly.empty else pd.DataFrame()
    tabs=st.tabs(["Evidencia de consumo","Explicación del modelo","Preparar inspección"])
    with tabs[0]:
        if long.empty:st.info("Las lecturas locales no están disponibles.")
        else:
            st.subheader("Evolución del consumo")
            mode=st.radio("Unidad",["kWh por día","kWh mensuales"],horizontal=True)
            charts.draw(charts.consumption_curve(long,"daily_kwh" if mode=="kWh por día" else "consumption_kwh"),"supply_consumption")
            left,right=st.columns(2,gap="large")
            with left:
                st.subheader("Comparación anual")
                if long.nominal_month.dt.year.nunique()>1:charts.draw(charts.year_comparison(long),"supply_year")
                else:st.info("No hay lecturas del año anterior para este suministro.")
            with right:
                st.subheader("Comparación con pares")
                raw=data.frame("data/ALIMENTADOR_2025.xlsx")
                peers=raw[raw.SED_ID.eq(row.get("SED_ID"))&raw.SUMINISTRO_ID.ne(supply)] if not raw.empty else pd.DataFrame()
                tariff=peers[peers.TARIFA.eq(row.get("TARIFA"))] if not peers.empty and "TARIFA" in peers else pd.DataFrame()
                group=tariff if len(tariff)>=5 else peers
                if len(group)>=5:
                    peer=monthly[monthly.SUMINISTRO_ID.isin(group.SUMINISTRO_ID)&monthly.nominal_month.dt.year.eq(2025)].groupby("nominal_month").daily_kwh.median().rename("peer_daily").reset_index()
                    charts.draw(charts.consumption_curve(long[long.nominal_month.dt.year.eq(2025)],peers=peer),"supply_peers")
                    st.caption(f"{len(group)} pares de la misma SED{' y tarifa' if len(tariff)>=5 else ''}. El suministro analizado queda excluido.")
                else:st.info("Menos de cinco pares disponibles; no se muestra una referencia poco representativa.")
            with st.expander("Inspeccionar lecturas y calidad de datos"):
                view=long[["nominal_month","reading_date","consumption_kwh","days_billed","daily_kwh","observation_status"]]
                st.dataframe(view,hide_index=True,width="stretch")
    with tabs[1]:
        left,right=st.columns([1.4,1],gap="large")
        with left:
            st.subheader("Variables que influyen en el componente")
            shap=data.frame("outputs/shap_values.csv")
            matching=shap[shap.SUMINISTRO_ID.eq(supply)] if not shap.empty else pd.DataFrame()
            if row.valid_prediction and not matching.empty:
                charts.draw(charts.shap_chart(matching.iloc[0]),"supply_shap")
                explanation=data.metadata("reports/delivery_manifest.json")
                component=explanation.get("shap_component","LightGBM")
                weight=explanation.get("shap_component_weight",1-bundle.get("catboost_weight",0))
                st.caption(f"SHAP de los componentes {component} (peso {weight:.0%}), en escala interna. Aportes positivos empujan su score hacia casos históricos positivos. No establecen causalidad.")
            else:st.info("No hay explicación de un score válido para este suministro.")
        with right:
            st.subheader("Qué tan consistente es la señal")
            if row.valid_prediction:
                theme.signal(f"Rango entre semillas: {row.get('seed_min',row.priority_score)*100:.1f}–{row.get('seed_max',row.priority_score)*100:.1f} /100")
                theme.signal(row.get("support_status","Soporte no disponible"))
                st.caption("El rango mide variación del modelo; no es un intervalo de probabilidad de una infracción.")
            st.subheader("Método de vulneración")
            st.write(row.get("method_hypothesis","No determinable con estos datos"))
            theme.note("Las lecturas mensuales no verifican el estado físico de la instalación. La validación actual no respalda una atribución individual del método.")
    with tabs[2]:
        st.subheader("Ficha para la visita")
        st.write("La ficha reúne identificador, prioridad, señales y limitaciones para orientar la verificación de campo.")
        st.markdown("- Contrastar lecturas y días facturados.\n- Revisar el historial de cambios y la identificación de la instalación.\n- Registrar evidencia y hallazgos de la inspección.\n- Distinguir errores de datos, problemas técnicos y vulneraciones observadas.")
        if row.valid_prediction and st.button("Añadir este suministro al plan",type="primary"):
            current=st.session_state.get("inspection_ids",[])
            st.session_state.inspection_ids=list(dict.fromkeys(current+[supply]));st.success("Suministro añadido al plan de esta sesión.")
        brief_fields={"Suministro":supply,"SED":row.get("SED_ID",""),"Posición":row.get("rank",""),
            "Índice /100":number(row.priority_score*100,1),"Señales":row.get("priority_signals",""),
            "Método":row.get("method_hypothesis","No determinable"),"Modelo":bundle.get("version","")}
        markup="<!doctype html><meta charset='utf-8'><title>Ficha Volt Patrol</title><style>body{font-family:Arial;max-width:850px;margin:48px auto;color:#13243a}dt{font-weight:bold;margin-top:18px}dd{margin:4px 0}small{color:#657386}</style><h1>VOLT PATROL · Ficha de inspección</h1><dl>"
        markup+="".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k,v in brief_fields.items())
        markup+="</dl><small>Prioridad para verificación. No constituye confirmación de infracción ni probabilidad calibrada.</small>"
        st.download_button("Descargar ficha imprimible",markup.encode(),file_name=f"ficha_{supply}.html",mime="text/html",on_click="ignore")
