from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dashboard import data,charts,theme
from dashboard.components import number,ranking_table
from src.evidence import feature_label


def methods(ranking,bundle,budget):
    theme.heading("04","Métodos documentados. Inferencia evaluada.","Explora las familias registradas en intervenciones confirmadas y la evidencia disponible para anticiparlas.")
    summary=bundle.get("method_classifier",{}).get("summary",{})
    catalog=data.frame("reports/method_catalog_summary.csv")
    if catalog.empty:st.info("El catálogo de métodos aún no está disponible.");return
    a,b,c,d=st.columns(4)
    with a:theme.metric("Intervenciones únicas",number(catalog.events.sum()))
    with b:theme.metric("Familias documentadas",number(catalog.family.nunique()))
    with c:theme.metric("Casos de prueba",number(summary.get("test_count")))
    with d:theme.metric("Atribución individual","Habilitada" if summary.get("enabled") else "No habilitada")
    theme.brief("La aplicación se abstiene cuando la evidencia no alcanza.",
        "Se entrenaron clasificadores con consumos anteriores a la intervención y se probaron en casos posteriores. Su rendimiento no permite atribuir de forma fiable un método a cada suministro nuevo."
        if not summary.get("enabled") else "Sólo se muestran hipótesis para familias y umbrales que superaron la validación. La verificación física sigue siendo necesaria.")
    tabs=st.tabs(["Catálogo histórico","Prueba de identificación del método"])
    with tabs[0]:
        plot=catalog.copy();plot["year"]=plot.year.astype(str)
        fig=px.bar(plot,x="events",y="family",color="year",orientation="h",barmode="group",
            labels={"events":"Intervenciones documentadas","family":"","year":"Año"},color_discrete_sequence=["#526c8f",charts.TEAL])
        charts.draw(charts.polish(fig,380),"method_history")
        descriptions={
            "Conexión directa / puenteo":"Hallazgos registrados como conexión directa o líneas directas en la instalación de medición.",
            "Conexión clandestina":"Intervenciones que documentan conexiones clandestinas en acometida o cable matriz.",
            "Manipulación electrónica":"Hallazgos documentados en componentes electrónicos o display del medidor.",
            "Manipulación mecánica":"Hallazgos documentados en elementos mecánicos de medición.",
            "Alteración del circuito de medición":"Intervenciones que documentan alteraciones del circuito de medición.",
            "Sustitución de medidor":"Registros donde el medidor observado no correspondía al sistema."}
        family=st.selectbox("Examinar familia",sorted(catalog.family.unique()))
        st.write(descriptions.get(family,"La familia se deriva del texto documentado en el histórico."))
        st.caption("Estos métodos fueron observados en el histórico. Su frecuencia no demuestra que ocurran en un suministro del ranking.")
    with tabs[1]:
        a,b,c=st.columns(3)
        with a:theme.metric("F1 macro en prueba",number(summary.get("macro_f1"),3),"Promedia el F1 de cada familia y da igual peso a familias frecuentes y menos frecuentes")
        with b:theme.metric("Exactitud del modelo",f"{summary.get('accuracy',0):.1%}","Sólo en positivos históricos y familias con soporte suficiente")
        with c:theme.metric("Referencia mayoritaria",f"{summary.get('majority_accuracy',0):.1%}","Resultado de asignar a todos la familia más frecuente del entrenamiento")
        st.write(summary.get("reason","Evaluación no disponible"))
        matrix=data.frame("reports/method_confusion.csv")
        if not matrix.empty:
            matrix=matrix.set_index("observed_family")
            normalized=matrix.div(matrix.sum(axis=1).replace(0,np.nan),axis=0)
            fig=px.imshow(normalized,text_auto=".0%",color_continuous_scale=["#101f33",charts.TEAL],zmin=0,zmax=1,
                labels=dict(x="Familia predicha",y="Familia observada",color="Proporción"))
            fig.update_layout(coloraxis_colorbar_tickformat=".0%")
            charts.draw(charts.polish(fig,410),"method_confusion")
        st.caption(f"{summary.get('excluded_rare_events',0)} eventos elegibles de familias raras quedaron fuera de esta comparación. La capacidad de inferencia se limita a las familias evaluadas.")
        with st.expander("Criterios para permitir una hipótesis"):
            st.write("Umbrales elegidos sólo en selección; al menos 30 predicciones por familia; límite inferior de precisión de Wilson ≥ 0,75 en selección y ≥ 0,60 en prueba; F1 macro ≥ 0,40; exactitud superior a la referencia mayoritaria en 5 puntos; AUC de origen ≤ 0,80. Si falla un criterio, no se atribuye método.")


def network(ranking,bundle,budget):
    theme.heading("05","La red aporta contexto.","Examina el balance de energía y agrupa las inspecciones por SED. Las pérdidas de red no constituyen evidencia individual.")
    balance=data.frame("data/BALANCE_SED.xlsx")
    if balance.empty:st.info("No está disponible el balance local de SED.");return
    balance["PERIODO BALANCE"]=pd.to_datetime(balance["PERIODO BALANCE"],errors="coerce")
    for column in balance.columns.difference(["SED_ID","PERIODO BALANCE"]):
        balance[column]=pd.to_numeric(balance[column],errors="coerce")
    target_seds=set(ranking.SED_ID.dropna())
    balance=balance[balance.SED_ID.isin(target_seds)&balance["PERIODO BALANCE"].lt("2026-01-01")].copy()
    if balance.empty:st.info("No hay balances anteriores al corte para las SED objetivo.");return
    choices=sorted(balance.SED_ID.unique())
    sed=st.selectbox("SED a investigar",choices)
    selected=balance[balance.SED_ID.eq(sed)].sort_values("PERIODO BALANCE")
    last=selected.iloc[-1]
    st.caption(f"Último balance disponible: {last['PERIODO BALANCE']:%m/%Y} · corte del ranking: diciembre de 2025")
    a,b,c,d=st.columns(4)
    for col,label,key,unit in [(a,"Energía distribuida","E. DISTRIBUIDA MES MWh"," MWh"),(b,"Energía facturada","E. FACTURADA MES MWh"," MWh"),(c,"Pérdida reportada","PERDIDA  MES MWh"," MWh"),(d,"Pérdida relativa","PERDIDA MES %","%")]:
        with col:theme.metric(label,number(last[key]*100 if unit=="%" else last[key],1)+unit)
    left,right=st.columns([1.35,1],gap="large")
    with left:
        st.subheader("Balance mensual")
        charts.draw(charts.energy_chart(selected),"sed_energy")
    with right:
        st.subheader("Suministros de esta SED")
        subset=ranking[ranking.SED_ID.eq(sed)]
        ranking_table(subset.drop(columns=["priority_signals","inspection_round","data_status"],errors="ignore").head(8),height=330)
        st.caption(f"{len(subset)} suministros de la población objetivo. Proyección no técnica del último balance: {number(last['PROYEC_PERDIDAS NO TECNICAS MES MWh'],2)} MWh.")
    a,b,c=st.tabs(["Potencia y clientes","Fallas registradas","Calidad de tensión"])
    with a:
        power=data.frame("data/POTENCIA_SED.xlsx")
        if not power.empty:
            p=power[power.SED_ID.eq(sed)&pd.to_numeric(power.iloc[:,-1],errors="coerce").le(2025)].copy()
            for col in ["POTENCIA [KVA]","Clientes BT Cantidad Referencial"]:
                p[col]=pd.to_numeric(p[col],errors="coerce")
            st.dataframe(p,hide_index=True,width="stretch")
            st.caption("Valores de referencia por año, tal como aparecen en el archivo de origen.")
    with b:
        failures=data.frame("data/FALLAS_SED.xlsx")
        if not failures.empty:
            f=failures[failures.SED_ID.eq(sed)&pd.to_numeric(failures.iloc[:,-1],errors="coerce").le(2025)]
            if f.empty:st.info("Sin registros de fallas para esta SED en el archivo disponible.")
            else:st.dataframe(f,hide_index=True,width="stretch")
            st.caption("Registros por llave y año. No se suman como eventos únicos porque pueden compartir ámbito de red.")
    with c:
        quality=data.frame("data/CALIDAD_TENSION.xlsx")
        if not quality.empty:
            q=quality[quality.SED_ID.eq(sed)].drop_duplicates()
            if q.empty:st.info("Sin observaciones de calidad de tensión en este archivo. Ausencia de registros no implica ausencia de incidencias.")
            else:st.dataframe(q,hide_index=True,width="stretch")
    with st.expander("Explorar todas las SED con balance"):
        latest=balance.sort_values("PERIODO BALANCE").drop_duplicates("SED_ID",keep="last")
        counts=ranking[ranking.valid_prediction].head(budget).groupby("SED_ID").size().rename("En primera ronda")
        latest=latest.merge(counts,left_on="SED_ID",right_index=True,how="left")
        latest["En primera ronda"]=latest["En primera ronda"].fillna(0).astype(int)
        st.dataframe(latest.sort_values("En primera ronda",ascending=False),hide_index=True,width="stretch")
        st.caption("Cada fila conserva su fecha de balance. No se agregan como si pertenecieran necesariamente al mismo mes.")


def models(ranking,bundle,budget):
    theme.heading("06","El modelo también rinde cuentas.","Resultados de una prueba posterior que no intervino en la selección. Métricas observables frente a una población sin etiqueta.")
    validation=bundle.get("validation",{});baseline=bundle.get("baseline_test",{})
    a,b,c,d=st.columns(4)
    with a:theme.metric("Hits @76",number(validation.get("hits_76")),"Positivos históricos entre las primeras 76 posiciones de la prueba")
    with b:theme.metric("Recall conocido @76",f"{validation.get('recall_known_76',0):.2%}")
    with c:theme.metric("AP observable",number(validation.get("average_precision_observable"),3))
    with d:theme.metric("Estabilidad Top 76",f"{bundle.get('top76_overlap',0):.1%}","Solapamiento medio en el alimentador objetivo entre semillas")
    theme.note("La población sin etiqueta puede incluir vulneraciones no descubiertas. Estas métricas no miden precisión real de hurto ni garantizan el resultado de las inspecciones del alimentador objetivo.")
    first,second,third=st.tabs(["Prueba final","Selección de arquitectura","Variables y trazabilidad"])
    with first:
        st.subheader("Comparación con el modelo de control")
        comparison=data.frame("reports/model_comparison.csv")
        if not comparison.empty:
            if "split" in comparison:
                test=comparison[comparison.split.eq("test")]
                st.dataframe(test[["model","hits_76","recall_known_76","average_precision_observable","lift_observable_76"]].rename(columns={"model":"Modelo","hits_76":"Hits@76","recall_known_76":"Recall conocido@76","average_precision_observable":"AP observable","lift_observable_76":"Lift observable@76"}),hide_index=True,width="stretch")
            st.caption("El control multiventana se reentrenó con las mismas particiones. La arquitectura se eligió por Hits@76 y después por AP en selección.")
        if baseline:
            hits=validation["hits_76"]-baseline["hits_76"]
            ap=validation["average_precision_observable"]-baseline["average_precision_observable"]
            st.write(f"En esta prueba: {hits:+.0f} casos conocidos en Top 76 y {ap:+.3f} puntos de AP frente al control. Una mejora en Top 76 puede coexistir con una AP menor.")
        curve=data.frame("reports/recovery_curve.csv")
        if not curve.empty:charts.draw(charts.recovery_curve(curve),"test_recovery")
        ci=bundle.get("bootstrap",{})
        if ci:
            st.caption(f"Bootstrap por SED, {ci['replicates']} repeticiones: intervalo 95% de Hits@76 {ci['hits76_low']:.0f}–{ci['hits76_high']:.0f}. Describe variación dentro de esta cohorte; no cubre el traslado a otra población.")
    with second:
        splits=bundle.get("split_counts",{})
        st.subheader("Separación antes de entrenar")
        if splits:st.dataframe(pd.DataFrame(splits).T.rename(columns={"positive":"Positivos","unlabeled":"Sin etiqueta","supplies":"Suministros"}),width="stretch")
        st.write("Positivos de 2024 para entrenamiento; enero–junio de 2025 para selección; julio–diciembre de 2025 para prueba. La población sin etiqueta se separa por identificador de suministro y se empareja por periodo.")
        if not comparison.empty and "split" in comparison:
            selection=comparison[comparison.split.eq("selection")]
            st.dataframe(selection[["model","hits_76","average_precision_observable"]],hide_index=True,width="stretch")
        st.caption(f"Arquitectura elegida antes de consultar la prueba: {bundle.get('selected_feature_set','No disponible')}.")
        st.subheader("Diferencias de población")
        st.metric("AUC del clasificador de origen",number(bundle.get("domain_auc"),3))
        st.caption("Distingue positivos históricos y objetivo usando las variables del modelo. Mezcla diferencias de origen y de etiqueta; no es una medición pura del cambio de dominio.")
    with third:
        importance=data.frame("reports/feature_importance.csv")
        if not importance.empty:
            top=importance.head(12).sort_values("mean_abs_shap")
            fig=go.Figure(go.Bar(x=top.mean_abs_shap,y=top.feature.map(feature_label),orientation="h",marker_color=charts.TEAL))
            fig.update_layout(xaxis_title="Media del aporte SHAP absoluto · LightGBM")
            charts.draw(charts.polish(fig,370),"global_shap")
        st.write(f"Versión {bundle.get('version')} · entrenamiento {bundle.get('trained_at','')[:10]} · semilla {bundle.get('seed')}")
        with st.expander("Configuración reproducible"):
            st.json(bundle.get("config",{}))
            st.write("Variables:",bundle.get("feature_columns",[]))
        st.caption("Las variables de resultado, tipificación, recupero e identificador no entran al predictor de prioridad.")


def quality(ranking,bundle,budget):
    theme.heading("08","Calidad que puedes inspeccionar.","Revisa cobertura, originales y controles temporales. Un dato faltante conserva su significado y nunca se transforma en consumo cero.")
    a,b,c=st.columns(3)
    with a:theme.metric("Población conservada",number(len(ranking)))
    with b:theme.metric("Sin score por datos",number((~ranking.valid_prediction).sum()))
    with c:theme.metric("Archivos de origen","7 Excel")
    tabs=st.tabs(["Lecturas y cobertura","Auditoría de archivos","Controles de validación"])
    with tabs[0]:
        long=data.monthly()
        if not long.empty:
            long=long[long.nominal_month.dt.year.eq(2025)].copy()
            labels={"valid":"Válida","observed_zero":"Cero observado","missing":"Faltante","invalid":"Valor inválido","inconsistent_reading":"Fecha inconsistente"}
            long["Estado"]=long.observation_status.map(labels)
            counts=long.groupby(["nominal_month","Estado"]).size().rename("Lecturas").reset_index()
            fig=px.bar(counts,x="nominal_month",y="Lecturas",color="Estado",barmode="stack",
                color_discrete_map={"Válida":charts.TEAL,"Cero observado":charts.BLUE,"Faltante":"#4d627e","Valor inválido":charts.AMBER,"Fecha inconsistente":"#ce7b94"})
            charts.draw(charts.polish(fig,330),"data_quality")
        with st.expander("Suministros que necesitan completar datos"):
            ranking_table(ranking[~ranking.valid_prediction].head(200),height=380)
        st.caption("Para el modelo actual deben existir tres meses de calendario recientes con consumo no negativo, días de 1–45 y fechas válidas anteriores al corte.")
    with tabs[1]:
        summary=data.frame("reports/file_summary.csv")
        st.dataframe(summary,hide_index=True,width="stretch")
        dictionary=data.frame("reports/data_dictionary.csv")
        if not dictionary.empty:
            selected=st.selectbox("Diccionario de archivo",sorted(dictionary.file.unique()))
            st.dataframe(dictionary[dictionary.file.eq(selected)],hide_index=True,width="stretch")
        st.caption("Los originales se abren para lectura. El repositorio incluye un paquete autorizado con CSV, modelo y reportes para ejecutar este dashboard. El histórico individual de intervenciones permanece local.")
    with tabs[2]:
        integrity=data.metadata("reports/split_integrity.json")
        if integrity:
            overlap=integrity["overlapping_supplies"]
            if overlap==0:st.success("0 suministros compartidos entre entrenamiento, selección y prueba.")
            else:st.error(f"Se detectaron {overlap} solapamientos de identificadores.")
        st.markdown("- La fecha de corte excluye el mes de intervención y lecturas futuras.\n- Los identificadores sólo sirven para unir y separar registros.\n- La selección termina antes de evaluar la prueba final.\n- El preprocesador se ajusta con entrenamiento y se reutiliza en inferencia.\n- Los archivos nuevos y las notas de inspección se mantienen en la sesión.\n- La aplicación no reentrena al subir datos.")
