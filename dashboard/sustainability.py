from __future__ import annotations
import pandas as pd
import streamlit as st
from dashboard import data,theme
from dashboard.components import number
from src.monitoring import drift,scenario


def render(ranking,bundle,budget):
    theme.heading("11","Escala, seguimiento y beneficio.","Evidencia de capacidad, pruebas entre zonas y herramientas para revisar el modelo con resultados de campo.")
    scale,validation,monitor,cost=st.tabs(["Escala y cobertura","Generalización y referencias","Monitoreo","Costo y beneficio"])
    with scale:
        report=data.metadata("reports/scale_benchmark.json")
        if report:
            a,b,c=st.columns(3)
            with a:theme.metric("Suministros procesados",number(report["rows"]))
            with b:theme.metric("Tiempo completo",f"{report['seconds']:.1f} s")
            with c:theme.metric("Pico de RAM muestreado",f"{report['peak_rss_mb']:.0f} MB")
            st.write("Lectura CSV por lotes, mismo pipeline de inferencia, comprobación global de duplicados y ordenamiento global en disco. Incluye escritura del ranking completo.")
            st.caption("Prueba de carga con consumos replicados e identificadores sintéticos. Mide capacidad del equipo documentado, no precisión predictiva ni capacidad garantizada de Streamlit Cloud.")
            with st.expander("Entorno y medición"):st.json(report)
        st.code("python -m src.batch entrada.csv ranking.csv --year 2025 --cutoff 2026-01-01 --batch-size 10000",language="bash")
        st.write("Los archivos grandes se procesan por lotes en el equipo de operación. La carga interactiva de Streamlit mantiene su límite de 100 MB por archivo.")
        st.caption("Reproducir la medición: python -m src.benchmark_scale. Los archivos de carga son sintéticos, quedan fuera de Git y no participan en el entrenamiento.")
        st.subheader("Suministros sin historial")
        st.write("Se conservan en la población, se identifican como insuficientes y pueden recibir visitas exploratorias con rotación por SED en Campañas mensuales. No se les asigna una probabilidad de hurto sin evidencia.")
        cold=data.metadata("reports/cold_start_validation.json")
        if cold:
            st.dataframe(pd.DataFrame(cold["cases"]),hide_index=True,width="stretch")
            st.caption(cold["limitation"])
    with validation:
        summary=data.metadata("reports/robustness_summary.json")
        geo=data.frame("reports/geographic_validation.csv")
        temporal=data.frame("reports/rolling_validation.csv")
        rules=data.frame("reports/operational_baselines.csv")
        if not geo.empty:
            st.subheader("SED retenidas fuera del entrenamiento")
            st.dataframe(geo,hide_index=True,width="stretch")
        if not temporal.empty:
            st.subheader("Evaluación por periodos posteriores")
            st.dataframe(temporal,hide_index=True,width="stretch")
        if not rules.empty:
            st.subheader("Comparación con reglas sencillas")
            st.dataframe(rules,hide_index=True,width="stretch")
        if summary:st.info(summary["limitation"])
        st.caption("Las reglas son referencias reproducibles, no una descripción del proceso actual de EQUANS. Las métricas con población sin etiqueta son observables y no precisión real de hurto.")
    with monitor:
        current=st.session_state.get("upload_result")
        if current is None:st.info("Evalúa un archivo en «Evaluar nueva data» para comparar su distribución con la referencia 2025.")
        else:
            first=1-float(ranking.valid_prediction.mean());second=1-float(current.valid_prediction.mean())
            a,b=st.columns(2)
            with a:st.metric("Sin score · referencia",f"{first:.1%}")
            with b:st.metric("Sin score · nueva población",f"{second:.1%}",delta=f"{second-first:+.1%}",delta_color="inverse")
            result=drift(ranking,current)
            if result.empty:st.info("Se necesitan al menos 20 valores válidos y variación suficiente para comparar distribuciones.")
            else:st.dataframe(result,hide_index=True,width="stretch")
            st.caption("PSI usa intervalos definidos en la referencia. El umbral 0,20 es una regla configurable de revisión; no demuestra caída de precisión ni activa reentrenamiento automático.")
        st.markdown("1. Revisar cobertura y cambios de esquema cada mes.\n2. Contrastar resultados concluyentes de visitas y su cobertura.\n3. Investigar cambios de población antes de reentrenar.\n4. Entrenar un candidato con nuevos resultados verificados y un periodo reservado.\n5. Comparar Top 76, estabilidad, cobertura y costo con la versión vigente.\n6. Publicar modelo, datos y manifiesto juntos; conservar la versión anterior para reversión.")
    with cost:
        st.write("Introduce supuestos propios. Esta simulación no usa la tasa histórica P/U como si fuera precisión real de campo.")
        a,b=st.columns(2)
        with a:
            visits=int(st.number_input("Visitas del escenario",min_value=1,value=budget))
            hit=st.number_input("Tasa de acierto supuesta (%)",min_value=0.,max_value=100.,value=None)
            price=st.number_input("Costo por visita",min_value=0.,value=None)
        with b:
            recovery=st.number_input("Recupero monetario por hallazgo",min_value=0.,value=None)
            operation=st.number_input("Costo mensual de operación del sistema",min_value=0.,value=None)
        st.caption("Usa una misma moneda. Los valores están vacíos hasta que el operador aporte sus supuestos.")
        if all(value is not None for value in (hit,price,recovery,operation)):
            result=scenario(visits,hit/100,price,recovery,operation)
            a,b,c=st.columns(3)
            with a:st.metric("Hallazgos esperados · escenario",number(result["expected_hits"],1))
            with b:st.metric("Beneficio neto · escenario",number(result["expected_net"],2))
            with c:st.metric("Tasa de equilibrio",f"{result['break_even_hit_rate']:.1%}" if result["break_even_hit_rate"] is not None else "No calculable")
        st.caption("Los costos, recuperos y tasas observadas se registran por visita en Campañas mensuales, separados de estos supuestos.")
