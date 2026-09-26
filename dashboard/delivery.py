from __future__ import annotations
import pandas as pd
import streamlit as st
from dashboard import data,theme
from dashboard.components import downloads,number,ranking_table
from dashboard.inference import read_upload,to_csv
from src.monitoring import evaluate_top


def render(ranking,bundle,budget):
    theme.heading("10","Una lista. Un criterio.","Ranking único para la entrega: mayor prioridad primero, sin suministros duplicados y con trazabilidad del año utilizado.")
    final=data.frame("outputs/delivery/RANKING_AUDITABLE.csv")
    manifest=data.metadata("reports/delivery_manifest.json")
    if final.empty:st.info("El paquete de entrega todavía no está disponible.");return
    a,b,c,d=st.columns(4)
    with a:theme.metric("Suministros únicos",number(len(final)))
    with b:theme.metric("Primeras posiciones","76")
    with c:theme.metric("Hurtos del conjunto ciego","76","Cantidad informada en la lámina del concurso; las identidades no están disponibles")
    with d:theme.metric("Precisión oficial @76","Por evaluar")
    theme.brief("Se unen 2024 y 2025 usando el año más reciente de cada suministro.",
        f"{manifest.get('shared_between_years',0):,} suministros aparecen en ambos años. Los casos sin score se conservan al final y su orden no se interpreta como riesgo. El score es relativo, no una probabilidad calibrada.")
    st.caption("Sin plantilla oficial proporcionada, el CSV de entrega contiene una columna SUMINISTRO_ID; el orden de las filas define el ranking. Se conservan alternativas por año para una eventual aclaración de las bases.")
    a,b=st.columns(2)
    with a:st.download_button("Descargar ranking de entrega",to_csv(final[["SUMINISTRO_ID"]]),"ENTREGA_RANKING.csv","text/csv",type="primary",on_click="ignore")
    with b:st.download_button("Descargar primeros 76",to_csv(final.head(76)[["SUMINISTRO_ID"]]),"ENTREGA_TOP_76.csv","text/csv",on_click="ignore")
    ranking_table(final.head(76),height=420)
    with st.expander("Auditar población, años y casos sin historial"):
        st.json(manifest)
        downloads(final,"delivery_audit","ranking_auditable")
        for year in (2024,2025):
            alternative=data.frame(f"outputs/delivery/ALTERNATIVA_{year}.csv")
            st.download_button(f"Alternativa sólo {year}",to_csv(alternative),f"RANKING_{year}.csv","text/csv",on_click="ignore")
    with st.expander("Calcular precisión cuando existan etiquetas reales"):
        st.write("Sube resultados con SUMINISTRO_ID y HURTO_REAL (1 = hurto confirmado; 0 = ausencia verificada). Las etiquetas desconocidas no se convierten en ceros.")
        truth=st.file_uploader("Resultados reales",type=["csv","xlsx"],key="delivery_truth")
        if truth:
            try:
                report=evaluate_top(final,read_upload(truth.name,truth.getvalue()))
                a,b,c=st.columns(3)
                with a:st.metric("Top 76 con resultado",f"{report['resolved']}/76")
                with b:st.metric("Hurtos confirmados",report["confirmed"])
                with c:st.metric("Precisión @76",f"{report['precision_at_k']:.1%}" if report["precision_at_k"] is not None else "Pendiente de etiquetas")
                st.caption("Resultado calculado con el archivo aportado; debe corresponder a las etiquetas verificadas del concurso. No se modifica el ranking ni se reentrena.")
                if report["labels_outside_ranking"]:st.warning(f"{report['labels_outside_ranking']} etiquetas corresponden a suministros fuera de esta lista.")
                baseline=st.file_uploader("Comparar con la lista del procedimiento actual (CSV ordenado con SUMINISTRO_ID)",type=["csv"],key="current_process_list")
                if baseline:
                    existing=read_upload(baseline.name,baseline.getvalue())
                    if "SUMINISTRO_ID" not in existing or existing.SUMINISTRO_ID.duplicated().any():
                        st.error("La lista actual debe incluir SUMINISTRO_ID sin duplicados.")
                    elif not existing.SUMINISTRO_ID.isin(final.SUMINISTRO_ID).all():
                        st.error("Compara listas sobre la misma población de suministros.")
                    else:
                        old=evaluate_top(existing,read_upload(truth.name,truth.getvalue()))
                        st.dataframe(pd.DataFrame([{"método":"Volt Patrol",**report},{"método":"Procedimiento actual aportado",**old}]),hide_index=True,width="stretch")
                        if old["precision_at_k"] is not None and report["precision_at_k"] is not None:
                            st.metric("Diferencia observada en aciertos @76",f"{report['confirmed']-old['confirmed']:+d}")
                        else:st.caption("Faltan resultados de alguno de los Top 76; todavía no se puede afirmar una mejora de precisión.")
            except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
