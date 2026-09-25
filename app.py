from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.charts import monthly_chart, score_histogram
from dashboard.components import badge
from dashboard.inference import infer,read_upload,to_xlsx
from dashboard.validators import validate

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Volt Patrol",page_icon="⚡",layout="wide")
st.markdown("""<style>
.block-container {max-width: 1450px; padding-top: 1.8rem}
h1,h2,h3 {letter-spacing: -.025em}
[data-testid="stMetric"] {background:#102234;padding:1rem;border-radius:12px;border:1px solid #1c3b50}
</style>""",unsafe_allow_html=True)


@st.cache_resource
def load_model():
    path=ROOT/"models/final_model.joblib"
    return joblib.load(path) if path.exists() else None


def load_ranking():
    path=ROOT/"outputs/VOLT_PATROL_RANKING_COMPLETO.csv"
    return pd.read_csv(path,dtype={"SUMINISTRO_ID":str}) if path.exists() else None


st.title("⚡ VOLT PATROL")
st.caption("Priorización de inspecciones eléctricas basada en patrones de consumo")
page=st.sidebar.radio("Navegación",["Dashboard","Ranking","Analizar suministro","Pérdidas por SED","Modelos","Evaluar nueva data","Acerca del modelo"])
bundle=load_model()
ranking=load_ranking()

if page=="Dashboard":
    st.header("Panorama operativo")
    if ranking is None:
        st.info("Ejecuta `python main.py` con los siete Excel autorizados para generar el ranking local.")
    else:
        cols=st.columns(4)
        with cols[0]:badge("Suministros objetivo",len(ranking))
        with cols[1]:badge("Predicciones válidas",int(ranking.valid_prediction.sum()))
        with cols[2]:badge("Datos insuficientes",int((~ranking.valid_prediction).sum()))
        with cols[3]:badge("Versión",bundle.get("version") if bundle else None)
        if bundle:
            cols=st.columns(4)
            with cols[0]:badge("Entrenamiento",bundle.get("trained_at","")[:10])
            with cols[1]:badge("Positivos históricos",bundle.get("positive_count"))
            with cols[2]:badge("Hits@76 observado",bundle.get("validation",{}).get("hits_76"))
            with cols[3]:badge("Recall conocido@76",f"{bundle['validation'].get('recall_known_76',0):.1%}")
        st.plotly_chart(score_histogram(ranking),width="stretch")
        st.caption("El puntaje representa prioridad relativa; no es una probabilidad calibrada ni confirma una vulneración.")

elif page=="Ranking":
    st.header("Ranking de inspección")
    if ranking is None:st.info("El ranking local aún no está disponible.")
    else:
        n=st.selectbox("Mostrar",[10,25,50,76,100,200,"Personalizado"],index=3)
        if n=="Personalizado":n=st.number_input("Cantidad",1,len(ranking),76)
        search=st.text_input("Buscar suministro")
        status=st.multiselect("Estado de datos",sorted(ranking.data_status.unique()))
        view=ranking
        if search:view=view[view.SUMINISTRO_ID.str.contains(search,case=False,regex=False)]
        if status:view=view[view.data_status.isin(status)]
        view=view.head(int(n))
        st.dataframe(view,width="stretch",hide_index=True)
        st.download_button("Descargar CSV",view.to_csv(index=False).encode("utf-8-sig"),"top_n.csv","text/csv")
        st.download_button("Descargar Excel",to_xlsx(view),"top_n.xlsx")

elif page=="Analizar suministro":
    st.header("Análisis de suministro")
    if ranking is None:st.info("El ranking local aún no está disponible.")
    else:
        supply=st.selectbox("Suministro",ranking.SUMINISTRO_ID.tolist())
        row=ranking.set_index("SUMINISTRO_ID").loc[supply]
        cols=st.columns(3)
        with cols[0]:badge("Prioridad",int(row["rank"]) if pd.notna(row["rank"]) else None)
        with cols[1]:badge("Score",f"{row.priority_score:.3f}" if pd.notna(row.priority_score) else None)
        with cols[2]:badge("Estado",row.data_status)
        path=ROOT/"data/ALIMENTADOR_2025.xlsx"
        if path.exists():
            raw=pd.read_excel(path)
            chart=monthly_chart(raw,supply)
            if chart:st.plotly_chart(chart,width="stretch")
            selected=raw[raw.SUMINISTRO_ID.astype(str).eq(supply)]
            st.dataframe(selected[[c for c in ["SED_ID","TARIFA","TIPO CONEXIONADO"] if c in selected]],hide_index=True)
        st.caption("La señal indica prioridad de inspección. La interpretación individual no demuestra causalidad.")

elif page=="Pérdidas por SED":
    st.header("Contexto de pérdidas por SED")
    path=ROOT/"data/BALANCE_SED.xlsx"
    if not path.exists():st.info("Se requiere el Excel local autorizado para ver el contexto de SED.")
    else:
        balance=pd.read_excel(path)
        target_path=ROOT/"data/ALIMENTADOR_2025.xlsx"
        if target_path.exists():
            target_sed=set(pd.read_excel(target_path,usecols=["SED_ID"]).SED_ID.dropna().astype(str))
            choices=sorted(set(balance.SED_ID.dropna().astype(str)) & target_sed)
        else:
            choices=sorted(set(balance.SED_ID.dropna().astype(str)))
        sed=st.selectbox("SED",choices)
        view=balance[balance.SED_ID.eq(sed) & pd.to_datetime(balance["PERIODO BALANCE"]).lt(pd.Timestamp("2026-01-01"))].sort_values("PERIODO BALANCE")
        if view.empty:
            st.info("No hay balances anteriores al cierre de 2025 para esta SED.")
            st.stop()
        latest=view.iloc[-1]
        cols=st.columns(4)
        for col,label,key in zip(cols,["Distribuida MWh","Facturada MWh","Pérdida MWh","Pérdida %"],
                                 ["E. DISTRIBUIDA MES MWh","E. FACTURADA MES MWh","PERDIDA  MES MWh","PERDIDA MES %"]):
            with col:
                value=latest[key]
                badge(label,f"{value:.2%}" if key=="PERDIDA MES %" else f"{value:,.2f}")
        st.write("Proyección no técnica MWh:",latest["PROYEC_PERDIDAS NO TECNICAS MES MWh"])
        extra=[]
        power_path=ROOT/"data/POTENCIA_SED.xlsx"
        if power_path.exists():
            power=pd.read_excel(power_path)
            subset=power[power.SED_ID.eq(sed) & pd.to_numeric(power.iloc[:,-1],errors="coerce").le(2025)]
            if not subset.empty:
                item=subset.iloc[-1]
                extra.append(f"Potencia: {item['POTENCIA [KVA]']} kVA; clientes BT referenciales: {item['Clientes BT Cantidad Referencial']}")
        failures_path=ROOT/"data/FALLAS_SED.xlsx"
        if failures_path.exists():
            failures=pd.read_excel(failures_path)
            subset=failures[failures.SED_ID.eq(sed) & pd.to_numeric(failures.iloc[:,-1],errors="coerce").le(2025)]
            if not subset.empty:extra.append(f"Fallas registradas: {pd.to_numeric(subset.Cant_Fallas,errors='coerce').sum():.0f}")
        quality_path=ROOT/"data/CALIDAD_TENSION.xlsx"
        if quality_path.exists():
            quality=pd.read_excel(quality_path)
            extra.append(f"Eventos de calidad de tensión: {quality.SED_ID.eq(sed).sum()}")
        for item in extra:st.write(item)
        st.dataframe(view.tail(12),width="stretch",hide_index=True)
        plot=view.melt(id_vars="PERIODO BALANCE",value_vars=["E. DISTRIBUIDA MES MWh","E. FACTURADA MES MWh"],var_name="Serie",value_name="MWh")
        st.plotly_chart(px.line(plot,x="PERIODO BALANCE",y="MWh",color="Serie",markers=True),width="stretch")
        st.caption("Las pérdidas de una SED no son evidencia de vulneración individual.")

elif page=="Modelos":
    st.header("Evaluación experimental")
    path=ROOT/"reports/model_comparison.csv"
    if path.exists():
        comparison=pd.read_csv(path)
        st.dataframe(comparison,width="stretch",hide_index=True)
        st.plotly_chart(px.bar(comparison,x="model",y="hits_76",title="Casos conocidos recuperados entre los primeros 76"),width="stretch")
        if bundle:
            badge("Solapamiento Top 76 entre semillas",f"{bundle.get('top76_overlap',float('nan')):.1%}")
            with st.expander("Configuración entrenada"):
                st.json(bundle.get("config",{}))
    else:st.info("Ejecuta el entrenamiento para generar la comparación.")
    st.caption("Los registros sin confirmación no son negativos verificados. Las métricas son observables y dependen del conjunto histórico.")

elif page=="Evaluar nueva data":
    st.header("Evaluar nueva data")
    st.write("Carga un Excel o CSV con el esquema mensual de un alimentador. El modelo ya entrenado se utiliza sin reentrenar.")
    upload=st.file_uploader("Archivo",type=["xlsx","csv"])
    year=st.number_input("Año de los meses",2020,2100,2025)
    if upload is not None:
        st.caption(f"{upload.name} · {upload.size/1024:.1f} KB")
        try:
            frame=read_upload(upload.name,upload.getvalue())
            errors,info=validate(frame)
            st.write(f"{info['rows']} filas · {info['columns']} columnas · {info['supplies']} suministros · meses {info['months']}")
            with st.expander("Columnas detectadas"):
                st.write(list(frame.columns))
            for warning in info["warnings"]:st.warning(warning)
            st.dataframe(frame.head(5),width="stretch",hide_index=True)
            if errors:
                for error in errors:st.error(error)
            elif bundle is None:st.error("No se encuentra el modelo local entrenado.")
            elif st.button("Generar ranking",type="primary"):
                with st.spinner("Preprocesando, generando características y ejecutando inferencia..."):
                    output=infer(bundle,frame,int(year))
                st.success(f"Ranking generado: {int(output.valid_prediction.sum())} predicciones válidas")
                st.dataframe(output.head(100),width="stretch",hide_index=True)
                st.plotly_chart(score_histogram(output),width="stretch")
                st.download_button("Descargar ranking CSV",output.to_csv(index=False).encode("utf-8-sig"),"ranking.csv","text/csv")
                st.download_button("Descargar ranking Excel",to_xlsx(output),"ranking.xlsx")
        except Exception as exc:
            st.error(f"No se pudo procesar el archivo: {exc}")

else:
    st.header("Acerca del modelo")
    st.write("Volt Patrol ordena suministros para orientar inspecciones. Aprende de intervenciones confirmadas y de una población sin etiqueta. El score no está calibrado como probabilidad de fraude.")
    if bundle:
        st.json({k:bundle[k] for k in ("version","trained_at","positive_count","seed","model_input")})
    st.write("La validación separa eventos de 2024 y 2025. Como el alimentador objetivo no comparte suministros con el histórico, el desempeño real en ese alimentador sigue sin verificarse.")
