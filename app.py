from __future__ import annotations
import streamlit as st
from dashboard import data,theme,operations,research,upload

st.set_page_config(page_title="Volt Patrol | Inteligencia de inspección",page_icon="⚡",layout="wide",initial_sidebar_state="auto")
theme.install()
PAGES=["Centro de control","Priorizar inspecciones","Investigar suministro","Métodos observados","Red y pérdidas","Laboratorio de modelos","Evaluar nueva data","Calidad y trazabilidad"]
bundle=data.model()
ranking=data.frame("outputs/VOLT_PATROL_RANKING_COMPLETO.csv")
with st.sidebar:
    st.markdown('<div class="vp-logo"><div class="vp-mark">ϟ</div><div class="vp-wordmark">VOLT PATROL<br><small>INTELIGENCIA DE INSPECCIÓN</small></div></div>',unsafe_allow_html=True)
    if st.session_state.get("navigation") not in PAGES:st.session_state["navigation"]=PAGES[0]
    page=st.radio("Espacio de trabajo",PAGES,key="navigation",label_visibility="collapsed")
    st.markdown('<div class="vp-kicker">PLANIFICACIÓN DE CAMPO</div>',unsafe_allow_html=True)
    budget=st.slider("Capacidad de la próxima ronda",min_value=10,max_value=200,value=76,step=1,key="inspection_budget")
    st.caption("Define cuántos suministros priorizar. Puedes elegir casos manualmente en el plan.")
    version=bundle.get("version","—") if bundle else "—"
    date=bundle.get("trained_at","")[:10] if bundle else "—"
    st.markdown(f'<div class="vp-foot">MODELO {version}<br>Entrenado {date}<br>Datos oficiales · corte 2025</div>',unsafe_allow_html=True)
st.markdown('<div class="vp-topline"><span>OPERACIONES / DISTRIBUCIÓN ELÉCTRICA</span><span class="vp-pill">● VOLT PATROL 2.0</span></div>',unsafe_allow_html=True)
if bundle is None:
    theme.heading("VOLT PATROL","Listo para conectar el modelo.","Este entorno todavía no dispone del paquete de ejecución.")
    st.info("Comprueba que runtime/models/final_model.joblib está incluido en la rama desplegada. Para regenerarlo localmente: python main.py y python -m src.package_runtime.")
elif ranking.empty and page!="Evaluar nueva data":
    st.info("No hay un ranking oficial local. Puedes evaluar un archivo desde Evaluar nueva data.")
else:
    views={PAGES[0]:operations.overview,PAGES[1]:operations.planner,PAGES[2]:operations.investigate,
           PAGES[3]:research.methods,PAGES[4]:research.network,PAGES[5]:research.models,
           PAGES[6]:upload.render,PAGES[7]:research.quality}
    views[page](ranking,bundle,budget)
    st.markdown('<div class="vp-foot">VOLT PATROL · Priorización con evidencia. La confirmación de una vulneración requiere una inspección. Los índices del modelo no son probabilidades calibradas.</div>',unsafe_allow_html=True)
