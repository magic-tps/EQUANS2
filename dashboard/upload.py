from __future__ import annotations
import hashlib
import pandas as pd
import streamlit as st
from dashboard import theme,charts
from dashboard.components import number,downloads,ranking_table
from dashboard.inference import read_upload,infer
from dashboard.validators import validate
from src.preprocess import monthly_long


def clear_upload():
    for key in ("upload_parsed","upload_result","upload_fingerprint","upload_content_hash"):
        st.session_state.pop(key,None)
    st.session_state["uploader_generation"]=st.session_state.get("uploader_generation",0)+1


def render(ranking,bundle,budget):
    theme.heading("07","Tu archivo. El mismo modelo.","Valida las lecturas, revisa la cobertura y genera una nueva lista de inspección. El modelo entrenado se reutiliza sin reentrenar.")
    st.caption("Los archivos y resultados de esta pantalla se conservan en la sesión, sin guardarlos de forma permanente ni compartirlos en caché global.")
    left,right=st.columns([2,1],gap="large")
    with left:
        uploaded=st.file_uploader("Arrastra un Excel o CSV de suministros",type=["xlsx","csv"],key=f"upload_{st.session_state.get('uploader_generation',0)}")
    with right:
        year=int(st.number_input("Año de los meses",min_value=2020,max_value=2100,value=2025,key="upload_year"))
        st.caption("Esquema: suministro + consumo, días y fecha de lectura por mes. La SED es opcional para calcular el score.")
    if uploaded is None:
        for key in ("upload_parsed","upload_result","upload_fingerprint","upload_content_hash"):st.session_state.pop(key,None)
        with st.expander("Qué debe contener el archivo"):
            st.markdown("- Una fila por suministro, con `SUMINISTRO_ID` único.\n- Meses identificados por nombre: por ejemplo `CONSUMO OCTUBRE`.\n- Días facturados y fecha de lectura del mismo mes.\n- Tres meses de calendario recientes completos para cada score.\n- Los registros insuficientes también aparecen en el ranking descargable.")
        return
    try:
        content=uploaded.getvalue();content_hash=hashlib.sha256(content).hexdigest()
        if st.session_state.get("upload_content_hash")!=content_hash:
            parsed=read_upload(uploaded.name,content)
            st.session_state["upload_parsed"]=parsed
            st.session_state["upload_content_hash"]=content_hash
            st.session_state.pop("upload_result",None)
        frame=st.session_state["upload_parsed"]
        errors,info=validate(frame)
        a,b,c,d=st.columns(4)
        for column,label,value in [(a,"Filas",len(frame)),(b,"Suministros",info["supplies"]),(c,"Columnas",info["columns"]),(d,"Meses completos",len(info["months"]))]:
            with column:theme.metric(label,number(value))
        st.caption(f"{uploaded.name} · {len(content)/1024:.1f} KB")
        with st.expander("Vista previa y validación",expanded=bool(errors)):
            st.dataframe(frame.head(8),hide_index=True,width="stretch")
            st.write("Columnas detectadas:",list(frame.columns))
            for error in errors:st.error(error)
            for warning in info["warnings"]:st.warning(warning)
        if errors:return
        monthly=monthly_long(frame,year)
        observed=monthly.loc[monthly.daily_kwh.notna(),"nominal_month"]
        default_cut=observed.max()+pd.offsets.MonthBegin(1) if not observed.empty else pd.Timestamp(year+1,1,1)
        cutoff=st.date_input("Fecha de corte para la evaluación",value=default_cut.date(),key=f"cutoff_{content_hash[:8]}_{year}",
            help="Se usan únicamente meses anteriores al mes del corte y lecturas anteriores a esta fecha")
        st.caption(f"El modelo buscará los tres meses completos anteriores a {pd.Timestamp(cutoff):%d/%m/%Y}.")
        fingerprint=hashlib.sha256(f"{content_hash}|{year}|{cutoff}|{bundle.get('trained_at')}".encode()).hexdigest()
        if st.session_state.get("upload_fingerprint")!=fingerprint:st.session_state.pop("upload_result",None)
        a,b=st.columns([2,1])
        with a:run=st.button("Validar y generar ranking",type="primary",width="stretch")
        with b:st.button("Borrar archivo y resultado",on_click=clear_upload,width="stretch")
        if run:
            bar=st.progress(.1,text="Modelo cargado · validando el archivo")
            result=infer(bundle,frame,year,cutoff=cutoff,progress=lambda fraction,message:bar.progress(fraction,text=message))
            st.session_state["upload_result"]=result
            st.session_state["upload_fingerprint"]=fingerprint
        result=st.session_state.get("upload_result")
        if result is not None:
            st.success(f"Ranking listo: {int(result.valid_prediction.sum())} scores y {int((~result.valid_prediction).sum())} registros insuficientes. Se conservaron los {len(result)} suministros.")
            ranking_table(result.head(100),height=420)
            downloads(result,"new_ranking","ranking_nuevo")
            with st.expander(f"Descargar primera ronda de {budget} suministros"):
                downloads(result[result.valid_prediction].head(budget),"new_top","primera_ronda")
            charts.draw(charts.score_histogram(result),"upload_scores")
    except (ValueError,KeyError,TypeError,OSError) as exc:
        st.error(f"No se pudo evaluar el archivo: {exc}")
    except Exception:
        st.error("No se pudo leer este archivo. Comprueba que sea un Excel o CSV válido y que no esté dañado.")
