from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.evidence import feature_label

TEAL="#67e2c4";AMBER="#f2b968";BLUE="#779fef";MUTED="#8ba1bc"
MONTHS=["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]


def polish(fig,height=320):
    fig.update_layout(height=height,margin=dict(l=8,r=12,t=26,b=16),paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Segoe UI, Arial",size=12,color="#a9bdd3"),
        hoverlabel=dict(bgcolor="#14263c",font_color="#e4f2ff"),
        legend=dict(orientation="h",y=1.10,x=0,title=None,font_size=11),
        colorway=[TEAL,BLUE,AMBER,"#c19aeb"],bargap=.22)
    fig.update_xaxes(showgrid=False,zeroline=False,title_font_size=11)
    fig.update_yaxes(gridcolor="#25364b",zeroline=False,title_font_size=11)
    return fig


def draw(fig,key=None):st.plotly_chart(fig,width="stretch",config={"displayModeBar":False},key=key)


def score_histogram(ranking):
    values=ranking.loc[ranking.valid_prediction,"priority_score"]*100
    fig=go.Figure(go.Histogram(x=values,nbinsx=28,marker_color=TEAL,marker_line_width=0,
        hovertemplate="Índice %{x:.1f}<br>%{y} suministros<extra></extra>"))
    fig.update_layout(xaxis_title="Índice del modelo · 0 a 100",yaxis_title="Suministros")
    return polish(fig,285)


def sed_concentration(ranking,limit=76):
    if "SED_ID" not in ranking:return None
    counts=ranking[ranking.valid_prediction].head(limit).groupby("SED_ID").size().nlargest(9).sort_values()
    fig=go.Figure(go.Bar(x=counts.values,y=counts.index,orientation="h",marker_color=BLUE,
        text=counts.values,textposition="outside",hovertemplate="%{y}<br>%{x} inspecciones<extra></extra>"))
    fig.update_layout(xaxis_title="Suministros en la ronda",yaxis_title=None)
    return polish(fig,285)


def consumption_curve(long,metric="daily_kwh",peers=None):
    label="kWh / día" if metric=="daily_kwh" else "Consumo mensual · kWh"
    fig=go.Figure()
    if peers is not None and not peers.empty:
        fig.add_trace(go.Scatter(x=peers.nominal_month,y=peers.peer_daily,mode="lines",name="Mediana de pares",
                                line=dict(color=BLUE,width=2,dash="dot")))
    for year,frame in long.groupby(long.nominal_month.dt.year):
        fig.add_trace(go.Scatter(x=frame.nominal_month,y=frame[metric],mode="lines+markers",name=str(year),
            connectgaps=False,line=dict(width=3,color=TEAL if year==2025 else MUTED),marker=dict(size=6),
            hovertemplate="%{x|%b %Y}<br>%{y:.2f} "+label+"<extra></extra>"))
    fig.update_layout(yaxis_title=label,xaxis_title=None,hovermode="x unified")
    return polish(fig,335)


def year_comparison(long):
    frame=long.copy();frame["Año"]=frame.nominal_month.dt.year.astype(str)
    frame["Mes"]=frame.nominal_month.dt.month.map(dict(enumerate(MONTHS,1)))
    fig=px.bar(frame,x="Mes",y="daily_kwh",color="Año",barmode="group",category_orders={"Mes":MONTHS},
        color_discrete_sequence=["#526a88",TEAL],labels={"daily_kwh":"kWh / día"})
    return polish(fig,290)


def heatmap(ranking,monthly,count=16):
    chosen=ranking[ranking.valid_prediction].head(count).SUMINISTRO_ID.tolist()
    frame=monthly[monthly.SUMINISTRO_ID.isin(chosen)&monthly.nominal_month.dt.year.eq(2025)].copy()
    frame["month"]=frame.nominal_month.dt.month
    pivot=frame.pivot(index="SUMINISTRO_ID",columns="month",values="daily_kwh").reindex(index=chosen,columns=range(1,13))
    med=pivot.median(axis=1).replace(0,np.nan)
    normalized=pivot.div(med,axis=0).clip(upper=2.5)
    fig=go.Figure(go.Heatmap(z=normalized.values,x=MONTHS,y=chosen,zmin=0,zmax=2.5,
        colorscale=[[0,"#edb366"],[.35,"#34514f"],[.4,"#21444a"],[.75,"#308e89"],[1,"#78e3cc"]],
        xgap=3,ygap=3,colorbar=dict(title="Ratio",thickness=9,len=.8),
        hovertemplate="%{y} · %{x}<br>%{z:.2f} × mediana personal<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return polish(fig,125+count*22)


def shap_chart(row):
    series=row.drop(labels=["SUMINISTRO_ID"],errors="ignore").astype(float)
    keys=series.abs().nlargest(9).index
    values=series.loc[keys].sort_values()
    fig=go.Figure(go.Bar(x=values.values,y=[feature_label(k) for k in values.index],orientation="h",
        marker_color=[AMBER if x>0 else BLUE for x in values.values],
        hovertemplate="%{y}<br>Aporte %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="Aporte al componente LightGBM · escala interna",yaxis_title=None)
    return polish(fig,360)


def energy_chart(balance):
    fig=go.Figure()
    for col,label,color in [("E. DISTRIBUIDA MES MWh","Distribuida",BLUE),("E. FACTURADA MES MWh","Facturada",TEAL)]:
        fig.add_trace(go.Scatter(x=balance["PERIODO BALANCE"],y=balance[col],name=label,mode="lines+markers",line=dict(color=color,width=2.5)))
    fig.update_layout(yaxis_title="Energía · MWh",hovermode="x unified")
    return polish(fig,330)


def recovery_curve(curve):
    fig=go.Figure(go.Scatter(x=curve.position,y=curve.known_hits,mode="lines",name="Modelo",line=dict(color=TEAL,width=3)))
    fig.add_trace(go.Scatter(x=curve.position,y=curve.random_reference,mode="lines",name="Referencia aleatoria",line=dict(color=MUTED,dash="dot")))
    fig.update_layout(xaxis_title="Primeras posiciones del ranking de prueba",yaxis_title="Positivos conocidos recuperados")
    return polish(fig,310)
