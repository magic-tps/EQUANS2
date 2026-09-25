from __future__ import annotations

import plotly.express as px
import pandas as pd


def score_histogram(ranking: pd.DataFrame):
    return px.histogram(ranking.dropna(subset=["priority_score"]),x="priority_score",nbins=35,
                        labels={"priority_score":"Puntaje de prioridad"},
                        title="Distribución del puntaje")


def monthly_chart(frame: pd.DataFrame, supply: str):
    from src.preprocess import monthly_long
    selected=frame[frame.SUMINISTRO_ID.astype(str).eq(str(supply))]
    if selected.empty:return None
    long=monthly_long(selected,2025)
    return px.line(long,x="nominal_month",y="daily_kwh",markers=True,
                   labels={"nominal_month":"Mes","daily_kwh":"kWh/día"},
                   title="Consumo diario observado")
