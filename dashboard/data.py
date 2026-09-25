"""Cache only the official local artifacts, never user uploads."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from src.preprocess import monthly_long
from src.artifacts import resolve

ROOT=Path(__file__).resolve().parents[1]


def signature(relative):
    path=resolve(ROOT,relative)
    return (str(path),path.stat().st_mtime_ns if path.exists() else 0)


@st.cache_resource(show_spinner=False,max_entries=2)
def _model(revision):
    path=resolve(ROOT,"models/final_model.joblib")
    return joblib.load(path) if path.exists() else None


def model():return _model(signature("models/final_model.joblib"))


@st.cache_data(show_spinner=False,max_entries=24)
def _frame(relative,revision):
    path=resolve(ROOT,relative)
    if not path.exists():return pd.DataFrame()
    if path.suffix==".xlsx":return pd.read_excel(path)
    return pd.read_csv(path)


def frame(relative):
    # Callers supply fixed application paths. Uploads never call this function.
    return _frame(relative,signature(relative))


def metadata(relative):
    path=resolve(ROOT,relative)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def summary():return metadata("reports/model_summary.json")


@st.cache_data(show_spinner=False,max_entries=2)
def _monthly(revision24,revision25):
    parts=[]
    for year in (2024,2025):
        raw=frame(f"data/ALIMENTADOR_{year}.xlsx")
        if not raw.empty:parts.append(monthly_long(raw,year))
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def monthly():
    return _monthly(signature("data/ALIMENTADOR_2024.xlsx"),signature("data/ALIMENTADOR_2025.xlsx"))
