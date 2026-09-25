from __future__ import annotations

import streamlit as st


def badge(label: str, value: object) -> None:
    st.metric(label, value if value is not None else "No disponible")
