import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Spam Features Extration")

st.markdown("# Features Extration with TF-IDF")
st.sidebar.header("Features Extration with TF-IDF")
st.write(
    """Features Extration with TF-IDF"""
)
