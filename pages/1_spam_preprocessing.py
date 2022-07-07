import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Spam Preprocessing", page_icon="📊")

st.markdown("# Spam Preprocessing")
st.sidebar.header("Spam Preprocessing")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)
