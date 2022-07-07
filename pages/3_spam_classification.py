import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Spam Classification")

st.markdown("# Spam Classification with Multinominal Naive Bayes")
st.sidebar.header("Spam Classification with Multinominal Naive Bayes")
st.write(
    "Spam Classification with Multinominal Naive Bayes"
)
