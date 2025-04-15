import pandas as pd
import json
from typing import Tuple
import streamlit as st

def load_json(file) -> pd.DataFrame:
    if isinstance(file, str):
        with open(file, encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json.load(file)
    return pd.DataFrame(data)

def file_uploader():
    uploaded_file = st.sidebar.file_uploader("Upload JSON Log", type=["json"])
    if uploaded_file:
        df = load_json(uploaded_file)
        return df
    return None
