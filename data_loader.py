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
    
    # Handle JefDagboek format
    if 'Form responses 1' in data:
        form_responses = pd.DataFrame(data['Form responses 1'])
        formatted_data = pd.DataFrame(data['Formatted Form Data'])
        
        # Merge the dataframes
        df = pd.concat([form_responses, formatted_data], ignore_index=True)
        
        # Convert Timestamp to datetime
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Convert Datum to datetime
        if 'Datum' in df.columns:
            df['Datum'] = pd.to_datetime(df['Datum'])
        
        return df
    else:
        return pd.DataFrame(data)

def file_uploader():
    uploaded_file = st.sidebar.file_uploader("Upload JSON Log", type=["json"])
    if uploaded_file:
        df = load_json(uploaded_file)
        return df
    return None
