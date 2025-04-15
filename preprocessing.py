import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def parse_duration(duration_str):
    try:
        if duration_str is None or duration_str == "NaN" or pd.isna(duration_str):
            return np.nan
        if isinstance(duration_str, float) and np.isnan(duration_str):
            return np.nan
        h, m, s = [int(x) for x in duration_str.split(":")]
        return h * 60 + m + s / 60
    except Exception:
        return np.nan

import numpy as np

def intensity_factor(duration):
    # Sigmoid ramp: ramps up near 90min
    return 2 / (1 + np.exp(-0.05 * (duration - 90)))

def preprocess(df: pd.DataFrame, use_jef_mode=False):
    df = df.copy()
    warnings = []
    # Parse dates
    df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    # Parse durations
    df['Duur_Training_min'] = df['Duur_Training'].apply(parse_duration)
    df['Duur_spelen_min'] = df['Duur_spelen'].apply(parse_duration) if 'Duur_spelen' in df else np.nan
    # Parse numerics
    df['Training_Intensity'] = pd.to_numeric(df['Training_Intensity'], errors='coerce')
    df['Overall_fatigue'] = pd.to_numeric(df['Overall_fatigue'], errors='coerce')
    df['Vinger_fatigue_tijdens_het_spelen'] = pd.to_numeric(df['Vinger_fatigue_tijdens_het_spelen'], errors='coerce')
    df['Vinger_pijn_stijfheid'] = pd.to_numeric(df['Vinger_pijn_stijfheid'], errors='coerce')
    # Feature engineering
    df['Week'] = df['Datum'].dt.isocalendar().week
    df['Training_Load'] = df['Duur_Training_min'] * df['Training_Intensity']
    df['JML'] = df['Duur_Training_min'] * intensity_factor(df['Duur_Training_min'])
    df['Pain_Flag'] = df['Vinger_pijn_stijfheid'] > 3
    df['Recovery_Effectiveness'] = df['Recovery_actions_taken'].apply(lambda x: 1 if isinstance(x, str) and x != "NaN" else 0)
    # Error handling: drop rows with missing critical data
    initial = len(df)
    df = df.dropna(subset=['Datum', 'Duur_Training_min', 'Training_Intensity'])
    if len(df) < initial:
        warnings.append(f"{initial-len(df)} rows dropped due to missing date, duration, or intensity.")
    # Warn about suspiciously high/low values
    if (df['Duur_Training_min'] > 600).any():
        warnings.append("Some durations > 10 hours. Check for data entry errors.")
    if (df['Training_Load'] > 2000).any():
        warnings.append("Some training loads > 2000. These are filtered in plots.")
    # Choose which load metric is active
    if use_jef_mode:
        df['Active_Load'] = df['JML']
        warnings.append("Jef Mode enabled: Using nonlinear Jef Mental Load (JML).")
    else:
        df['Active_Load'] = df['Training_Load']
    return df, warnings
