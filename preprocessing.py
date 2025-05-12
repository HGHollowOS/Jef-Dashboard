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
    if 'Datum' in df.columns:
        df['Datum'] = pd.to_datetime(df['Datum'], errors='coerce')
    # Parse durations
    df['duur_training_min'] = df['Duur Training'].apply(parse_duration)
    df['duur_spelen_min'] = df['duur_spelen'].apply(parse_duration) if 'duur_spelen' in df else np.nan
    
    # Convert column names to snake_case
    df.columns = df.columns.str.lower() \
        .str.replace(' ', '_') \
        .str.replace('/', '_') \
        .str.replace('(', '') \
        .str.replace(')', '') \
        .str.replace('=', '_') \
        .str.replace('-', '_') \
        .str.replace('___', '_') \
        .str.replace('__', '_') \
        .str.strip('_')
    # Parse numerics
    df['training_intensity'] = pd.to_numeric(df['training_intensity'], errors='coerce')
    df['overall_fatigue'] = pd.to_numeric(df['overall_fatigue'], errors='coerce')
    df['vinger_fatigue_tijdens_het_spelen'] = pd.to_numeric(df['vinger_fatigue_tijdens_het_spelen'], errors='coerce')
    df['vinger_pijn_stijfheid'] = pd.to_numeric(df['vinger_pijn_stijfheid'], errors='coerce')
    # Feature engineering
    df['week'] = df['datum'].dt.isocalendar().week
    df['training_load'] = df['duur_training_min'] * df['training_intensity']
    df['jml'] = df['duur_training_min'] * intensity_factor(df['duur_training_min'])
    df['pain_flag'] = df['vinger_pijn_stijfheid'] > 3
    df['recovery_effectiveness'] = df['recovery_actions_taken'].apply(lambda x: 1 if isinstance(x, str) and x != "NaN" else 0)
    # Error handling: drop rows with missing critical data
    initial = len(df)
    df = df.dropna(subset=['datum', 'duur_training_min', 'training_intensity'])
    if len(df) < initial:
        warnings.append(f"{initial-len(df)} rows dropped due to missing date, duration, or intensity.")
    # Warn about suspiciously high/low values
    if (df['duur_training_min'] > 600).any():
        warnings.append("Some durations > 10 hours. Check for data entry errors.")
    if (df['training_load'] > 2000).any():
        warnings.append("Some training loads > 2000. These are filtered in plots.")
    # Choose which load metric is active
    if use_jef_mode:
        df['active_load'] = df['jml']
        warnings.append("Jef Mode enabled: Using nonlinear Jef Mental Load (JML).")
    else:
        df['active_load'] = df['training_load']
    return df, warnings
