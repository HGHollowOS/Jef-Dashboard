import streamlit as st
import data_loader
import preprocessing
import visualizations
import nlp_utils

st.set_page_config(page_title="Training & Music Dashboard", layout="wide", initial_sidebar_state="expanded", page_icon="📊")
st.markdown("""
<style>
body { background-color: #18181b; color: #e5e7eb; }
.reportview-container { background: #18181b; }
.css-1d391kg { background: #18181b; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Training, Recovery & Music Log Dashboard")
st.sidebar.header("Options")

# Automatically load JefDagboek.json
df = data_loader.load_json("JefDagboek.json")

if df is not None:
    toggle_jef_mode = st.sidebar.toggle("Enable Jef Mode (Nonlinear Load)", value=False, help="Use nonlinear Jef Mental Load (JML) instead of classic Training Load.")
    df, warnings = preprocessing.preprocess(df, use_jef_mode=toggle_jef_mode)
    st.sidebar.success(f"Loaded {len(df)} entries.")
    load_label = "Jef Mental Load (JML)" if toggle_jef_mode else "Training Load"
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Training Overview",
        "Fatigue & Recovery",
        "Music Analysis",
        "Correlations",
        "Subjective Notes (NLP)"
    ])
    with tab1:
        visualizations.training_overview_tab(df, warnings=warnings, load_label=load_label)
    with tab2:
        visualizations.fatigue_recovery_tab(df, load_label=load_label)
    with tab3:
        visualizations.music_analysis_tab(df)
    with tab4:
        visualizations.correlations_tab(df, load_label=load_label)
    with tab5:
        visualizations.subjective_notes_tab(df, nlp_utils)
    st.sidebar.subheader("Export")
    st.sidebar.download_button("Export Filtered CSV", df.to_csv(index=False), file_name="filtered_log.csv", mime="text/csv")
    st.sidebar.download_button("Export Filtered JSON", df.to_json(orient="records", force_ascii=False), file_name="filtered_log.json", mime="application/json")

# --- Advanced ML/Recommendation (optional) ---
# from sklearn.ensemble import RandomForestRegressor
# # Predict fatigue or pain tomorrow using last 3 days
# # Add code here for ML model and recommendations
