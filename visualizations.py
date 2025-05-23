import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

def training_overview_tab(df, warnings=None, load_label="Training Load"):
    st.header("Training Overview")
    if warnings:
        for w in warnings:
            st.warning(w)
    st.subheader(f"Total Duration and {load_label}")
    try:
        agg_type = st.radio("Aggregate by", options=["Day", "Week"], horizontal=True)
        if agg_type == "Week":
            grouped = df.groupby('week').agg({
                'duur_training_min': 'sum',
                'active_load': 'sum',
                'datum': 'min'
            }).reset_index()
            x_col = 'week'
            hover = grouped['datum'].dt.strftime('%Y-%m-%d')
        else:
            grouped = df.groupby('datum').agg({
                'duur_training_min': 'sum',
                'active_load': 'sum'
            }).reset_index()
            x_col = 'datum'
            hover = grouped[x_col].astype(str)
        grouped = grouped[(grouped['duur_training_min'] < 1000) & (grouped['active_load'] < 2000)]
        if grouped.empty:
            st.info("No data available for the selected aggregation or after filtering.")
            return
        fig = px.line()
        fig.add_scatter(x=grouped[x_col], y=grouped['duur_training_min'], name='Duration (min)', mode='lines+markers', yaxis='y1', hovertext=hover)
        fig.add_scatter(x=grouped[x_col], y=grouped['active_load'], name=load_label, mode='lines+markers', yaxis='y2', hovertext=hover)
        fig.update_layout(
            yaxis=dict(title='Duration (min)', side='left'),
            yaxis2=dict(title=load_label, overlaying='y', side='right', showgrid=False),
            xaxis_title=x_col,
            legend=dict(x=0.01, y=0.99),
            hovermode='x unified',
            template='plotly_dark',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Duration and {load_label} are plotted on separate axes for clarity. Suspiciously high values are filtered.")
    except Exception as e:
        st.error(f"Failed to plot duration/load: {e}")
    st.subheader("Training Type Frequency")
    try:
        type_counts = df['type_training'].value_counts().reset_index()
        type_counts.columns = ['type_training', 'count']
        if type_counts.empty:
            st.info("No training type data available.")
        else:
            fig2 = px.bar(type_counts, x='type_training', y='count', labels={'type_training':'Type Training','count':'Count'})
            st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot training type frequency: {e}")
    st.subheader("Grip Type Usage")
    try:
        if 'meest_gebruikte_grip_type' in df:
            grip_types = df['meest_gebruikte_grip_type'].dropna().str.split(',').explode().str.strip()
            grip_counts = grip_types.value_counts().reset_index()
            grip_counts.columns = ['meest_gebruikte_grip_type', 'count']
            fig3 = px.pie(grip_counts, names='meest_gebruikte_grip_type', values='count', title='Grip Type Distribution')
            st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to plot grip type usage: {e}")


def fatigue_recovery_tab(df, load_label="Training Load"):
    st.header("Fatigue & Recovery")
    st.subheader("Overall Fatigue Heatmap")
    pivot = df.pivot_table(index=df['datum'].dt.strftime('%Y-%m-%d'), values='overall_fatigue')
    st.dataframe(pivot)

    # --- Predictive Overlay ---
    st.subheader("Predictive Overlay: JML → Pain/Fatigue/Rest")
    df_sorted = df.sort_values('datum').reset_index(drop=True)
    # Shift jml by 2 days
    df_sorted['jml_prev2'] = df_sorted['jml'].shift(2)
    # Simple predictive model: high jml_prev2 increases pain/fatigue likelihood
    pain_threshold = 3
    df_sorted['pain2dlikely'] = (df_sorted['jml_prev2'] > df_sorted['jml'].median()) | (df_sorted['jml_prev2'] > 200)
    df_sorted['restrecommendation'] = df_sorted['jml_prev2'] > df_sorted['jml'].quantile(0.85)
    # Overlay on pain/stiffness trend
    fig2 = px.line(df_sorted, x='datum', y='vinger_pijn_stijfheid', markers=True, title="Pain/Stiffness with Predicted Risk")
    fig2.add_scatter(x=df_sorted.loc[df_sorted['pain2dlikely'], 'datum'],
                     y=df_sorted.loc[df_sorted['pain2dlikely'], 'vinger_pijn_stijfheid'],
                     mode='markers', name='Predicted Pain Risk', marker=dict(color='red', size=10, symbol='star'))
    fig2.add_scatter(x=df_sorted.loc[df_sorted['restrecommendation'], 'datum'],
                     y=df_sorted.loc[df_sorted['restrecommendation'], 'vinger_pijn_stijfheid'],
                     mode='markers', name='Rest Recommended', marker=dict(color='orange', size=12, symbol='diamond'))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Red stars: high pain risk (from JML 2 days ago). Orange diamonds: rest recommended.")

    st.subheader(f"{load_label} vs Fatigue Next Day")
    df_sorted['nextdayfatigue'] = df_sorted['overall_fatigue'].shift(-1)
    fig = px.scatter(df_sorted, x='active_load', y='nextdayfatigue', hover_data=['datum'], labels={'active_load':load_label,'nextdayfatigue':'Next Day Fatigue'})
    st.plotly_chart(fig, use_container_width=True)

    # --- Weekly Summary Table ---
    st.subheader("Weekly Summary Table")
    week_grp = df_sorted.groupby('week')
    weekly_summary = week_grp.agg(
        total_duration=('duur_training_min', 'sum'),
        avg_jml=('jml', 'mean'),
        highest_jml=('jml', 'max'),
        peak_day=('datum', lambda x: x.iloc[np.argmax(df_sorted.loc[x.index, 'jml'])])
    ).reset_index()
    # Pain level the day after peak
    pain_after_peak = []
    for _, row in weekly_summary.iterrows():
        peak_day = row['peak_day']
        idx = df_sorted.index[df_sorted['datum'] == peak_day]
        if not idx.empty and idx[0] + 1 < len(df_sorted):
            pain_after = df_sorted.loc[idx[0] + 1, 'vinger_pijn_stijfheid']
        else:
            pain_after = np.nan
        pain_after_peak.append(pain_after)
    weekly_summary['pain_after_peak'] = pain_after_peak
    st.dataframe(weekly_summary[['week', 'total_duration', 'avg_jml', 'highest_jml', 'pain_after_peak']])


def music_analysis_tab(df):
    st.header("Music Analysis")
    if 'duur_spelen_min' in df and 'type_van_spelen_toepassing' in df:
        st.subheader("Music Session Duration by Type")
        music = df.groupby('type_van_spelen_toepassing').agg({'duur_spelen_min':'sum'}).reset_index()
        fig = px.area(music, x='type_van_spelen_toepassing', y='duur_spelen_min')
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Music Days vs Fatigue/Pain")
    if 'duur_spelen_min' in df:
        fig2 = px.scatter(df, x='duur_spelen_min', y='overall_fatigue', color='vinger_pijn_stijfheid', hover_data=['datum'])
        st.plotly_chart(fig2, use_container_width=True)


def correlations_tab(df, load_label="Training Load"):
    st.header("Correlations")
    st.subheader("Correlation Matrix")
    # Use active_load instead of training_load
    num_cols = ['training_intensity', 'duur_training_min', 'active_load', 'overall_fatigue', 'vinger_pijn_stijfheid', 'vinger_fatigue_tijdens_het_spelen']
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu', labels={'active_load':load_label})
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Daily Stats")
    st.dataframe(df[['datum']+num_cols])


def subjective_notes_tab(df, nlp_utils=None):
    st.header("Subjective Notes Analysis")
    if nlp_utils is None:
        st.warning("NLP features require textblob and wordcloud. Install all requirements and reload.")
        return
    
    try:
        nlp_utils.check_dependencies()
    except ImportError as e:
        st.warning(str(e))
        return
    
    if 'Subjectieve_notes' in df:
        st.subheader("Sentiment Scoring on Notes")
        sentiments = df['Subjectieve_notes'].apply(nlp_utils.sentiment_score)
        st.line_chart(sentiments)
        st.subheader("Word Cloud for High Pain Days")
        wc_fig = nlp_utils.wordcloud_for_high_pain(df)
        if wc_fig is not None:
            st.pyplot(wc_fig)
            st.caption("Most common words in subjective notes on days with high pain/stiffness.")
        else:
            st.info("No notes available for high-pain days, so no word cloud can be generated.")
    else:
        st.info("No subjective notes available in the data.")
