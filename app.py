# ======================================
# US-Analytics (CricViz Style App)
# ======================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from io import BytesIO

st.set_page_config(
    page_title="US-Analytics | Cricket Insights",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body { background-color: #0E1117; color: white; }
    .stApp { background-color: #0E1117; color: white; }
    h1, h2, h3, h4, h5, h6, p, div, span { color: white !important; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🏏 US-Analytics Dashboard")
st.sidebar.markdown("**CricViz-style insights for ball-by-ball data**")

uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🏏 Player Insights",
        "🤝 Partnerships",
        "📈 Team & Venue Analysis",
        "🧠 Strategy Planner"
    ])

    with tab1:
        st.subheader("Tournament Overview")
        st.dataframe(df.head())
        match_count = df['match_id'].nunique()
        total_runs = df['total_runs'].sum()
        st.metric("Matches", match_count)
        st.metric("Total Runs", total_runs)

        fig = px.histogram(df, x="batsman_runs", title="Distribution of Batsman Runs", color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Player Insights")

        player = st.selectbox("Select Batsman", sorted(df['batsman'].dropna().unique()))
        player_data = df[df['batsman'] == player]

        runs = player_data['batsman_runs'].sum()
        balls = player_data.shape[0]
        avg = round(runs / (player_data['dismissal_kind'].notna().sum() or 1), 2)
        sr = round((runs / balls) * 100, 2)

        st.metric("Runs", runs)
        st.metric("Balls", balls)
        st.metric("Average", avg)
        st.metric("Strike Rate", sr)

        fig2 = px.pie(player_data, names='shot_name', title=f"{player} Shot Type Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Top Partnerships")
        partnerships = df.groupby(['batting_team', 'batsman', 'non_striker'])['total_runs'].sum().reset_index()
        top_pairs = partnerships.sort_values(by='total_runs', ascending=False).head(10)
        fig3 = px.bar(top_pairs, x='total_runs', y='batsman', color='non_striker', orientation='h', title="Top Partnerships")
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader("Team and Venue Analysis")

        col1, col2 = st.columns(2)
        with col1:
            team_runs = df.groupby('batting_team')['total_runs'].sum().sort_values(ascending=False)
            st.bar_chart(team_runs)

        with col2:
            venue_runs = df.groupby('Venue')['total_runs'].sum().sort_values(ascending=False)
            st.bar_chart(venue_runs)

    with tab5:
        st.subheader("Strategy Planner (Strengths & Weaknesses)")
        selected_batsman = st.selectbox("Select Batsman for Strategy", df['batsman'].unique())

        batsman_data = df[df['batsman'] == selected_batsman]
        strong_zone = batsman_data[batsman_data['connection_name'] == 'Perfect']['shot_name'].mode()[0] if not batsman_data.empty else 'N/A'
        weak_zone = batsman_data[batsman_data['connection_name'] == 'Miss']['shot_name'].mode()[0] if not batsman_data.empty else 'N/A'

        st.info(f"🟢 **Strong Zone:** {strong_zone}")
        st.warning(f"🔴 **Weak Zone:** {weak_zone}")

        if st.button("📄 Export Player Report (PDF)"):
            buffer = BytesIO()
            pdf = canvas.Canvas(buffer)
            pdf.setFont("Helvetica", 14)
            pdf.drawString(100, 800, f"US-Analytics Player Report: {selected_batsman}")
            pdf.drawString(100, 770, f"Strong Zone: {strong_zone}")
            pdf.drawString(100, 740, f"Weak Zone: {weak_zone}")
            pdf.save()
            buffer.seek(0)
            st.download_button("⬇️ Download Report", buffer, file_name=f"{selected_batsman}_report.pdf")

else:
    st.warning("👈 Please upload your CSV file to begin analysis.")
