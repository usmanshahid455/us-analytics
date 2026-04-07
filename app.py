import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="Talking Bat | Pre-Match Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background-color: #0E1117; color: #EDEDED; }
        h1, h2, h3, h4, h5, h6, p, div, span, label { color: #EDEDED !important; }
        .metric-card { border: 1px solid #1f2937; border-radius: 10px; padding: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = [
    "match_id",
    "innings",
    "over",
    "ball",
    "batsman",
    "bowler",
    "batting_team",
    "bowling_team",
    "batsman_runs",
    "total_runs",
    "dismissal_kind",
    "player_dismissed",
    "bowler_type",
    "Venue",
    "Date",
    "Wide",
    "NB",
]


def safe_series(df: pd.DataFrame, name: str, default=None) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def validate_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    present = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return present, missing


@st.cache_data(show_spinner=False)
def load_and_prepare(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes))

    for c in ["Wide", "NB", "batsman_runs", "total_runs", "over", "ball", "innings"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "dismissal_kind" in df.columns:
        df["dismissal_kind"] = df["dismissal_kind"].fillna("not_out")

    # Legal ball and phase mapping by legal ball count.
    df["is_legal_ball"] = ((safe_series(df, "Wide", 0) == 0) & (safe_series(df, "NB", 0) == 0)).astype(int)

    sort_cols = [c for c in ["match_id", "innings", "over", "ball"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).copy()

    if {"match_id", "innings"}.issubset(df.columns):
        df["legal_ball_in_innings"] = df.groupby(["match_id", "innings"])["is_legal_ball"].cumsum()
    else:
        df["legal_ball_in_innings"] = df["is_legal_ball"].cumsum()

    def map_phase(legal_ball):
        if legal_ball <= 36:
            return "Overs 1-6"
        if legal_ball <= 60:
            return "Overs 7-10"
        if legal_ball <= 84:
            return "Overs 11-14"
        return "Overs 15-20"

    df["phase"] = df["legal_ball_in_innings"].apply(map_phase)
    df["is_wicket"] = (safe_series(df, "player_dismissed", "").fillna("") != "").astype(int)
    df["dot_ball"] = (safe_series(df, "batsman_runs", 0) == 0).astype(int)
    df["is_boundary"] = safe_series(df, "batsman_runs", 0).isin([4, 6]).astype(int)

    return df


def compute_team_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("batting_team", dropna=False).agg(
        runs=("total_runs", "sum"),
        balls=("is_legal_ball", "sum"),
        wickets=("is_wicket", "sum"),
        dots=("dot_ball", "sum"),
        boundaries=("is_boundary", "sum"),
    )
    grp["run_rate"] = np.where(grp["balls"] > 0, grp["runs"] * 6 / grp["balls"], 0)
    grp["dot_pct"] = np.where(grp["balls"] > 0, grp["dots"] / grp["balls"] * 100, 0)
    grp["boundary_pct"] = np.where(grp["balls"] > 0, grp["boundaries"] / grp["balls"] * 100, 0)
    return grp.reset_index()


def projection_band(value: float, spread: float = 0.2) -> str:
    low = max(0, round(value * (1 - spread), 1))
    high = round(value * (1 + spread), 1)
    return f"{low}–{high}"


def model_confidence(sample: int) -> str:
    if sample >= 300:
        return "High"
    if sample >= 100:
        return "Medium"
    return "Low"


st.sidebar.title("🏏 Talking Bat Pre-Match Engine")
uploaded_file = st.sidebar.file_uploader("Upload ball-by-ball CSV", type=["csv"])

if not uploaded_file:
    st.info("Upload a ball-by-ball dataset to generate the pre-match intelligence dashboard.")
    st.stop()

df = load_and_prepare(uploaded_file.getvalue())
present, missing = validate_columns(df)

if missing:
    st.warning(f"Some columns are missing and affected views will use fallbacks: {', '.join(missing)}")

st.sidebar.success(f"Loaded: {df.shape[0]:,} rows | {df.shape[1]} columns")

teams = sorted(df["batting_team"].dropna().unique().tolist()) if "batting_team" in df.columns else []
venues = sorted(df["Venue"].dropna().unique().tolist()) if "Venue" in df.columns else []

team_a = st.sidebar.selectbox("Team A", teams, index=0 if teams else None)
team_b = st.sidebar.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0)
selected_venue = st.sidebar.selectbox("Venue", venues, index=0 if venues else None)

match_df = df.copy()
if selected_venue and "Venue" in df.columns:
    match_df = match_df[match_df["Venue"] == selected_venue]

if team_a and team_b:
    match_df = match_df[match_df["batting_team"].isin([team_a, team_b]) | match_df["bowling_team"].isin([team_a, team_b])]

team_summary = compute_team_summary(match_df)

if team_summary.empty:
    st.error("Not enough filtered data for the selected teams/venue.")
    st.stop()

# Rule-based win probability proxy.
score = {}
for _, r in team_summary.iterrows():
    score[r["batting_team"]] = (r["run_rate"] * 0.45) + ((100 - r["dot_pct"]) * 0.25) + (r["boundary_pct"] * 0.30)

if team_a in score and team_b in score:
    total = score[team_a] + score[team_b]
    base_a = round((score[team_a] / total) * 100, 1) if total > 0 else 50.0
else:
    base_a = 50.0
base_b = round(100 - base_a, 1)

par_score = int(match_df.groupby(["match_id", "innings"], dropna=False)["total_runs"].sum().mean()) if "match_id" in match_df.columns else int(match_df["total_runs"].mean() * 120)

st.title("Talking Bat — Elite Pre-Match Prediction Engine")

st.caption("Quick start: Upload CSV → pick Team A/Team B/Venue in sidebar → read tabs left to right (Overview to Final Prediction).")


TAB_NAMES = [
    "Match Overview",
    "Venue Intelligence",
    "Team Comparison",
    "Batting Predictions",
    "Bowling Predictions",
    "Key Matchups",
    "Dismissal Risk",
    "Phase Battle",
    "Strategy Engine",
    "Final Prediction",
]

tabs = st.tabs(TAB_NAMES)

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Team A", team_a or "N/A")
    c2.metric("Team B", team_b or "N/A")
    c3.metric("Venue", selected_venue or "All")
    c4.metric("Par Score", par_score)
    st.write(f"**Base win probability:** {team_a}: {base_a}% | {team_b}: {base_b}%")

with tabs[1]:
    venue_grp = match_df.groupby("phase", dropna=False).agg(runs=("total_runs", "sum"), balls=("is_legal_ball", "sum"), wickets=("is_wicket", "sum")).reset_index()
    venue_grp["run_rate"] = np.where(venue_grp["balls"] > 0, venue_grp["runs"] * 6 / venue_grp["balls"], 0)
    st.dataframe(venue_grp, use_container_width=True)
    st.plotly_chart(px.bar(venue_grp, x="phase", y="run_rate", color="phase", title="Venue Phase Run Rate"), use_container_width=True)

with tabs[2]:
    st.dataframe(team_summary.sort_values("run_rate", ascending=False), use_container_width=True)
    fig = px.bar(team_summary, x="batting_team", y=["run_rate", "dot_pct", "boundary_pct"], barmode="group", title="Team Style Indicators")
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    batter_pool = match_df[match_df["batting_team"].isin([team_a, team_b])].copy()
    batter = batter_pool.groupby("batsman", dropna=False).agg(
        runs=("batsman_runs", "sum"),
        balls=("is_legal_ball", "sum"),
        outs=("is_wicket", "sum"),
        boundaries=("is_boundary", "sum"),
    ).reset_index()
    batter = batter[batter["balls"] > 0]
    batter["sr"] = batter["runs"] * 100 / batter["balls"]
    batter["dismissal_risk"] = np.where(batter["balls"] > 0, batter["outs"] / batter["balls"] * 100, 0)
    top_bat = batter.sort_values("runs", ascending=False).head(10)
    st.dataframe(top_bat, use_container_width=True)

    if not top_bat.empty:
        sel = st.selectbox("Select batter projection", top_bat["batsman"].tolist())
        row = top_bat[top_bat["batsman"] == sel].iloc[0]
        st.info(
            f"Expected Runs: {projection_band(row['runs'] / max(1, row['balls']) * 20)} | "
            f"Expected Balls: {projection_band(20, 0.25)} | "
            f"Expected SR: {projection_band(row['sr'], 0.12)} | "
            f"Dismissal Risk: {round(row['dismissal_risk'],1)}%"
        )

with tabs[4]:
    bowl_df = match_df.groupby("bowler", dropna=False).agg(
        balls=("is_legal_ball", "sum"),
        wickets=("is_wicket", "sum"),
        runs=("total_runs", "sum"),
        dots=("dot_ball", "sum"),
    ).reset_index()
    bowl_df = bowl_df[bowl_df["balls"] > 0]
    bowl_df["economy"] = bowl_df["runs"] * 6 / bowl_df["balls"]
    bowl_df["dot_pct"] = bowl_df["dots"] * 100 / bowl_df["balls"]
    top_bowl = bowl_df.sort_values(["wickets", "dot_pct"], ascending=False).head(10)
    st.dataframe(top_bowl, use_container_width=True)

with tabs[5]:
    h2h = match_df.groupby(["batsman", "bowler"], dropna=False).agg(runs=("batsman_runs", "sum"), balls=("is_legal_ball", "sum"), outs=("is_wicket", "sum")).reset_index()
    h2h = h2h[h2h["balls"] >= 6]
    h2h["sr"] = h2h["runs"] * 100 / h2h["balls"]
    h2h["edge_score"] = h2h["sr"] - (h2h["outs"] * 12)
    top_h2h = h2h.sort_values("edge_score", ascending=False).head(15)
    st.dataframe(top_h2h, use_container_width=True)

with tabs[6]:
    dismissals = match_df[match_df["dismissal_kind"] != "not_out"].copy()
    risk = dismissals.groupby(["batsman", "dismissal_kind"], dropna=False).size().reset_index(name="count")
    st.dataframe(risk.sort_values("count", ascending=False).head(20), use_container_width=True)
    if not risk.empty:
        st.plotly_chart(px.bar(risk.sort_values("count", ascending=False).head(15), x="batsman", y="count", color="dismissal_kind", title="Top Dismissal Modes"), use_container_width=True)

with tabs[7]:
    phase_team = match_df.groupby(["phase", "batting_team"], dropna=False).agg(runs=("total_runs", "sum"), balls=("is_legal_ball", "sum"), wickets=("is_wicket", "sum")).reset_index()
    phase_team["run_rate"] = np.where(phase_team["balls"] > 0, phase_team["runs"] * 6 / phase_team["balls"], 0)
    st.dataframe(phase_team, use_container_width=True)
    st.plotly_chart(px.bar(phase_team, x="phase", y="run_rate", color="batting_team", barmode="group", title="Phase Dominance"), use_container_width=True)

with tabs[8]:
    toss_rec = "Bowl first" if base_a < 55 else "Bat first"
    st.subheader("Coach Strategy Notes")
    st.write(f"**Recommended toss decision:** {toss_rec}")
    st.markdown(
        "- Attack top-order with full straight pace early.\n"
        "- Use spin-control in overs 7–14 if dot% rises.\n"
        "- Save highest dot% bowler for overs 17–20.\n"
        "- Avoid feeding width to finishers with high boundary rates."
    )

with tabs[9]:
    st.subheader("Final Prediction")
    st.write(f"**{team_a} win probability:** {base_a}%")
    st.write(f"**{team_b} win probability:** {base_b}%")
    st.write(f"If bowling first (+6% adjustment): {team_a} {min(100, base_a + 6)}% | {team_b} {max(0, base_b - 6)}%")
    st.write(f"If batting first (+6% adjustment): {team_a} {max(0, base_a - 6)}% | {team_b} {min(100, base_b + 6)}%")

    sample = len(match_df)
    st.caption(f"Confidence: {model_confidence(sample)} (sample size: {sample} balls)")

    if st.button("Export Coach Report (PDF)"):
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer)
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(72, 800, "Talking Bat - Pre-Match Report")
        pdf.setFont("Helvetica", 11)
        pdf.drawString(72, 780, f"Teams: {team_a} vs {team_b}")
        pdf.drawString(72, 764, f"Venue: {selected_venue}")
        pdf.drawString(72, 748, f"Win Probabilities: {team_a} {base_a}% | {team_b} {base_b}%")
        pdf.drawString(72, 732, f"Par Score: {par_score}")
        pdf.drawString(72, 716, f"Toss Advice: {toss_rec}")
        pdf.save()
        buffer.seek(0)
        st.download_button("Download PDF", buffer, file_name="prematch_report.pdf", mime="application/pdf")
