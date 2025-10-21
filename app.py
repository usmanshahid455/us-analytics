"""Cricket analytics dashboard for ball-by-ball T20 data."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.pdfgen import canvas


DATA_PATH = Path(__file__).with_name("Women U18 Ball by ball 2 Matches Missing updated.csv")


def balls_to_overs(balls: Iterable[int]) -> pd.Series:
    """Convert ball counts into cricket style overs (e.g. 4.2)."""

    balls = pd.Series(balls, dtype="float")
    overs = (balls // 6).astype(int)
    balls_remaining = (balls % 6).astype(int)
    return overs.astype(str) + "." + balls_remaining.astype(str)


def format_float(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{decimals}f}"


@st.cache_data(show_spinner=False)
def load_data(source: Optional[Path] = None, uploaded_file: Optional[BytesIO] = None) -> pd.DataFrame:
    """Load and lightly preprocess the ball-by-ball dataset."""

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif source is not None and source.exists():
        df = pd.read_csv(source)
    else:
        raise FileNotFoundError("No data source available. Upload a CSV to proceed.")

    df = df.convert_dtypes()
    df["over_number"] = df["over"].astype(int) + 1
    df["ball_in_over"] = df["ball"].astype(int)
    df["over_ball"] = df["over_number"] + (df["ball_in_over"] - 1) / 10
    df["is_boundary"] = df["batsman_runs"].isin([4, 6])
    df["is_wicket"] = df["player_dismissed"].notna()
    df["innings_label"] = df["innings"].astype(str) + " Innings"
    df["match_label"] = df["match_id"].astype(str) + " - " + df["Venue"].astype(str)
    return df


def filter_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar driven filters to the dataframe."""

    st.sidebar.header("Filters")

    matches = ["All matches"] + sorted(df["match_label"].unique().tolist())
    selected_match = st.sidebar.selectbox("Match", matches)

    innings_options = ["All innings"] + sorted(df["innings_label"].unique().tolist())
    selected_innings = st.sidebar.selectbox("Innings", innings_options)

    teams = ["All teams"] + sorted(df["batting_team"].dropna().unique().tolist())
    selected_team = st.sidebar.selectbox("Batting side", teams)

    shot_types = ["All shots"] + sorted(df["shot_name"].dropna().unique().tolist())
    selected_shot = st.sidebar.selectbox("Shot type", shot_types)

    filtered = df.copy()
    if selected_match != "All matches":
        filtered = filtered[filtered["match_label"] == selected_match]

    if selected_innings != "All innings":
        filtered = filtered[filtered["innings_label"] == selected_innings]

    if selected_team != "All teams":
        filtered = filtered[filtered["batting_team"] == selected_team]

    if selected_shot != "All shots":
        filtered = filtered[filtered["shot_name"] == selected_shot]

    return filtered


def summary_metrics(df: pd.DataFrame) -> None:
    total_runs = int(df["total_runs"].sum())
    total_balls = df.shape[0]
    total_overs = total_balls / 6 if total_balls else np.nan
    run_rate = total_runs / total_overs if total_overs else np.nan
    wickets = int(df["is_wicket"].sum())
    boundaries = int(df["is_boundary"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", f"{total_runs}")
    c2.metric("Run rate", format_float(run_rate))
    c3.metric("Wickets", f"{wickets}")
    c4.metric("Boundaries", f"{boundaries}")


def plot_scoring_worm(df: pd.DataFrame) -> None:
    progression = (
        df.sort_values(["match_id", "innings", "over", "ball"])
        .assign(cumulative_runs=lambda d: d.groupby(["match_id", "innings"])["total_runs"].cumsum())
    )
    if progression.empty:
        st.info("Not enough data to render scoring progression.")
        return

    fig = px.line(
        progression,
        x="over_ball",
        y="cumulative_runs",
        color="innings_label",
        labels={"over_ball": "Over", "cumulative_runs": "Runs", "innings_label": "Innings"},
        markers=True,
        title="Scoring progression",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    st.plotly_chart(fig, use_container_width=True)


def batting_highlights(df: pd.DataFrame) -> None:
    batting = (
        df.groupby("batsman")
        .agg(
            runs=("batsman_runs", "sum"),
            balls=("ball", "count"),
            dismissals=("is_wicket", "sum"),
            boundaries=("is_boundary", "sum"),
        )
        .reset_index()
    )
    batting = batting[batting["balls"] > 0]
    batting["strike_rate"] = batting["runs"] / batting["balls"] * 100
    batting["average"] = batting.apply(
        lambda row: row["runs"] / row["dismissals"] if row["dismissals"] else np.nan, axis=1
    )
    batting = batting.sort_values("runs", ascending=False)

    top_batting = batting.head(10)
    st.subheader("Top batters")
    st.dataframe(
        top_batting.assign(
            strike_rate=top_batting["strike_rate"].map(lambda v: round(v, 2)),
            average=top_batting["average"].map(lambda v: "-" if pd.isna(v) else round(v, 2)),
            balls=top_batting["balls"].astype(int),
            boundaries=top_batting["boundaries"].astype(int),
        ),
        use_container_width=True,
    )

    shot_breakdown = df.groupby("shot_name")["batsman_runs"].sum().reset_index()
    shot_breakdown = shot_breakdown.sort_values("batsman_runs", ascending=False)
    fig = px.bar(
        shot_breakdown,
        x="batsman_runs",
        y="shot_name",
        orientation="h",
        labels={"batsman_runs": "Runs", "shot_name": "Shot"},
        title="Runs by shot type",
    )
    st.plotly_chart(fig, use_container_width=True)


def bowling_highlights(df: pd.DataFrame) -> None:
    bowling = (
        df.groupby("bowler")
        .agg(
            balls=("ball", "count"),
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
        )
        .reset_index()
    )
    bowling = bowling[bowling["balls"] > 0]
    bowling["overs_float"] = bowling["balls"] / 6
    bowling["economy"] = bowling.apply(
        lambda row: row["runs"] / row["overs_float"] if row["overs_float"] else np.nan, axis=1
    )
    bowling["overs"] = balls_to_overs(bowling["balls"])
    bowling = bowling.sort_values(["wickets", "runs"], ascending=[False, True])

    st.subheader("Top bowlers")
    st.dataframe(
        bowling.assign(
            runs=bowling["runs"].astype(int),
            wickets=bowling["wickets"].astype(int),
            economy=bowling["economy"].map(lambda v: format_float(v, 2)),
        )[["bowler", "overs", "runs", "wickets", "economy"]].head(10),
        use_container_width=True,
    )

    dismissal = df[df["is_wicket"]].groupby("dismissal_kind").size().reset_index(name="count")
    dismissal = dismissal.sort_values("count", ascending=False)
    fig = px.pie(
        dismissal,
        names="dismissal_kind",
        values="count",
        title="Dismissal breakdown",
        hole=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)


def partnership_matrix(df: pd.DataFrame) -> None:
    partnerships = (
        df.groupby(["match_id", "innings", "batsman", "non_striker"])["total_runs"].sum().reset_index()
    )
    partnerships = partnerships.sort_values("total_runs", ascending=False).head(15)
    fig = px.bar(
        partnerships,
        x="total_runs",
        y="batsman",
        color="non_striker",
        orientation="h",
        labels={"total_runs": "Runs", "batsman": "Striker", "non_striker": "Partner"},
        title="Leading partnerships",
    )
    st.plotly_chart(fig, use_container_width=True)


def match_centre(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Apply a match filter to view ball-by-ball commentary.")
        return

    grouped = (
        df.sort_values(["over", "ball"])
        .assign(
            cumulative_runs=lambda d: d["total_runs"].cumsum(),
            over_display=lambda d: d.apply(lambda row: f"{int(row['over_number'])}.{int(row['ball_in_over'])}", axis=1),
        )
    )

    st.subheader("Ball-by-ball timeline")
    st.dataframe(
        grouped[[
            "over_display",
            "batsman",
            "bowler",
            "batsman_runs",
            "extra_runs",
            "total_runs",
            "cumulative_runs",
            "dismissal_kind",
            "shot_name",
        ]],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered data",
        grouped.to_csv(index=False).encode("utf-8"),
        file_name="filtered_ball_by_ball.csv",
        mime="text/csv",
    )


def strategy_insights(df: pd.DataFrame) -> None:
    st.subheader("Match-up insights")
    batters = sorted(df["batsman"].dropna().unique().tolist())
    bowlers = sorted(df["bowler"].dropna().unique().tolist())

    if not batters or not bowlers:
        st.info("Not enough data for match-up analysis.")
        return

    batter = st.selectbox("Batter", batters)
    bowler = st.selectbox("Bowler", ["All bowlers"] + bowlers)

    batter_df = df[df["batsman"] == batter]
    if bowler != "All bowlers":
        batter_df = batter_df[batter_df["bowler"] == bowler]

    runs = batter_df["batsman_runs"].sum()
    balls = batter_df.shape[0]
    dismissals = batter_df["is_wicket"].sum()
    strike_rate = runs / balls * 100 if balls else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Runs", int(runs))
    c2.metric("Balls", int(balls))
    c3.metric("Strike rate", format_float(strike_rate))

    contact = (
        batter_df.groupby("connection_name")["batsman_runs"].sum().reset_index().sort_values("batsman_runs", ascending=False)
    )
    shot = (
        batter_df.groupby("shot_name")["batsman_runs"].sum().reset_index().sort_values("batsman_runs", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Runs by contact quality")
        st.dataframe(contact, use_container_width=True, hide_index=True)
    with col2:
        st.caption("Runs by shot")
        st.dataframe(shot.head(10), use_container_width=True, hide_index=True)

    strong = batter_df[batter_df["connection_name"] == "Perfect"]["shot_name"].mode()
    weak = batter_df[batter_df["connection_name"] == "Miss"]["shot_name"].mode()

    st.info(f"🟢 Strongest scoring option: {strong.iloc[0] if not strong.empty else 'Not enough data'}")
    st.warning(f"🔴 Most missed shot: {weak.iloc[0] if not weak.empty else 'Not enough data'}")

    if st.button("Export player report (PDF)"):
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(40, 800, "US-Analytics | Player Match-up Report")
        pdf.setFont("Helvetica", 12)
        pdf.drawString(40, 770, f"Batter: {batter}")
        pdf.drawString(40, 750, f"Bowler: {bowler}")
        pdf.drawString(40, 730, f"Runs: {runs} off {balls} balls")
        pdf.drawString(40, 710, f"Strike rate: {format_float(strike_rate)}")
        pdf.drawString(40, 690, f"Best scoring option: {strong.iloc[0] if not strong.empty else 'N/A'}")
        pdf.drawString(40, 670, f"Weakest shot: {weak.iloc[0] if not weak.empty else 'N/A'}")
        pdf.showPage()
        pdf.save()
        buffer.seek(0)
        st.download_button(
            "Download PDF",
            buffer,
            file_name=f"{batter}_matchup_report.pdf",
            mime="application/pdf",
        )


def main() -> None:
    st.set_page_config(
        page_title="US-Analytics | Cricket Insights",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        body { background-color: #0E1117; color: white; }
        .stApp { background-color: #0E1117; color: white; }
        h1, h2, h3, h4, h5, h6, p, div, span { color: white !important; }
        .stTabs [data-baseweb="tab-list"] button { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("US-Analytics | CricViz-style dashboard")
    st.caption("Interactive insights built on ball-by-ball T20 data.")

    st.sidebar.title("🏏 Data loading")
    uploaded_file = st.sidebar.file_uploader("Upload a ball-by-ball CSV", type=["csv"])

    try:
        df = load_data(uploaded_file=uploaded_file if uploaded_file else None, source=DATA_PATH)
    except FileNotFoundError:
        st.error("No dataset available. Upload a CSV file to continue.")
        st.stop()

    if uploaded_file is None:
        st.sidebar.success("Using bundled Women U18 T20 dataset")
    else:
        st.sidebar.success("Custom dataset loaded")

    st.sidebar.markdown("---")
    filtered_df = filter_frame(df)

    st.subheader("Overview")
    st.write(
        f"Displaying **{filtered_df.shape[0]}** balls from **{filtered_df['match_id'].nunique()}** matches and "
        f"**{filtered_df['batting_team'].nunique()}** batting sides."
    )
    summary_metrics(filtered_df)

    tab_overview, tab_batting, tab_bowling, tab_partnerships, tab_match, tab_strategy = st.tabs(
        [
            "📈 Scoring progression",
            "🏏 Batting",
            "🎯 Bowling",
            "🤝 Partnerships",
            "🗒 Match centre",
            "🧠 Strategy",
        ]
    )

    with tab_overview:
        plot_scoring_worm(filtered_df)

    with tab_batting:
        batting_highlights(filtered_df)

    with tab_bowling:
        bowling_highlights(filtered_df)

    with tab_partnerships:
        partnership_matrix(filtered_df)

    with tab_match:
        match_centre(filtered_df)

    with tab_strategy:
        strategy_insights(filtered_df)


if __name__ == "__main__":
    main()
