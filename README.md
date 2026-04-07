# Talking Bat — What To Do Now (Step-by-Step)

If you're feeling stuck, follow this exact order.

## 1) Run the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload your ball-by-ball CSV in the sidebar.

---

## 2) Validate your dataset first (most important)

Your CSV should ideally include these core columns:

- `match_id`, `innings`, `over`, `ball`
- `batsman`, `bowler`
- `batting_team`, `bowling_team`
- `batsman_runs`, `total_runs`
- `dismissal_kind`, `player_dismissed`
- `Venue`, `Date`, `Wide`, `NB`

If columns are missing, the app will still run but some tabs become weaker.

---

## 3) What the current V1 already gives you

- Legal-ball based phase splits (`Overs 1-6`, `7-10`, `11-14`, `15-20`)
- Team profile summary (`run_rate`, `dot%`, `boundary%`)
- Basic batter and bowler projection cards
- H2H matchup table (minimum 6-ball sample)
- Dismissal mode view and phase battle
- Rule-based toss/win recommendation + PDF coach report

---

## 4) What you should build next (recommended)

### V2 (ML core)

1. **Batter runs model** (regression): expected runs, balls, SR band  
2. **Dismissal model** (classification): out/not out + dismissal kind probabilities  
3. **Bowler wickets/economy models**: expected wickets + economy band  
4. **Win probability model**: calibrated probabilities with time-based split

### V3 (simulation)

1. Toss scenario simulation (bat first vs bowl first)
2. Over-range score simulation
3. Playing-XI sensitivity scenarios

---

## 5) Immediate next coding task (start here)

Create a new file:

- `src/features.py`

Move all feature engineering from `app.py` into reusable functions:

- `build_legal_ball_features(df)`
- `build_phase_features(df)`
- `build_batter_features(df, n_matches=10)`
- `build_bowler_features(df, n_matches=10)`
- `build_team_features(df)`

Then import those functions in `app.py`.

This will make model training possible without rewriting the dashboard.

---

## 6) Minimum model validation rules (do not skip)

Use **time-based split only**:

- Train: older matches
- Test: recent matches

Track:

- Runs model: MAE, RMSE
- Dismissal model: accuracy, log loss
- Win model: accuracy, ROC-AUC, calibration

---

## 7) If you want me to implement the next part for you

Say exactly one of these:

- **"build feature tables now"**
- **"add ML training pipeline now"**
- **"upgrade the dashboard visuals now"**

I’ll then implement that end-to-end in code.
