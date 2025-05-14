"""
Worlds ranking predictor – Streamlit edition
-------------------------------------------
Drop this file, your CSV, and requirements.txt in the same repo,
then deploy on Streamlit Community Cloud.

Author: <you>
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────
CSV_FILE    = "Worlds Design2.csv"   # Name of the file in the repo
TOTAL_RUNS  = 10                     # How many runs each team will have
DROPS       = 2                      # Number of worst runs discarded
REPS        = 10_000                 # Monte-Carlo replicates
RNG_SEED    = 42                     # For reproducibility
DEFAULT_SIGMA = 5.0                  # Stdev fallback for 1-run teams
# ─────────────────────────────────────────────────────────────────────────

# 0) PAGE SET-UP
st.set_page_config(page_title="Top-20 Predictor", layout="wide")

# 1) LOAD CSV
csv_path = Path(__file__).with_name(CSV_FILE)
if not csv_path.exists():
    st.error(f"❌ Can’t find “{CSV_FILE}” in the repo. "
             "Upload it or fix CSV_FILE at the top of worlds_rankings_app.py.")
    st.stop()

raw = pd.read_csv(csv_path)

score_cols = [c for c in raw.columns if "Score" in c]
team_cols  = [c for c in raw.columns if "Team"  in c]

if not score_cols or not team_cols:
    st.error("Auto-detection of column names failed – "
             "edit score_cols / team_cols lists in the source.")
    st.stop()

# 2) SEPARATE PLAYED & UNPLAYED
played = raw.loc[~raw[score_cols].isna().any(axis=1)].copy()
todo   = raw.loc[ raw[score_cols].isna().any(axis=1)].copy()

st.title("VEX IQ Worlds – interactive top-20 predictor")
st.markdown("Enter **predicted scores** for each un-played match. "
            "Leave blank for ‘unknown’ – the simulator will sample "
            "from that team’s own mean ± σ.")

# Editable grid
edited = st.data_editor(
    todo,
    num_rows="dynamic",
    hide_index=True,
    column_config={c: st.column_config.NumberColumn(step=1) for c in score_cols},
    key="todo_editor"
)

# Merge edits back
raw.update(edited)

# 3) BUILD SCORE LIST PER TEAM
scores = {}
for _, row in raw.iterrows():
    for s_col in score_cols:
        colour = s_col.split()[0]               # "Red"/"Blue"
        team_list = [c for c in team_cols if c.startswith(colour)]
        val = row[s_col]
        if pd.isna(val):
            continue
        for t_col in team_list:
            team = str(row[t_col]).strip()
            if team and team.lower() != "nan":
                scores.setdefault(team, []).append(float(val))

teams = sorted(scores)
n_teams = len(teams)

# 4) PRE-COMPUTE μ & σ FROM COMPLETED RUNS
mu, sigma, played_ct = {}, {}, {}
for t in teams:
    arr = np.asarray(scores[t], float)
    played_ct[t] = len(arr)
    mu[t] = arr.mean() if arr.size else 150.0
    sigma[t] = arr.std(ddof=1) if arr.size > 1 else DEFAULT_SIGMA

# 5) MONTE-CARLO
rng = np.random.default_rng(RNG_SEED)
all_preds = np.zeros((n_teams, REPS))
hits = np.zeros(n_teams, int)
cutlines = []

for _ in range(REPS):
    avgs = np.empty(n_teams)
    for i, t in enumerate(teams):
        need = TOTAL_RUNS - played_ct[t]
        fut = rng.normal(mu[t], sigma[t], need)
        run_scores = np.concatenate([scores[t], fut])
        run_scores.sort()
        avgs[i] = run_scores[DROPS:].mean()
    order = np.argsort(-avgs)
    hits[order[:20]] += 1
    cutlines.append(avgs[order[19]])
    all_preds[:, _] = avgs

# 6) SUMMARY TABLE
mean_pred = all_preds.mean(axis=1)
ci_lo = np.percentile(all_preds, 2.5, axis=1)
ci_hi = np.percentile(all_preds, 97.5, axis=1)
p20   = hits / REPS

summary = (pd.DataFrame({
            "Team": teams,
            "Predicted Avg": mean_pred,
            "CI Low": ci_lo,
            "CI High": ci_hi,
            "P(Top 20)": p20})
           .sort_values("P(Top 20)", ascending=False)
           .reset_index(drop=True))

st.subheader("Projected qualification table")
st.dataframe(
    summary.style.format({"Predicted Avg":"{:.1f}",
                          "CI Low":"{:.1f}",
                          "CI High":"{:.1f}",
                          "P(Top 20)":"{:.1%}"}),
    use_container_width=True,
    height=min(600, 25+28*len(summary))
)

# 7) VISUALS
with st.expander("Visual summaries"):
    # Cut-line histogram
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.hist(cutlines, bins=25, edgecolor="black")
    ax1.set_xlabel("Score of 20-th-place team")
    ax1.set_ylabel("Simulations")
    ax1.set_title("Cut-line distribution")
    st.pyplot(fig1)

    # Top-30 probability bars
    top30 = summary.head(30).iloc[::-1]
    fig2, ax2 = plt.subplots(figsize=(9, 0.28*len(top30)+1.2))
    ax2.barh(top30["Team"], top30["P(Top 20)"])
    ax2.set_xlabel("Probability of finishing in top 20")
    ax2.set_title("Top-30 teams")
    st.pyplot(fig2)

# 8) DOWNLOAD
st.download_button(
    "Download table as CSV",
    summary.to_csv(index=False).encode(),
    file_name="predicted_rankings_with_probabilities.csv",
    mime="text/csv"
)
