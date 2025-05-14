# leaderboard_predictor_app.py
"""
VEX IQ Worlds – live leaderboard **plus** interactive forecast
-------------------------------------------------------------
Built on the counting logic we just fixed:
* A run for an alliance is counted **only if**
  1. that alliance’s JSON score is positive, **or**
  2. the match has been certified (`scored == true`) **and** the JSON score is
     numeric (0 or > 0).
* Each team’s practice run is already absent from the schedule, so no practice
  toggle is needed.
* Current leaderboard: drop ⌊runs / 4⌋ lowest scores.
* Forecast to 10 runs: drop the lowest 2 of 10.
* You can type predictions for still‑blank future rounds; the app recomputes:
  – 95 % CI of each team’s final average.
  – Probability of finishing in the top 20.
  – Histogram of the 20‑place cut‑line.
  – Bar‑chart of top‑20 probabilities (top 30 teams).
"""
import datetime as dt, typing, math, requests, streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# ── RobotEvents token ──────────────────────────────────────────────────
RE_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzIiwianRpIjoiMjRkYjJkYjY3NTYwNTk3ZDgyYjQzZjhiMzU2ZmE4ODhkYTQ5MGU5MDZmNTIwZjdhNmJlMGY4NmRmNGQyNmRiMmE3ZjY3MTVjMGFiOTJjNWQiLCJpYXQiOjE3NDcxOTk5ODQuOTg0Mjc3LCJuYmYiOjE3NDcxOTk5ODQuOTg0Mjg4LCJleHAiOjI2OTM4ODQ3ODQuOTc4NTUsInN1YiI6IjExMTkyOCIsInNjb3BlcyI6W119.QAjwPG4z1s6D1OKU9zhtSCIVJicUkSSgnA9MgQBIlK9XomAt8OnE1i5-6GcutO8AKxIv5jU_auo5WGti0ms5ukmjV0b74GdXRyiVLd2yYSKzJwJo1nftGACKjG4YvNTjJEmmlx4Irs2rfCF7O6jJwsLt-up3gw8zHBHGJ-4j0Ras54AJ-_5fLNzW4NiY5yAx-UXa4CK2jCFqlaBbRNXhHesrFYKaxhhN4tBue6yl2XyX8Ry6CNWQVjhpVyHZdeTfB4qS01wGc8UphCv7Kb9zsuSrnS0QvbdR-HSMjbBv1p8SmnMLTXj_YvTUhDXg7F-qDyHnZzs59ohqSQQweNOn2LtxBH9GJh6y19Ma6z1pLjdzw8rl8MX-h1yBFl3Tmy3nrqKjkbJaPPo04Z7FOA29wVv9IFxRt78d4DI-xiJIB-NBkKiVjbidVB6RUtyP-AW-3_xOZAidjR3Rlij9jTSEtwb_Ln2JqZdJ6qURgqbxfoI-PlFm7075aGrO8GQ8W0I942UxBWhHANJ_mQDXRmbjMgYBU2MkM41ap60rvgzEW_UlUB6pAP8zia9PbhDhdleKPzmiqNmjprJ4Kmk9XJxpvpYSwAjL0B1E9vDbe18uUSkIGdg1AvWoDVmTlFcrP_NFJ6rR5A8OPv_RZSBqq4k4DKZyR1ilxWwX4q0JSglsgns"  # <– replace with your token
# ----------------------------------------------------------------------

DEFAULT_EVENT, DEFAULT_DIV = "58913", "4"
CACHE_TTL = 600    # sec cache RobotEvents JSON
TOTAL_RUNS = 10    # final number of scored runs
FINAL_DROPS = 2    # drop lowest 2 of 10
REPLICATES = 10_000
RNG_SEED = 42
DEFAULT_MEAN, DEFAULT_SIGMA, ONE_SIGMA = 150.0, 30.0, 5.0

st.set_page_config("VEX IQ Predictor", layout="wide")

# ── sidebar ────────────────────────────────────────────────────────────
st.sidebar.header("RobotEvents IDs")
event_id = st.sidebar.text_input("Event ID", value=DEFAULT_EVENT)
division_id = st.sidebar.text_input("Division ID", value=DEFAULT_DIV)

if not (event_id and division_id):
    st.stop()

# ── fetch helper (cached) ──────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching RobotEvents…", ttl=CACHE_TTL)
def fetch_matches(eid: str, did: str, token: str) -> pd.DataFrame:
    url = f"https://www.robotevents.com/api/v2/events/{eid}/divisions/{did}/matches"
    hdrs = {"accept": "application/json", "Authorization": f"Bearer {token}"}
    rows, page = [], url
    while page:
        r = requests.get(page, params={"round[]": 2, "per_page": 50}, headers=hdrs, timeout=15)
        r.raise_for_status()
        j = r.json(); rows += j["data"]; page = j["meta"]["next_page_url"]
    rec = []
    for m in rows:
        red  = next(a for a in m["alliances"] if a["color"].lower() == "red")
        blue = next(a for a in m["alliances"] if a["color"].lower() == "blue")
        rec.append({
            "match":      m["matchnum"],
            "scored":     m["scored"],
            "Red Team":   red ["teams"][0]["team"]["name"]  if red ["teams"] else None,
            "Blue Team":  blue["teams"][0]["team"]["name"] if blue["teams"] else None,
            "Red Score":  red ["score"],
            "Blue Score": blue["score"],
        })
    return pd.DataFrame(rec).sort_values("match", ignore_index=True)

try:
    df = fetch_matches(event_id.strip(), division_id.strip(), RE_TOKEN)
except Exception as e:
    st.error(e)
    st.stop()

st.caption(f"{len(df)} match rows fetched  ({dt.datetime.utcnow():%Y-%m-%d %H:%MZ})")

score_cols, team_cols = ["Red Score","Blue Score"], ["Red Team","Blue Team"]

# ── mark completed alliances (same rule as leaderboard) ───────────────
completed_mask = []
for idx, row in df.iterrows():
    red_done  = (not pd.isna(row["Red Score"])  and row["Red Score"] > 0) or (row["scored"] and not pd.isna(row["Red Score"]))
    blue_done = (not pd.isna(row["Blue Score"]) and row["Blue Score"] > 0) or (row["scored"] and not pd.isna(row["Blue Score"]))
    completed_mask.append(red_done or blue_done)
df["finished_any"] = completed_mask

completed_df = df[df["finished_any"]].copy()
future_df    = df[~df["finished_any"]].copy()

# ── editable grid for future rounds ───────────────────────────────────
st.title("VEX IQ Predictor – leaderboard + forecast")
st.markdown("Type predictions for still‑blank future rounds. The table and charts update live.")

edited = st.data_editor(
    future_df,
    hide_index=True,
    num_rows="dynamic",
    column_config={c: st.column_config.NumberColumn(step=1) for c in score_cols},
)
future_df.update(edited)

full = pd.concat([completed_df, future_df], ignore_index=True)

# ── per‑team scores (completed + user predictions) ────────────────────
per_team = {}
for _, row in full.iterrows():
    for s_col, t_col in zip(score_cols, team_cols):
        team = str(row[t_col]).strip()
        if not team or team.lower() == "nan":
            continue
        val = row[s_col]
        # count if val is numeric (prediction or completed criteria met)
        if pd.isna(val):
            continue
        # ensure counting rules (for predictions we accept any numeric value)
        if (val > 0) or (row["scored"] and not pd.isna(val)):
            per_team.setdefault(team, []).append(float(val))

# ── leaderboard (drop ⌊n/4⌋) ─────────────────────────────────────────
leader_rows = []
for team, scores in per_team.items():
    n = len(scores)
    drops = n // 4
    avg = np.mean(sorted(scores)[drops:]) if scores else 0.0
    leader_rows.append((team, n, avg))
leader_df = (pd.DataFrame(leader_rows, columns=["Team","Runs","Current Avg"])  
             .sort_values("Current Avg", ascending=False).reset_index(drop=True))

# ── Monte‑Carlo forecast (to 10 runs, drop 2) ─────────────────────────
rng = np.random.default_rng(RNG_SEED)
pred   = np.zeros((n_teams, REPS))
hits20 = np.zeros(n_teams, int)
cut    = []

for _ in range(REPS):
    avgs = np.empty(n_teams)
    for i, t in enumerate(teams):
        need = max(0, TOTAL_RUNS - n_played[t])
        if need > 0:
            sig = sigma[t] if sigma[t] > 0 and not np.isnan(sigma[t]) else 1.0
            future = rng.normal(mu[t], sig, need)
        else:
            future = np.empty(0)

        all_scores = np.concatenate([scores[t], future])
        all_scores.sort()
        avgs[i] = all_scores[FINAL_DROPS:].mean()

    order = np.argsort(-avgs)
    hits20[order[:20]] += 1
    cut.append(avgs[order[19]])
    pred[:, _] = avgs

# ── display tables and charts ─────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Current leaderboard  –  drop ⌊runs / 4⌋")
    st.dataframe(leader_df.style.format({"Runs":"{:d}", "Current Avg":"{:.1f}"}),
        use_container_width=True, height=min(600, 28*25))
with col2:
    st.subheader("Forecast to 10 runs  –  drop lowest 2")
    st.dataframe(forecast_df.style.format({
        "Predicted Avg":"{:.1f}",
        "CI Low":"{:.1f}", "CI High":"{:.1f}",
        "P(top 20)":"{:.1%}"}),
        use_container_width=True, height=min(600, 28*25))

with st.expander("Visual summaries"):
    # cut‑line histogram
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.hist(cut_line, bins=25, edgecolor="black")
    ax1.set_xlabel("20‑place cut‑line score")
    ax1.set_ylabel("Simulations")
    ax1.set_title("Cut‑line distribution (10 k runs)")
    st.pyplot(fig1)

    # probability bar chart
    top30 = forecast_df.head(30).iloc[::-1]
    fig2, ax2 = plt.subplots(figsize=(9, 0.28*len(top30)+1.2))
    ax2.barh(top30["Team"], top30["P(top 20)"])
    ax2.set_xlabel("Probability of top‑20")
    ax2.set_title("Top‑30 teams – forecast")
    st.pyplot(fig2)

# download
st.download_button("Download forecast CSV",
                   forecast_df.to_csv(index=False).encode(),
                   "predicted_rankings.csv","text/csv")
