# worlds_rankings_app.py
"""
VEX IQ Predictor
• Accurate leaderboard (drop ⌊runs/4⌋)
• Interactive predictions for future rounds
• 10 000-rep Monte-Carlo forecast to 10 runs (drop lowest 2)
• 95 % CIs, P(top-20), cut-line histogram, top-30 probability bars
"""

# ────────── imports ──────────
import datetime as dt, requests, streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# ────────── constants (must appear first) ──────────
RE_TOKEN     = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzIiwianRpIjoiMjRkYjJkYjY3NTYwNTk3ZDgyYjQzZjhiMzU2ZmE4ODhkYTQ5MGU5MDZmNTIwZjdhNmJlMGY4NmRmNGQyNmRiMmE3ZjY3MTVjMGFiOTJjNWQiLCJpYXQiOjE3NDcxOTk5ODQuOTg0Mjc3LCJuYmYiOjE3NDcxOTk5ODQuOTg0Mjg4LCJleHAiOjI2OTM4ODQ3ODQuOTc4NTUsInN1YiI6IjExMTkyOCIsInNjb3BlcyI6W119.QAjwPG4z1s6D1OKU9zhtSCIVJicUkSSgnA9MgQBIlK9XomAt8OnE1i5-6GcutO8AKxIv5jU_auo5WGti0ms5ukmjV0b74GdXRyiVLd2yYSKzJwJo1nftGACKjG4YvNTjJEmmlx4Irs2rfCF7O6jJwsLt-up3gw8zHBHGJ-4j0Ras54AJ-_5fLNzW4NiY5yAx-UXa4CK2jCFqlaBbRNXhHesrFYKaxhhN4tBue6yl2XyX8Ry6CNWQVjhpVyHZdeTfB4qS01wGc8UphCv7Kb9zsuSrnS0QvbdR-HSMjbBv1p8SmnMLTXj_YvTUhDXg7F-qDyHnZzs59ohqSQQweNOn2LtxBH9GJh6y19Ma6z1pLjdzw8rl8MX-h1yBFl3Tmy3nrqKjkbJaPPo04Z7FOA29wVv9IFxRt78d4DI-xiJIB-NBkKiVjbidVB6RUtyP-AW-3_xOZAidjR3Rlij9jTSEtwb_Ln2JqZdJ6qURgqbxfoI-PlFm7075aGrO8GQ8W0I942UxBWhHANJ_mQDXRmbjMgYBU2MkM41ap60rvgzEW_UlUB6pAP8zia9PbhDhdleKPzmiqNmjprJ4Kmk9XJxpvpYSwAjL0B1E9vDbe18uUSkIGdg1AvWoDVmTlFcrP_NFJ6rR5A8OPv_RZSBqq4k4DKZyR1ilxWwX4q0JSglsgns"    # ← put your RobotEvents API token here
TOTAL_RUNS   = 10                   # target runs per team
FINAL_DROPS  = 2                    # drop lowest 2 in final standings
REPS         = 10_000               # Monte-Carlo replicates
RNG_SEED     = 42
ONE_SIGMA    = 5.0                  # stdev fallback when a team has <2 runs
CACHE_TTL    = 600                  # sec to cache RobotEvents JSON
# ───────────────────────────────────────────────────

st.set_page_config("VEX IQ Predictor", layout="wide")

# ——— sidebar IDs ———
st.sidebar.header("RobotEvents IDs")
event_id    = st.sidebar.text_input("Event ID",    value="58913")
division_id = st.sidebar.text_input("Division ID", value="4")
if not (event_id and division_id):
    st.stop()

# ——— fetch helper ———
@st.cache_data(show_spinner="Fetching RobotEvents …", ttl=CACHE_TTL)
def pull_matches(eid, did, token):
    url = f"https://www.robotevents.com/api/v2/events/{eid}/divisions/{did}/matches"
    hdr = {"accept":"application/json", "Authorization":f"Bearer {token}"}
    rows, page = [], url
    while page:
        r = requests.get(page, params={"round[]":2,"per_page":50},
                         headers=hdr, timeout=15)
        r.raise_for_status()
        j = r.json(); rows += j["data"]; page = j["meta"]["next_page_url"]

    rec=[]
    for m in rows:
        red  = next(a for a in m["alliances"] if a["color"].lower()=="red")
        blue = next(a for a in m["alliances"] if a["color"].lower()=="blue")
        rec.append({
            "match":     m["matchnum"],
            "scored":    m["scored"],        # certified flag
            "Red Team":  red ["teams"][0]["team"]["name"]  if red ["teams"] else None,
            "Blue Team": blue["teams"][0]["team"]["name"] if blue["teams"] else None,
            "Red Score": red ["score"],
            "Blue Score":blue["score"],
        })
    df = pd.DataFrame(rec).sort_values("match", ignore_index=True)

    # any non-certified row → blank scores so the user can edit
    unc = ~df["scored"]
    df.loc[unc, ["Red Score","Blue Score"]] = np.nan
    return df

try:
    df = pull_matches(event_id.strip(), division_id.strip(), RE_TOKEN)
except Exception as e:
    st.error(e); st.stop()

st.caption(f"{len(df)} matches fetched  ({dt.datetime.utcnow():%H:%MZ})")

score_cols = ["Red Score","Blue Score"]
team_cols  = ["Red Team","Blue Team"]

# ——— editor for future rounds ———
future = df[df[score_cols].isna().all(axis=1)].copy()

st.title("VEX IQ Predictor  –  leaderboard & forecast")
st.subheader("Type predictions for un-played matches")
edited = st.data_editor(
    future,
    hide_index=True,
    use_container_width=True,
    column_config={c: st.column_config.NumberColumn(step=1) for c in score_cols},
)

# merge predictions by match number
for col in score_cols:
    df.loc[df["match"].isin(edited["match"]), col] = edited.set_index("match")[col]

# ——— build per-team completed score lists ———
per_team = {}
for _, row in df.iterrows():
    for s_col, t_col in zip(score_cols, team_cols):
        team = str(row[t_col]).strip()
        if not team or team.lower()=="nan": continue
        val = row[s_col]
        completed = (
            (not pd.isna(val) and val > 0) or    # positive score
            (row["scored"] and not pd.isna(val)) # certified (0 or >0)
        )
        if completed:
            per_team.setdefault(team, []).append(float(val))

# ——— current leaderboard (drop ⌊n/4⌋) ———
lead = []
for t, scores in per_team.items():
    drops = len(scores)//4
    avg   = np.mean(sorted(scores)[drops:]) if scores else 0.0
    lead.append((t,len(scores),avg))
leader_df=(pd.DataFrame(lead, columns=["Team","Runs","Current Avg"])
           .sort_values("Current Avg",ascending=False).reset_index(drop=True))

st.subheader("Current Leaderboard – drop ⌊runs / 4⌋")
st.dataframe(
    leader_df.head(30).style.format({"Runs":"{:d}","Current Avg":"{:.1f}"}),
    use_container_width=True, height=650)

# ——— Monte-Carlo forecast ———
teams   = sorted(per_team)
n_teams = len(teams)

if n_teams == 0:
    st.warning("No completed runs yet – forecast will appear once scores are posted.")
    st.stop()

played  = {t: len(per_team[t]) for t in teams}
mu      = {t: np.mean(per_team[t]) for t in teams}
sigma   = {t: (np.std(per_team[t],ddof=1) if len(per_team[t])>=2 else ONE_SIGMA)
           for t in teams}

rng   = np.random.default_rng(RNG_SEED)
pred  = np.zeros((n_teams, REPS))
hits  = np.zeros(n_teams, int)
cutln = []

for k in range(REPS):
    avgs = np.empty(n_teams)
    for i, t in enumerate(teams):
        need = max(0, TOTAL_RUNS - played[t])
        fut  = rng.normal(mu[t], sigma[t], need) if need else np.empty(0)
        all_scores = np.concatenate([per_team[t], fut])
        all_scores.sort()
        avgs[i] = all_scores[FINAL_DROPS:].mean()

    order = np.argsort(-avgs)
    hits[order[:20]] += 1
    cutln.append(avgs[order[19]] if len(order) >= 20 else np.nan)
    pred[:, k] = avgs

forecast_df = (pd.DataFrame({
                "Team": teams,
                "Predicted Avg": pred.mean(axis=1),
                "CI Low":  np.percentile(pred,  2.5, axis=1),
                "CI High": np.percentile(pred, 97.5, axis=1),
                "P(Top 20)": hits / REPS})
              .sort_values("P(Top 20)", ascending=False)
              .reset_index(drop=True))

st.subheader("Forecast to 10 runs – drop lowest 2")
st.dataframe(
    forecast_df.style.format({
        "Predicted Avg":"{:.1f}","CI Low":"{:.1f}",
        "CI High":"{:.1f}","P(Top 20)":"{:.1%}"}),
    use_container_width=True,
    height=min(650, 28*len(forecast_df)+25))

# ——— visuals ———
with st.expander("Visual summaries"):
    valid_cut = [x for x in cutln if not np.isnan(x)]
    if valid_cut:
        fig1,ax1 = plt.subplots(figsize=(8,4))
        ax1.hist(valid_cut, bins=25, edgecolor="black")
        ax1.set_xlabel("20-th place score"); ax1.set_ylabel("Simulations")
        ax1.set_title("Cut-line distribution"); st.pyplot(fig1)

    top30 = forecast_df.head(30).iloc[::-1]
    fig2,ax2 = plt.subplots(figsize=(9, 0.28*len(top30)+1.2))
    ax2.barh(top30["Team"], top30["P(Top 20)"])
    ax2.set_xlabel("Probability of top-20")
    ax2.set_title("Top-30 teams – forecast")
    st.pyplot(fig2)

# ——— download CSV ———
st.download_button("Download forecast CSV",
                   forecast_df.to_csv(index=False).encode(),
                   "predicted_rankings.csv",
                   "text/csv")
