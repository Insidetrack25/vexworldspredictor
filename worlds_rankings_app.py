# worlds_rankings_app.py
import datetime as dt, requests, streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# ─── PUT YOUR PERSONAL ROBOTEVENTS TOKEN HERE ───────────────────────────
RE_TOKEN = "YOUR_TOKEN_HERE"
# -----------------------------------------------------------------------

# ---------- GLOBAL CONSTANTS ----------
TOTAL_RUNS   = 10          # target runs per team
FINAL_DROPS  = 2           # drops in final standings
REPS         = 10_000      # Monte-Carlo replicates
RNG_SEED     = 42
DEFAULT_MEAN = 150.0
DEFAULT_SIGMA= 30.0
ONE_SIGMA    = 5.0
CACHE_TTL    = 600         # seconds cache RobotEvents JSON
# -------------------------------------

st.set_page_config("VEX IQ Predictor", layout="wide")

# ─── SIDEBAR – IDs & token ──────────────────────────────────────────────
st.sidebar.header("RobotEvents IDs")
event_id    = st.sidebar.text_input("Event ID",    value="58913")
division_id = st.sidebar.text_input("Division ID", value="4")

if not (event_id and division_id):
    st.stop()

# ─── FETCH MATCHES (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching RobotEvents …", ttl=CACHE_TTL)
def fetch_matches(eid: str, did: str, token: str) -> pd.DataFrame:
    url  = f"https://www.robotevents.com/api/v2/events/{eid}/divisions/{did}/matches"
    hdrs = {"accept":"application/json", "Authorization":f"Bearer {token}"}
    rows, page = [], url
    while page:
        r = requests.get(page, params={"round[]":2, "per_page":50},
                         headers=hdrs, timeout=15)
        r.raise_for_status()
        j = r.json()
        rows.extend(j["data"])
        page = j["meta"]["next_page_url"]

    rec = []
    for m in rows:
        red  = next(a for a in m["alliances"] if a["color"].lower()=="red")
        blue = next(a for a in m["alliances"] if a["color"].lower()=="blue")
        rec.append({
            "match":      m["matchnum"],
            "scored":     m["scored"],          # certified flag
            "Red Team":   red ["teams"][0]["team"]["name"]  if red ["teams"] else None,
            "Blue Team":  blue["teams"][0]["team"]["name"] if blue["teams"] else None,
            "Red Score":  red ["score"],
            "Blue Score": blue["score"],
        })
    return pd.DataFrame(rec).sort_values("match", ignore_index=True)

try:
    df = fetch_matches(event_id.strip(), division_id.strip(), RE_TOKEN)
except Exception as ex:
    st.error(ex); st.stop()

st.caption(f"{len(df)} match rows fetched  "
           f"({dt.datetime.utcnow():%Y-%m-%d %H:%MZ})")

score_cols = ["Red Score", "Blue Score"]
team_cols  = ["Red Team",  "Blue Team"]

# ─── EDITOR FOR FUTURE ROUNDS ───────────────────────────────────────────
future_mask = df.apply(lambda r: pd.isna(r["Red Score"]) and pd.isna(r["Blue Score"]) and not r["scored"], axis=1)
future_df   = df[future_mask].copy()

st.title("VEX IQ Predictor  –  leaderboard & forecast")

st.markdown(
"*Blank cells* below are future matches – type predictions if you wish.  \n"
"A run is counted for an alliance only when its **score > 0** or the match "
"is **certified (`scored = true`)** (which also includes legit 0-point runs)."
)

edited = st.data_editor(
    future_df,
    hide_index=True,
    column_config={c: st.column_config.NumberColumn(step=1) for c in score_cols},
)
df.update(edited)                     # merge predictions back

# ─── BUILD per-team score lists ─────────────────────────────────────────
per_team = {}
for _, row in df.iterrows():
    for s_col, t_col in zip(score_cols, team_cols):
        team = str(row[t_col]).strip()
        if not team or team.lower()=="nan": 
            continue

        val = row[s_col]

        completed = (
            (not pd.isna(val) and val > 0) or           # positive score
            (row["scored"] and not pd.isna(val))        # certified (0 or >0)
        )
        if not completed:
            continue

        per_team.setdefault(team, []).append(float(val))

# ─── CURRENT LEADERBOARD  (drop ⌊n/4⌋) ─────────────────────────────────
leader_rows=[]
for t,scores in per_team.items():
    drops=len(scores)//4
    avg=np.mean(sorted(scores)[drops:]) if scores else 0.0
    leader_rows.append((t,len(scores),avg))

leader_df=(pd.DataFrame(leader_rows,columns=["Team","Runs","Current Avg"])
           .sort_values("Current Avg",ascending=False)
           .reset_index(drop=True))

st.subheader("Current Leaderboard – drop ⌊runs / 4⌋")
st.dataframe(
    leader_df.head(30).style.format({"Runs":"{:d}","Current Avg":"{:.1f}"}),
    use_container_width=True,
    height=min(650,28*30+25)
)

# ─── PREP FOR MONTE-CARLO ──────────────────────────────────────────────
teams   = sorted(per_team)
n_teams = len(teams)

n_played = {t: len(per_team[t]) for t in teams}
mu   = {t: (np.mean(per_team[t]) if per_team[t] else DEFAULT_MEAN) for t in teams}
sigma={}
for t in teams:
    arr=np.array(per_team[t])
    sigma[t]=np.std(arr,ddof=1) if len(arr)>=2 else ONE_SIGMA

rng=np.random.default_rng(RNG_SEED)
pred   = np.zeros((n_teams, REPS))
hits20 = np.zeros(n_teams,int)
cutln  = []

for k in range(REPS):
    avgs=np.empty(n_teams)
    for i,t in enumerate(teams):
        need=max(0,TOTAL_RUNS-n_played[t])
        fut=rng.normal(mu[t], sigma[t], need) if need else np.empty(0)
        all_scores=np.concatenate([per_team[t], fut])
        all_scores.sort()
        avgs[i]=all_scores[FINAL_DROPS:].mean()
    order=np.argsort(-avgs)
    hits20[order[:20]]+=1
    cutln.append(avgs[order[19]])
    pred[:,k]=avgs

forecast_df=(pd.DataFrame({
    "Team":teams,
    "Predicted Avg":pred.mean(axis=1),
    "CI Low": np.percentile(pred, 2.5, axis=1),
    "CI High":np.percentile(pred,97.5,axis=1),
    "P(Top 20)": hits20/REPS})
    .sort_values("P(Top 20)",ascending=False)
    .reset_index(drop=True))

st.subheader("Forecast to 10 runs – drop lowest 2")
st.dataframe(
    forecast_df.style.format({
        "Predicted Avg":"{:.1f}","CI Low":"{:.1f}",
        "CI High":"{:.1f}","P(Top 20)":"{:.1%}"}),
    use_container_width=True,
    height=min(650,28*len(forecast_df)+25)
)

# ─── VISUAL SUMMARIES ──────────────────────────────────────────────────
with st.expander("Visual summaries"):
    fig1,ax1=plt.subplots(figsize=(8,4))
    ax1.hist(cutln,bins=25,edgecolor="black")
    ax1.set_xlabel("Score of 20-th place team"); ax1.set_ylabel("Simulations")
    ax1.set_title("Cut-line distribution"); st.pyplot(fig1)

    top30=forecast_df.head(30).iloc[::-1]
    fig2,ax2=plt.subplots(figsize=(9,0.28*len(top30)+1.2))
    ax2.barh(top30["Team"], top30["P(Top 20)"])
    ax2.set_xlabel("Probability of top-20"); ax2.set_title("Top-30 – forecast")
    st.pyplot(fig2)

# ─── DOWNLOAD ──────────────────────────────────────────────────────────
st.download_button("Download forecast CSV",
                   forecast_df.to_csv(index=False).encode(),
                   "predicted_rankings.csv","text/csv")
