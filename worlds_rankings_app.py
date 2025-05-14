# worlds_rankings_app.py  (interactive forecast + accurate leaderboard)
import datetime as dt, requests, streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# ─── PUT YOUR PERSONAL ROBOTEVENTS TOKEN HERE ───────────────────────────
RE_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzIiwianRpIjoiMjRkYjJkYjY3NTYwNTk3ZDgyYjQzZjhiMzU2ZmE4ODhkYTQ5MGU5MDZmNTIwZjdhNmJlMGY4NmRmNGQyNmRiMmE3ZjY3MTVjMGFiOTJjNWQiLCJpYXQiOjE3NDcxOTk5ODQuOTg0Mjc3LCJuYmYiOjE3NDcxOTk5ODQuOTg0Mjg4LCJleHAiOjI2OTM4ODQ3ODQuOTc4NTUsInN1YiI6IjExMTkyOCIsInNjb3BlcyI6W119.QAjwPG4z1s6D1OKU9zhtSCIVJicUkSSgnA9MgQBIlK9XomAt8OnE1i5-6GcutO8AKxIv5jU_auo5WGti0ms5ukmjV0b74GdXRyiVLd2yYSKzJwJo1nftGACKjG4YvNTjJEmmlx4Irs2rfCF7O6jJwsLt-up3gw8zHBHGJ-4j0Ras54AJ-_5fLNzW4NiY5yAx-UXa4CK2jCFqlaBbRNXhHesrFYKaxhhN4tBue6yl2XyX8Ry6CNWQVjhpVyHZdeTfB4qS01wGc8UphCv7Kb9zsuSrnS0QvbdR-HSMjbBv1p8SmnMLTXj_YvTUhDXg7F-qDyHnZzs59ohqSQQweNOn2LtxBH9GJh6y19Ma6z1pLjdzw8rl8MX-h1yBFl3Tmy3nrqKjkbJaPPo04Z7FOA29wVv9IFxRt78d4DI-xiJIB-NBkKiVjbidVB6RUtyP-AW-3_xOZAidjR3Rlij9jTSEtwb_Ln2JqZdJ6qURgqbxfoI-PlFm7075aGrO8GQ8W0I942UxBWhHANJ_mQDXRmbjMgYBU2MkM41ap60rvgzEW_UlUB6pAP8zia9PbhDhdleKPzmiqNmjprJ4Kmk9XJxpvpYSwAjL0B1E9vDbe18uUSkIGdg1AvWoDVmTlFcrP_NFJ6rR5A8OPv_RZSBqq4k4DKZyR1ilxWwX4q0JSglsgns"
# -----------------------------------------------------------------------

TOTAL_RUNS   = 10
FINAL_DROPS  = 2
REPS         = 10_000
RNG_SEED     = 42
DEFAULT_MEAN = 150.0
DEFAULT_SIGMA= 30.0
ONE_SIGMA    = 5.0
CACHE_TTL    = 600   # sec

st.set_page_config("VEX IQ Predictor", layout="wide")

# ── sidebar IDs ─────────────────────────────────────────────────────────
st.sidebar.header("RobotEvents IDs")
event_id    = st.sidebar.text_input("Event ID",    value="58913")
division_id = st.sidebar.text_input("Division ID", value="4")
if not (event_id and division_id):
    st.stop()

# ── fetch helper (cached) ───────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching RobotEvents …", ttl=CACHE_TTL)
def fetch_matches(eid: str, did: str, token: str) -> pd.DataFrame:
    url  = f"https://www.robotevents.com/api/v2/events/{eid}/divisions/{did}/matches"
    hdrs = {"accept":"application/json", "Authorization":f"Bearer {token}"}
    rows, page = [], url
    while page:
        r = requests.get(page, params={"round[]":2,"per_page":50},
                         headers=hdrs, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows.extend(data["data"])
        page = data["meta"]["next_page_url"]

    rec=[]
    for m in rows:
        red  = next(a for a in m["alliances"] if a["color"].lower()=="red")
        blue = next(a for a in m["alliances"] if a["color"].lower()=="blue")
        rec.append({
            "match":      m["matchnum"],
            "scored":     m["scored"],
            "Red Team":   red ["teams"][0]["team"]["name"]  if red ["teams"] else None,
            "Blue Team":  blue["teams"][0]["team"]["name"] if blue["teams"] else None,
            "Red Score":  red ["score"],
            "Blue Score": blue["score"],
        })
    df = pd.DataFrame(rec).sort_values("match", ignore_index=True)

    # Any row that isn't certified yet (scored=False) → treat scores as NaN
    uncert = ~df["scored"]
    df.loc[uncert, ["Red Score","Blue Score"]] = np.nan
    return df

try:
    df = fetch_matches(event_id.strip(), division_id.strip(), RE_TOKEN)
except Exception as ex:
    st.error(ex); st.stop()

st.caption(f"{len(df)} matches fetched  ({dt.datetime.utcnow():%Y-%m-%d %H:%MZ})")

score_cols = ["Red Score","Blue Score"]
team_cols  = ["Red Team","Blue Team"]

# ── editable table for future rounds (scored == False) ──────────────────
future_df = df[~df["scored"]].copy()

st.title("VEX IQ Predictor  –  leaderboard & forecast")
st.subheader("Enter predictions for still-to-play matches")
edited = st.data_editor(
    future_df,
    hide_index=True,
    column_config={c: st.column_config.NumberColumn(step=1) for c in score_cols},
)

# merge predictions back by match number & alliance
for col in score_cols:
    for match, val in zip(edited["match"], edited[col]):
        if not pd.isna(val):
            df.loc[(df["match"] == match), col] = val

# ── build per-team completed-run lists ──────────────────────────────────
per_team = {}
for _, row in df.iterrows():
    for s_col, t_col in zip(score_cols, team_cols):
        team = str(row[t_col]).strip()
        if not team or team.lower()=="nan": continue
        val  = row[s_col]
        if pd.isna(val): continue  # future & still blank
        # completed if score>0 OR certified flag True
        if val > 0 or row["scored"]:
            per_team.setdefault(team, []).append(float(val))

# ── leaderboard (drop ⌊n/4⌋) ───────────────────────────────────────────
leader=[]
for t, runs in per_team.items():
    drops=len(runs)//4
    avg=np.mean(sorted(runs)[drops:]) if runs else 0.0
    leader.append((t,len(runs),avg))
leader_df=(pd.DataFrame(leader,columns=["Team","Runs","Current Avg"])
           .sort_values("Current Avg",ascending=False)
           .reset_index(drop=True))

st.subheader("Current Leaderboard – drop ⌊runs / 4⌋")
st.dataframe(leader_df.head(30).style.format({
    "Runs":"{:d}","Current Avg":"{:.1f}"}), use_container_width=True,
    height=min(650,28*30+25))

# ── Monte-Carlo forecast ───────────────────────────────────────────────
teams   = sorted(per_team)
n_teams = len(teams)
n_played= {t:len(per_team[t]) for t in teams}
mu      = {t:(np.mean(per_team[t]) if per_team[t] else DEFAULT_MEAN) for t in teams}
sigma   = {t:(np.std(per_team[t],ddof=1) if len(per_team[t])>=2 else ONE_SIGMA)
           for t in teams}

rng=np.random.default_rng(RNG_SEED)
pred   = np.zeros((n_teams, REPS))
hits20 = np.zeros(n_teams,int)
cutln  = []

for _ in range(REPS):
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
    pred[:,_] = avgs

forecast_df=(pd.DataFrame({
    "Team":teams,
    "Predicted Avg": pred.mean(axis=1),
    "CI Low": np.percentile(pred,2.5,axis=1),
    "CI High":np.percentile(pred,97.5,axis=1),
    "P(Top 20)": hits20/REPS})
    .sort_values("P(Top 20)",ascending=False).reset_index(drop=True))

st.subheader("Forecast to 10 runs – drop lowest 2")
st.dataframe(forecast_df.style.format({
    "Predicted Avg":"{:.1f}","CI Low":"{:.1f}",
    "CI High":"{:.1f}","P(Top 20)":"{:.1%}"}),
    use_container_width=True,
    height=min(650,28*len(forecast_df)+25))

# ── visuals ────────────────────────────────────────────────────────────
with st.expander("Visual summaries"):
    fig1,ax1=plt.subplots(figsize=(8,4))
    ax1.hist(cutln,bins=25,edgecolor="black")
    ax1.set_xlabel("20-th place score"); ax1.set_ylabel("Simulations")
    ax1.set_title("Cut-line distribution"); st.pyplot(fig1)

    top30=forecast_df.head(30).iloc[::-1]
    fig2,ax2=plt.subplots(figsize=(9,0.28*len(top30)+1.2))
    ax2.barh(top30["Team"], top30["P(Top 20)"])
    ax2.set_xlabel("Probability of top-20"); ax2.set_title("Top-30 – forecast")
    st.pyplot(fig2)

# ── download ───────────────────────────────────────────────────────────
st.download_button("Download forecast CSV",
                   forecast_df.to_csv(index=False).encode(),
                   "predicted_rankings.csv","text/csv")
