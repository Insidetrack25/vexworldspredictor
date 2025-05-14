# worlds_rankings_app.py
import datetime as dt, requests, streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# ─── put your personal API token here ───────────────────────────────────
RE_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzIiwianRpIjoiMjRkYjJkYjY3NTYwNTk3ZDgyYjQzZjhiMzU2ZmE4ODhkYTQ5MGU5MDZmNTIwZjdhNmJlMGY4NmRmNGQyNmRiMmE3ZjY3MTVjMGFiOTJjNWQiLCJpYXQiOjE3NDcxOTk5ODQuOTg0Mjc3LCJuYmYiOjE3NDcxOTk5ODQuOTg0Mjg4LCJleHAiOjI2OTM4ODQ3ODQuOTc4NTUsInN1YiI6IjExMTkyOCIsInNjb3BlcyI6W119.QAjwPG4z1s6D1OKU9zhtSCIVJicUkSSgnA9MgQBIlK9XomAt8OnE1i5-6GcutO8AKxIv5jU_auo5WGti0ms5ukmjV0b74GdXRyiVLd2yYSKzJwJo1nftGACKjG4YvNTjJEmmlx4Irs2rfCF7O6jJwsLt-up3gw8zHBHGJ-4j0Ras54AJ-_5fLNzW4NiY5yAx-UXa4CK2jCFqlaBbRNXhHesrFYKaxhhN4tBue6yl2XyX8Ry6CNWQVjhpVyHZdeTfB4qS01wGc8UphCv7Kb9zsuSrnS0QvbdR-HSMjbBv1p8SmnMLTXj_YvTUhDXg7F-qDyHnZzs59ohqSQQweNOn2LtxBH9GJh6y19Ma6z1pLjdzw8rl8MX-h1yBFl3Tmy3nrqKjkbJaPPo04Z7FOA29wVv9IFxRt78d4DI-xiJIB-NBkKiVjbidVB6RUtyP-AW-3_xOZAidjR3Rlij9jTSEtwb_Ln2JqZdJ6qURgqbxfoI-PlFm7075aGrO8GQ8W0I942UxBWhHANJ_mQDXRmbjMgYBU2MkM41ap60rvgzEW_UlUB6pAP8zia9PbhDhdleKPzmiqNmjprJ4Kmk9XJxpvpYSwAjL0B1E9vDbe18uUSkIGdg1AvWoDVmTlFcrP_NFJ6rR5A8OPv_RZSBqq4k4DKZyR1ilxWwX4q0JSglsgns"
# -----------------------------------------------------------------------

# ---------- constants ----------
TOTAL_RUNS   = 10         # target runs/team
FINAL_DROPS  = 2          # drop in final standings
REPS         = 10_000     # MC reps
RNG_SEED     = 42
ONE_SIGMA    = 5.0
CACHE_TTL    = 600
# --------------------------------

st.set_page_config("VEX IQ Predictor", layout="wide")

# ─── sidebar IDs ────────────────────────────────────────────────────────
st.sidebar.header("RobotEvents IDs")
event_id    = st.sidebar.text_input("Event ID",    value="58913")
division_id = st.sidebar.text_input("Division ID", value="4")
if not (event_id and division_id):
    st.stop()

# ─── fetch helper (cached) ──────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching RobotEvents …", ttl=CACHE_TTL)
def pull_matches(eid, did, token):
    url  = f"https://www.robotevents.com/api/v2/events/{eid}/divisions/{did}/matches"
    hdr  = {"accept":"application/json", "Authorization":f"Bearer {token}"}
    rows, page = [], url
    while page:
        r = requests.get(page, params={"round[]":2,"per_page":50}, headers=hdr, timeout=15)
        r.raise_for_status()
        j = r.json(); rows += j["data"]; page = j["meta"]["next_page_url"]
    rec=[]
    for m in rows:
        red  = next(a for a in m["alliances"] if a["color"].lower()=="red")
        blue = next(a for a in m["alliances"] if a["color"].lower()=="blue")
        rec.append({
            "match": m["matchnum"],
            "scored": m["scored"],
            "Red Team":  red ["teams"][0]["team"]["name"]  if red ["teams"] else None,
            "Blue Team": blue["teams"][0]["team"]["name"] if blue["teams"] else None,
            "Red Score": red ["score"],
            "Blue Score":blue["score"],
        })
    df = pd.DataFrame(rec).sort_values("match", ignore_index=True)

    # force scores to NaN for not-yet-certified rows so they appear blank
    mask = ~df["scored"]
    df.loc[mask, ["Red Score","Blue Score"]] = np.nan
    return df

try:
    df = pull_matches(event_id.strip(), division_id.strip(), RE_TOKEN)
except Exception as e:
    st.error(e); st.stop()

st.caption(f"Fetched {len(df)} matches  ({dt.datetime.utcnow():%H:%MZ})")

score_cols, team_cols = ["Red Score","Blue Score"], ["Red Team","Blue Team"]

# ─── editor for future rounds (scores still NaN) ────────────────────────
future = df[df[score_cols].isna().all(axis=1)].copy()
st.title("VEX IQ Predictor – leaderboard & forecast")
st.subheader("Type predictions for un-played matches")

edited = st.data_editor(
    future,
    use_container_width=True,
    hide_index=True,
    column_config={c: st.column_config.NumberColumn(step=1) for c in score_cols},
)
# merge predictions back
for col in score_cols:
    df.loc[df["match"].isin(edited["match"]), col] = edited.set_index("match")[col]

# ─── build per-team completed scores ────────────────────────────────────
per_team = {}
for _, row in df.iterrows():
    for s_col, t_col in zip(score_cols, team_cols):
        team = str(row[t_col]).strip()
        if not team or team.lower()=="nan": continue
        s = row[s_col]
        # count if positive OR certified row with numeric (0-pt possible)
        if (not pd.isna(s) and (s > 0 or row["scored"])):
            per_team.setdefault(team, []).append(float(s))

# ─── leaderboard (drop ⌊n/4⌋) ───────────────────────────────────────────
lead=[]
for t, runs in per_team.items():
    drops=len(runs)//4
    avg = np.mean(sorted(runs)[drops:]) if runs else 0.0
    lead.append((t,len(runs),avg))
leader = (pd.DataFrame(lead, columns=["Team","Runs","Current Avg"])
          .sort_values("Current Avg",ascending=False).reset_index(drop=True))

st.subheader("Current leaderboard – drop ⌊runs / 4⌋")
st.dataframe(leader.head(30).style.format({"Runs":"{:d}","Current Avg":"{:.1f}"}),
             use_container_width=True, height=650)

# ─── Monte-Carlo forecast to 10 runs (drop 2) ───────────────────────────
rng = np.random.default_rng(RNG_SEED)
pred   = np.zeros((n_teams, REPS))
hits20 = np.zeros(n_teams, int)
cutln  = []

for k in range(REPS):
    avgs = np.empty(n_teams)
    for i, t in enumerate(teams):
        need = max(0, TOTAL_RUNS - played[t])
        fut  = rng.normal(mu[t], sigma[t], need) if need else np.empty(0)
        all_ = np.concatenate([per_team[t], fut])
        all_.sort()
        avgs[i] = all_[FINAL_DROPS:].mean()

    order = np.argsort(-avgs)        # indices, best → worst
    hits20[order[:20]] += 1          # OK even when <20

    # store cut-line only if 20 teams exist
    cutln.append(avgs[order[19]] if len(order) >= 20 else np.nan)

    pred[:, k] = avgs
# ─── visual summaries ──────────────────────────────────────────────────
with st.expander("Visual summaries"):
    fig1,ax1=plt.subplots(figsize=(8,4))
    ax1.hist(cutln,bins=25,edgecolor="black")
    ax1.set_xlabel("20-th place score"); ax1.set_ylabel("Simulations")
    ax1.set_title("Cut-line distribution"); st.pyplot(fig1)

    top30 = forecast.head(30).iloc[::-1]
    fig2,ax2=plt.subplots(figsize=(9,0.28*len(top30)+1.2))
    ax2.barh(top30["Team"], top30["P(Top 20)"])
    ax2.set_xlabel("Probability of top-20")
    ax2.set_title("Top-30 teams – forecast"); st.pyplot(fig2)

# ─── download ──────────────────────────────────────────────────────────
st.download_button("Download forecast CSV",
                   forecast.to_csv(index=False).encode(),
                   "predicted_rankings.csv","text/csv")
