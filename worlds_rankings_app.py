"""
VEX IQ Worlds – Top-20 predictor (token embedded)

* Pulls Teamwork-qualification matches from RobotEvents
  https://www.robotevents.com/api/v2/events/{event}/divisions/{div}/matches?round[]=2
* Lets you fill in scores still to be played
* Monte-Carlo (drop 2 of 10) → predicted avg, 95 % CI, P(top-20)
"""

import datetime as dt
from pathlib import Path
import requests, streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt

# ───►  PUT YOUR TOKEN HERE  ◄─────────────────────────────────────────────
RE_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIzIiwianRpIjoiMTgxZjk1ODg3YmZmOGJkMTc5ODdmZWI1MzhhNWVlMzIzZjcyNTBiNjMwZDRkOTI3OGE3MWZhMTMzNGI5OGJiYzU5ZTQ1OTFlMzhiMThjOTUiLCJpYXQiOjE3NDcxOTg0MDAuNDQ5MDQ0OSwibmJmIjoxNzQ3MTk4NDAwLjQ0OTA0NTksImV4cCI6MjY5Mzg4MzIwMC40NDQ3NzE4LCJzdWIiOiIxMTE5MjgiLCJzY29wZXMiOltdfQ.bw4iODUD-TjkUd5g37wsdlNwHShOF1kzJwJr5s3aCZdo4XBfoVGo_d4Bco9rq83-8VTgVQsWJhiy8VG4Yq4rulIwfiletZrhDVDkg4Vj9cSlgTQ5P7uy1d89eXy4XNCEDkv7-bxWSRgrov1tuReXOXryqYgA4-6rT-tC_x2zxtrwlqx1ci6-98vBPQFlnnIyr-BW8z3_0LdljQ7bTgodrawQOg1Lfem_STOYYv8K5G6LkJnHBsH9hLc6FC74O8YoN1UbVvcz3R83ox6b31UdvdE0upAw_hpuiZ-14h5zzaDHzAIegjgvsSamSjcGuIFQTOdH85fjZGavASlqabf3u-wqTVpBVdAFChYpCg5__NlXc9BlYpWPW38bZc_XbivjjmYBt0eHmuJnxrNGU3ameC0WLaNhHun1Mky2GpiUJaBoj-1PjJdVWYcAwBUO73ogZh7EUp_ElD_jZ-v01WIUeRQv2DrsaSO0Ww2fMBXpHfFQlkVnt1oX80V0ngqBKzVfLmel-t5qBhZZy2El0lFzaNb08Tq6cWlEXkimYQPzmCft5UPun0rAAdRVJ-cBFOCaE4g6ceb1K-R2fOltkh0YXOIwjKS0vC4mX0pvsQ8nMl1UQ4iT6rD2iSfRlI22g9qy0z9psulZBJ00egI4X0RrLnrhJn0UTXs-juI0SsevusE"
# ─────────────────────────────────────────────────────────────────────────

TOTAL_RUNS   = 10
DROPS        = 2
REPS         = 10_000
RNG_SEED     = 42
DEFAULT_SIGMA = 5.0
CACHE_TTL    = 600
CSV_FALLBACK = "Worlds Design2.csv"

st.set_page_config("Top-20 Predictor", layout="wide")

# ═════════════ Sidebar – input Event & Division ════════════════════════
event_id    = st.sidebar.text_input("Event ID",    placeholder="58913")
division_id = st.sidebar.text_input("Division ID", placeholder="4")
use_api     = bool(event_id and division_id)

if use_api:
    st.sidebar.success("Using live RobotEvents data")
else:
    st.sidebar.warning("Enter Event + Division IDs or CSV fallback is used")

# ═════════════ Fetch matches (cached) ═════════════════════════════════=
@st.cache_data(show_spinner="Fetching RobotEvents …", ttl=CACHE_TTL)
def fetch_matches(event_id: str, div_id: str, token: str) -> pd.DataFrame:
    base = (f"https://www.robotevents.com/api/v2/events/{event_id}"
            f"/divisions/{div_id}/matches")
    params  = {"round[]": 2, "per_page": 50}
    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}

    rows, url = [], base
    while url:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"API {r.status_code}: {r.text[:200]}")
        j = r.json()
        rows.extend(j["data"])
        url, params = j["meta"]["next_page_url"], None

    if not rows:
        raise ValueError("No matches returned – check IDs or token")

    recs = []
    for m in rows:
        red, blue = None, None
        for a in m["alliances"]:
            if a["color"].lower() == "red":  red = a
            if a["color"].lower() == "blue": blue = a
        recs.append({
            "match":       m["matchnum"],
            "Red Team 1":  red["teams"][0]["team"]["name"]  if red["teams"]  else None,
            "Blue Team 1": blue["teams"][0]["team"]["name"] if blue["teams"] else None,
            "Red Score":   red["score"],
            "Blue Score":  blue["score"],
        })
    return pd.DataFrame(recs).sort_values("match", ignore_index=True)

# choose live API or CSV fallback
if use_api:
    try:
        raw = fetch_matches(event_id.strip(), division_id.strip(), RE_TOKEN)
        st.caption(f"📶  Pulled {len(raw)} matches "
                   f"({dt.datetime.utcnow():%Y-%m-%d %H:%MZ})")
    except Exception as e:
        st.error(f"RobotEvents fetch failed → {e}")
        st.stop()
else:
    csv = Path(__file__).with_name(CSV_FALLBACK)
    if not csv.exists():
        st.error("No API IDs supplied and CSV fallback missing.")
        st.stop()
    raw = pd.read_csv(csv)
    st.caption(f"📄  Loaded {len(raw)} rows from {CSV_FALLBACK}")

score_cols = ["Red Score", "Blue Score"]
team_cols  = ["Red Team 1", "Blue Team 1"]

# ═════ split played vs un-played ═════
todo_df = raw[raw[score_cols].isna().any(axis=1)].copy()
st.title("VEX IQ Worlds – interactive top-20 predictor")
st.markdown(
"Enter **expected scores** for the un-played matches below. "
"Leave blank → simulator samples that alliance’s score from its own μ ± σ."
)

editable = st.data_editor(
    todo_df,
    hide_index=True,
    num_rows="dynamic",
    column_config={c: st.column_config.NumberColumn(step=1) for c in score_cols},
)
raw.update(editable)

# ═════ build per-team score lists ═════
scores = {}
for _, r in raw.iterrows():
    for col in ("Red", "Blue"):
        t = str(r[f"{col} Team 1"]).strip()
        if not t or t.lower() == "nan":
            continue
        val = r[f"{col} Score"]
        if pd.isna(val):
            continue
        scores.setdefault(t, []).append(float(val))

teams = sorted(scores)
mu, sigma, n_played = {}, {}, {}
for t in teams:
    arr = np.asarray(scores[t], float)
    n_played[t] = len(arr)
    mu[t]    = arr.mean() if arr.size else 150.0
    sigma[t] = arr.std(ddof=1) if arr.size > 1 else DEFAULT_SIGMA

# ═════ Monte-Carlo ═════
rng = np.random.default_rng(RNG_SEED)
preds = np.zeros((len(teams), REPS))
hits  = np.zeros(len(teams), int)
cut   = []

for k in range(REPS):
    avgs = np.empty(len(teams))
    for i, t in enumerate(teams):
        fut = rng.normal(mu[t], sigma[t], TOTAL_RUNS - n_played[t])
        all_scores = np.concatenate([scores[t], fut])
        all_scores.sort()
        avgs[i] = all_scores[DROPS:].mean()
    order = np.argsort(-avgs)
    hits[order[:20]] += 1
    cut.append(avgs[order[19]])
    preds[:, k] = avgs

df = pd.DataFrame({
        "Team": teams,
        "Predicted Avg": preds.mean(axis=1),
        "CI Low":  np.percentile(preds,  2.5, axis=1),
        "CI High": np.percentile(preds, 97.5, axis=1),
        "P(Top 20)": hits / REPS}).sort_values("P(Top 20)", ascending=False)

# ═════ table ═════
st.subheader("Projected qualification table")
st.dataframe(
    df.style.format({"Predicted Avg":"{:.1f}",
                     "CI Low":"{:.1f}",
                     "CI High":"{:.1f}",
                     "P(Top 20)":"{:.1%}"}),
    use_container_width=True)

# ═════ visuals ═════
with st.expander("Visual summaries"):
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.hist(cut, bins=25, edgecolor="black")
    ax1.set_xlabel("Score of 20-th-place team")
    ax1.set_ylabel("Simulations")
    ax1.set_title("Cut-line distribution")
    st.pyplot(fig1)

    top30 = df.head(30).iloc[::-1]
    fig2, ax2 = plt.subplots(figsize=(9, 0.28*len(top30)+1.2))
    ax2.barh(top30["Team"], top30["P(Top 20)"])
    ax2.set_xlabel("Probability of finishing in top 20")
    ax2.set_title("Top-30 teams")
    st.pyplot(fig2)

st.download_button("Download CSV",
                   df.to_csv(index=False).encode(),
                   "predicted_rankings.csv", "text/csv")
