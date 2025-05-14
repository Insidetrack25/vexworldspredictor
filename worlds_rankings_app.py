"""
VEX IQ Worlds – interactive top-20 probability app
• Pulls live Teamwork-qualification matches from RobotEvents
• Lets you edit still-to-play scores
• Monte-Carlo forecast (drop 2 of 10 runs)
• 95 % confidence intervals & P(top-20)
"""

from pathlib import Path
import datetime as dt
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────── Settings ──
TOTAL_RUNS   = 10            # planned runs per team
DROPS        = 2             # # worst scores to discard
REPS         = 10_000        # Monte-Carlo replicates
RNG_SEED     = 42
DEFAULT_SIGMA = 5.0          # fallback stdev for 1-run teams
CSV_FALLBACK = "Worlds Design2.csv"   # local file if API not used
CACHE_TTL    = 600           # seconds RobotEvents JSON is cached
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config("Top-20 Predictor", layout="wide")

# ───────────── Sidebar – choose data source ──────────────
st.sidebar.header("RobotEvents source")
event_id    = st.sidebar.text_input("Event ID",    placeholder="58913")
division_id = st.sidebar.text_input("Division ID", placeholder="4")
use_api     = event_id.strip() != "" and division_id.strip() != ""

st.sidebar.markdown(
"""
*Event ID & Division ID are at the end of an event URL,  
e.g. `.../events/58913/divisions/4`.  
The endpoint is public – **no token required.***
""",
    help="Qualification Teamwork rounds are `round[]=2`."
)

# ═════════════════════ 1  FETCH MATCHES ═════════════════════
@st.cache_data(show_spinner="Fetching RobotEvents …", ttl=CACHE_TTL)
def fetch_matches(event_id: str, division_id: str) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
        match, Red Team 1, Blue Team 1, Red Score, Blue Score
    containing ALL Teamwork-qualification matches (round = 2)
    for the specified event & division.
    """
    base = (f"https://www.robotevents.com/api/v2/events/{event_id}"
            f"/divisions/{division_id}/matches")
    params = {"round[]": 2, "per_page": 50}
    hdrs   = {"accept": "application/json"}

    rows, url = [], base
    while url:
        r = requests.get(url, params=params, headers=hdrs, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"RobotEvents error {r.status_code}: {r.text[:150]}")
        j = r.json()
        rows.extend(j["data"])
        url, params = j["meta"]["next_page_url"], None   # params only 1st call

    if not rows:
        raise ValueError("No matches returned – check IDs or round filter")

    tbl = []
    for m in rows:
        red, blue = None, None
        for a in m["alliances"]:
            if a["color"].lower() == "red":
                red = a
            elif a["color"].lower() == "blue":
                blue = a
        if red is None or blue is None:
            continue
        tbl.append({
            "match":       m["matchnum"],
            "Red Team 1":  red["teams"][0]["team"]["name"] if red["teams"] else None,
            "Blue Team 1": blue["teams"][0]["team"]["name"] if blue["teams"] else None,
            "Red Score":   red["score"],
            "Blue Score":  blue["score"],
        })

    return pd.DataFrame(tbl).sort_values("match", ignore_index=True)


# choose API or CSV fallback
if use_api:
    try:
        raw = fetch_matches(event_id.strip(), division_id.strip())
        st.caption(f"📶 Pulled {len(raw)} matches from RobotEvents "
                   f"({dt.datetime.utcnow():%Y-%m-%d %H:%MZ})")
    except Exception as exc:
        st.error(f"Couldn’t fetch API data → {exc}")
        st.stop()
else:
    csv_path = Path(__file__).with_name(CSV_FALLBACK)
    if not csv_path.exists():
        st.error("No API IDs given and CSV fallback missing.")
        st.stop()
    raw = pd.read_csv(csv_path)
    st.caption(f"📄 Using {len(raw)} rows from {CSV_FALLBACK}")

score_cols = ["Red Score", "Blue Score"]
team_cols  = ["Red Team 1", "Blue Team 1"]

# ═════════════ 2  Split played vs un-played ════════════════
played_df = raw.dropna(subset=score_cols)
todo_df   = raw[raw[score_cols].isna().any(axis=1)].copy()

st.title("VEX IQ Worlds – interactive top-20 predictor")
st.markdown(
"Enter **expected scores** for the un-played matches below. "
"Leave a cell blank to let the simulation draw that alliance’s score "
"from its own mean ± σ."
)

editable = st.data_editor(
    todo_df,
    hide_index=True,
    num_rows="dynamic",
    column_config={c: st.column_config.NumberColumn(step=1) for c in score_cols},
    key="todo_editor"
)
raw.update(editable)                     # merge user edits

# ═════════════ 3  Build score lists per team ═══════════════
scores = {}
for _, row in raw.iterrows():
    for color in ("Red", "Blue"):
        team = str(row[f"{color} Team 1"]).strip()
        if not team or team.lower() == "nan":
            continue
        val = row[f"{color} Score"]
        if pd.isna(val):
            continue
        scores.setdefault(team, []).append(float(val))

teams = sorted(scores)
n_teams = len(teams)

# ═════════════ 4  Pre-compute μ, σ, n_played ══════════════
mu, sigma, n_played = {}, {}, {}
for t in teams:
    arr = np.asarray(scores[t], float)
    n_played[t] = len(arr)
    mu[t] = arr.mean() if arr.size else 150.0
    sigma[t] = arr.std(ddof=1) if arr.size > 1 else DEFAULT_SIGMA

# ═════════════ 5  Monte-Carlo simulation ══════════════════
rng = np.random.default_rng(RNG_SEED)
all_preds = np.zeros((n_teams, REPS))
hit_top20 = np.zeros(n_teams, int)
cutline   = []

for k in range(REPS):
    avgs = np.empty(n_teams)
    for i, t in enumerate(teams):
        needed = TOTAL_RUNS - n_played[t]
        fut = rng.normal(mu[t], sigma[t], needed)
        all_scores = np.concatenate([scores[t], fut])
        all_scores.sort()                     # lowest first
        avgs[i] = all_scores[DROPS:].mean()   # drop bottom DROPS
    order = np.argsort(-avgs)
    hit_top20[order[:20]] += 1
    cutline.append(avgs[order[19]])
    all_preds[:, k] = avgs

mean_pred = all_preds.mean(axis=1)
ci_low    = np.percentile(all_preds,  2.5, axis=1)
ci_high   = np.percentile(all_preds, 97.5, axis=1)
p_top20   = hit_top20 / REPS

summary = (pd.DataFrame({
            "Team": teams,
            "Predicted Avg": mean_pred,
            "CI Low": ci_low,
            "CI High": ci_high,
            "P(Top 20)": p_top20})
           .sort_values("P(Top 20)", ascending=False)
           .reset_index(drop=True))

# ═════════════ 6  Display table ═══════════════════════════
st.subheader("Projected qualification table")
st.dataframe(
    summary.style.format({
        "Predicted Avg": "{:.1f}",
        "CI Low":        "{:.1f}",
        "CI High":       "{:.1f}",
        "P(Top 20)":     "{:.1%}"
    }),
    use_container_width=True,
    height=min(600, 28*len(summary)+25)
)

# ═════════════ 7  Visual summaries ════════════════════════
with st.expander("Show visual summaries"):
    # cut-line histogram
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.hist(cutline, bins=25, edgecolor="black")
    ax1.set_xlabel("Score of 20-th-place team")
    ax1.set_ylabel("Simulations")
    ax1.set_title("Cut-line distribution")
    st.pyplot(fig1)

    # top-30 probability bar chart
    top30 = summary.head(30).iloc[::-1]
    fig2, ax2 = plt.subplots(figsize=(9, 0.28*len(top30)+1.2))
    ax2.barh(top30["Team"], top30["P(Top 20)"])
    ax2.set_xlabel("Probability of finishing in top 20")
    ax2.set_title("Top-30 teams")
    st.pyplot(fig2)

# ═════════════ 8  Download CSV ════════════════════════════
st.download_button(
    "Download table as CSV",
    summary.to_csv(index=False).encode(),
    file_name="predicted_rankings_with_probabilities.csv",
    mime="text/csv"
)
