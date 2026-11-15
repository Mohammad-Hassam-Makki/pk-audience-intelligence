"""Pakistan Social Audience & Campaign Intelligence Engine: Streamlit front end.  Run:  streamlit run app.py"""
import os, sys, io
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from nlp import tag_comments, TOPICS
import analysis as A

st.set_page_config(page_title="Audience & Campaign Intelligence", page_icon="📊", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.4rem;max-width:1250px}
div[data-testid="stMetricValue"]{font-size:1.5rem} .rec{border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:.8rem 1rem;margin-bottom:.6rem}
.rec .tag{font-size:.72rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;opacity:.7}</style>""", unsafe_allow_html=True)
PAL = px.colors.qualitative.Safe
LABEL = {"price": "Price", "discount": "Discount", "delivery": "Delivery", "availability": "Availability", "size": "Size", "fabric_quality": "Fabric / quality",
         "purchase_intent": "Purchase intent", "praise": "Praise", "general": "General"}

@st.cache_data(show_spinner="Reading and tagging comments…")
def load(posts_bytes, comments_bytes, mtime):
    p = pd.read_csv(io.BytesIO(posts_bytes)); c = pd.read_csv(io.BytesIO(comments_bytes))
    if "sim_anomaly" in p: p["sim_anomaly"] = p["sim_anomaly"].fillna("")
    return A.prepare_posts(p), tag_comments(c)

@st.cache_data(show_spinner="Clustering comments…")
def cluster(c, k):
    return A.cluster_comments(c, k=k)

# ---------------- sidebar: data + filters ----------------
st.sidebar.header("Data")
up_p = st.sidebar.file_uploader("posts.csv (Table A)", type="csv"); up_c = st.sidebar.file_uploader("comments.csv (Table B)", type="csv")
if up_p and up_c: pb, cb, mt = up_p.getvalue(), up_c.getvalue(), "upload"
else:
    here = os.path.dirname(__file__); fp, fc = f"{here}/data/posts.csv", f"{here}/data/comments.csv"
    if not (os.path.exists(fp) and os.path.exists(fc)): st.error("No data found. Run `python src/simulate.py` or `python src/collect_youtube.py`, or upload both CSVs."); st.stop()
    pb, cb, mt = open(fp, "rb").read(), open(fc, "rb").read(), os.path.getmtime(fp)
posts, comments = load(pb, cb, mt)
prov = posts.get("data_provenance", pd.Series(["unknown"] * len(posts))).fillna("unknown").unique().tolist()

st.sidebar.header("Filters")
brands = st.sidebar.multiselect("Brand", sorted(posts.brand.unique()), default=sorted(posts.brand.unique()))
camps = st.sidebar.multiselect("Campaign", sorted(posts.campaign.unique()), default=sorted(posts.campaign.unique()))
lawn = st.sidebar.checkbox("Lawn suits only", value="product_category" in posts and (posts.product_category == "Lawn suit").any())
min_age = st.sidebar.slider("Ignore posts newer than (days)", 0, 60, 7, help="Young posts are still collecting likes and comments, so they look worse than they are.")
k_opt = st.sidebar.selectbox("Number of comment clusters", ["auto"] + list(range(5, 21)), help="'auto' picks the k with the best silhouette score between 8 and 14.")
sens = st.sidebar.slider("Anomaly sensitivity (% of posts flagged at most)", 1, 10, 3)

pf = posts[posts.brand.isin(brands) & posts.campaign.isin(camps)]
if lawn: pf = pf[pf.product_category == "Lawn suit"]
cf = comments[comments.post_id.isin(pf.post_id)]
if len(pf) < 20 or len(cf) < 200: st.warning("Too little data after filtering. Widen the filters."); st.stop()

st.title("Pakistan Social Audience & Campaign Intelligence Engine")
if "simulated" in prov: st.info("**Demo data:** these posts and comments are *simulated* (brands anonymised). Run `src/collect_youtube.py` with your YouTube API key to replace them with real public data. See the *Data & method* tab.", icon="🧪")
m = st.columns(5); m[0].metric("Posts", f"{len(pf):,}"); m[1].metric("Comments analysed", f"{len(cf):,}"); m[2].metric("Brands", pf.brand.nunique())
m[3].metric("Median engagement rate", f"{pf.engagement_rate.median():.2%}"); m[4].metric("Data source", ", ".join(prov))

prof = A.post_comment_profile(cf); anom = A.detect_anomalies(pf, prof, contamination=sens / 100)
t1, t2, t3, t4, t5, t6 = st.tabs(["① Content", "② What they want", "③ Unusual posts", "④ Audience segments", "⑤ What to do next", "Data & method"])

# ---------- ① content ----------
with t1:
    st.subheader("Which content attracts attention?")
    by = st.radio("Compare by", ["content_type", "campaign", "brand"], horizontal=True, format_func=lambda s: s.replace("_", " ").title())
    ct = A.content_performance(pf, by, min_age)
    fig = go.Figure(go.Bar(x=ct[by], y=ct.median_engagement_rate, marker_color=[PAL[0] if not l else "#bbbbbb" for l in ct.low_sample],
        error_y=dict(type="data", symmetric=False, array=ct.p75 - ct.median_engagement_rate, arrayminus=ct.median_engagement_rate - ct.p25),
        text=[f"{v:.2%}" for v in ct.median_engagement_rate], textposition="outside", customdata=ct.posts, hovertemplate="%{x}<br>median %{y:.2%}<br>%{customdata} posts<extra></extra>"))
    fig.update_layout(template="plotly_white", yaxis_tickformat=".1%", yaxis_title="Median engagement rate (likes + comments + shares ÷ views)", height=380, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True); st.caption("Bars show the median; whiskers show the middle 50% of posts. Grey bars have fewer than 10 posts. Shares are not available from YouTube and are counted as 0.")
    c1, c2 = st.columns([3, 2])
    with c1:
        piv = pf[pf.days_since_post.fillna(999) >= min_age].pivot_table(index="content_type", columns="campaign", values="engagement_rate", aggfunc="median")
        hm = px.imshow(piv, aspect="auto", color_continuous_scale="Blues", text_auto=".1%", title="Median engagement rate: content type × campaign"); hm.update_layout(template="plotly_white", height=380, coloraxis_showscale=False)
        st.plotly_chart(hm, use_container_width=True)
    with c2:
        st.dataframe(ct.rename(columns={"median_views": "median views", "median_engagement_rate": "eng. rate", "median_like_rate": "like rate", "median_comment_rate": "comment rate"})
            [[by, "posts", "median views", "eng. rate", "like rate", "comment rate"]].style.format({"median views": "{:,.0f}", "eng. rate": "{:.2%}", "like rate": "{:.2%}", "comment rate": "{:.3%}"}), hide_index=True, use_container_width=True)

# ---------- ② audience wants ----------
with t2:
    st.subheader("What does the audience want?")
    st.caption("Each comment is tagged by a keyword lexicon covering English, Roman Urdu and Urdu script. Tags are transparent but approximate; segment discovery in tab ④ does not use them to cluster.")
    ts = cf.topic.value_counts(normalize=True).rename_axis("topic").reset_index(name="share"); ts["topic"] = ts.topic.map(LABEL)
    a, b = st.columns(2)
    with a:
        f = px.bar(ts.sort_values("share"), x="share", y="topic", orientation="h", text=ts.sort_values("share").share.map("{:.1%}".format), color_discrete_sequence=[PAL[0]])
        f.update_layout(template="plotly_white", height=360, xaxis_tickformat=".0%", title="Share of comments by topic", yaxis_title="", xaxis_title="", margin=dict(t=40)); st.plotly_chart(f, use_container_width=True)
    with b:
        mix = cf.merge(pf[["post_id", "content_type"]], on="post_id"); piv = pd.crosstab(mix.content_type, mix.topic.map(LABEL), normalize="index")
        h = px.imshow(piv, aspect="auto", color_continuous_scale="Blues", text_auto=".0%", title="Topic mix by content type"); h.update_layout(template="plotly_white", height=360, coloraxis_showscale=False, margin=dict(t=40)); st.plotly_chart(h, use_container_width=True)
    a, b = st.columns(2)
    with a:
        lg = cf.language_type.value_counts(normalize=True).rename_axis("language").reset_index(name="share")
        f = px.bar(lg, x="language", y="share", text=lg.share.map("{:.0%}".format), color_discrete_sequence=[PAL[2]]); f.update_layout(template="plotly_white", height=300, yaxis_tickformat=".0%", title="Comment language", margin=dict(t=40), xaxis_title="", yaxis_title=""); st.plotly_chart(f, use_container_width=True)
    with b:
        sd = pd.crosstab(cf.topic.map(LABEL), cf.sentiment, normalize="index").reset_index().melt("topic", var_name="sentiment", value_name="share")
        f = px.bar(sd, x="share", y="topic", color="sentiment", orientation="h", barmode="stack", color_discrete_map={"negative": "#d1495b", "neutral": "#c7c7c7", "positive": "#2a9d8f"}); f.update_layout(template="plotly_white", height=300, xaxis_tickformat=".0%", title="Sentiment within each topic", margin=dict(t=40), xaxis_title="", yaxis_title=""); st.plotly_chart(f, use_container_width=True)
    pick = st.selectbox("Read example comments for", [LABEL[t] for t in TOPICS + ["general"]]); inv = {v: k for k, v in LABEL.items()}
    st.dataframe(cf[cf.topic == inv[pick]].sample(min(12, (cf.topic == inv[pick]).sum()), random_state=1)[["comment_text", "language_type", "sentiment", "purchase_intent"]], hide_index=True, use_container_width=True)

# ---------- ③ anomalies ----------
with t3:
    st.subheader("Which marketing patterns are unusual?")
    st.caption("Each post is compared with posts of the same content type: relative views (vs the brand's typical post), like rate, comment rate, share of negative / delivery / price / duplicate comments. Flagged = highest combined robust-z and Isolation-Forest score.")
    an = anom.dropna(subset=["engagement_rate"]); f = px.scatter(an, x="views", y="engagement_rate", color=an.is_anomaly.map({True: "Flagged", False: "Normal"}), log_x=True, log_y=True, opacity=.7,
        color_discrete_map={"Flagged": "#d1495b", "Normal": "#b8c4d6"}, hover_data=["post_id", "brand", "content_type", "campaign"], category_orders={"color": ["Normal", "Flagged"]})
    f.update_layout(template="plotly_white", height=420, legend_title="", xaxis_title="Views (log)", yaxis_title="Engagement rate (log)", margin=dict(t=20)); st.plotly_chart(f, use_container_width=True)
    fl = anom[anom.is_anomaly].sort_values("anomaly_score", ascending=False)
    st.markdown(f"**{len(fl)} posts flagged for review**")
    cols = ["post_id", "brand", "content_type", "campaign", "views", "engagement_rate", "why_unusual"] + (["source_url"] if "source_url" in fl and fl.source_url.fillna("").str.startswith("http").any() else [])
    st.dataframe(fl[cols].style.format({"views": "{:,.0f}", "engagement_rate": "{:.2%}"}), hide_index=True, use_container_width=True,
                 column_config={"source_url": st.column_config.LinkColumn("link")} if "source_url" in cols else None)
    st.caption("A flag means 'different from comparable posts', not 'bad'. A viral post is worth copying; a delivery storm or spam wave needs a response.")

# ---------- ④ segments ----------
with t4:
    st.subheader("Which audience needs should the brand address?")
    k = None if k_opt == "auto" else int(k_opt); cd, seg, info = cluster(cf, k)
    st.caption(f"{info['k']} clusters found with TF-IDF character + word n-grams (robust to Roman-Urdu spelling) → SVD → KMeans. Mean cluster purity against the keyword tags: {info['mean_purity']:.0%}. Clusters group by wording *and* language, so one need can appear as several clusters.")
    need = cd.groupby("topic").size().rename("comments").reset_index(); need["share"] = need.comments / need.comments.sum()
    if True:
        f = px.bar(seg.sort_values("size"), x="size", y="segment", orientation="h", color="dominant_topic", color_discrete_sequence=PAL, labels={"size": "Comments", "segment": "", "dominant_topic": "Dominant topic"})
        f.update_layout(template="plotly_white", height=max(380, 30 * len(seg)), margin=dict(t=20, l=330), yaxis=dict(automargin=True), legend=dict(orientation="h", y=-0.12)); st.plotly_chart(f, use_container_width=True)
    if True:
        st.dataframe(seg[["segment", "share", "neg_share", "purity"]].rename(columns={"share": "share of comments", "neg_share": "negative", "purity": "purity"}).style.format({"share of comments": "{:.1%}", "negative": "{:.0%}", "purity": "{:.0%}"}), hide_index=True, use_container_width=True, height=min(520, 40 + 35 * len(seg)))
    if info["silhouette_by_k"]:
        s = pd.Series(info["silhouette_by_k"]); f = px.line(x=s.index, y=s.values, markers=True, labels={"x": "k", "y": "silhouette"}); f.update_layout(template="plotly_white", height=230, title="Choosing k (higher silhouette = tighter clusters)", margin=dict(t=40)); st.plotly_chart(f, use_container_width=True)
    sel = st.selectbox("Inspect a segment", seg.segment); r = seg[seg.segment == sel].iloc[0]
    c = st.columns(4); c[0].metric("Comments", f"{r['size']:,}"); c[1].metric("Negative", f"{r.neg_share:.0%}"); c[2].metric("Purity", f"{r.purity:.0%}"); c[3].metric("Languages", r.languages)
    st.markdown(f"**Characteristic terms:** {r.top_terms}"); st.markdown("**Most typical comments:**"); 
    for x in r.examples.split(" | "): st.markdown(f"- {x}")

# ---------- ⑤ recommendations ----------
with t5:
    st.subheader("What should the marketing team do next?")
    ctab = A.content_performance(pf, "content_type", min_age); _, seg5, _ = cluster(cf, None if k_opt == "auto" else int(k_opt))
    recs = A.recommend(pf, ctab, seg5, anom, cf, min_age); icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
    for _, r in recs.iterrows():
        st.markdown(f"<div class='rec'><div class='tag'>{icon[r.priority]} {r.priority} priority · {r.area}</div><b>{r.action}</b><br><span style='opacity:.8'>Evidence: {r.evidence}</span></div>", unsafe_allow_html=True)
    md = "# Marketing recommendations\n\n" + "\n".join(f"{i+1}. **[{r.priority} · {r.area}]** {r.action}\n   - Evidence: {r.evidence}" for i, r in recs.iterrows())
    d1, d2 = st.columns(2); d1.download_button("Download as Markdown", md, "recommendations.md"); d2.download_button("Download as CSV", recs.to_csv(index=False), "recommendations.csv")

# ---------- method ----------
with t6:
    st.subheader("Data & method")
    st.markdown(f"""**Provenance:** `{', '.join(prov)}`. Columns follow Tables A and B of the brief; `source_url`, `collected_at` and `data_provenance` are kept for reproducibility.

**Collecting real data (YouTube Data API v3).** Create an API key in Google Cloud Console, then run
`python src/collect_youtube.py --brands brands.txt --max-videos 100 --max-comments 100 --lawn-only`. It writes `data/posts.csv` and `data/comments.csv`, which this app reads automatically.
YouTube provides views, likes and comment counts but **no share counts**, and some channels hide likes or disable comments. Use the data within the YouTube API Terms and do not redistribute the comments.

**Pipeline.** (1) keyword lexicon tags topic, language, sentiment and purchase intent; (2) TF-IDF + SVD + KMeans discovers comment segments; (3) per-post features are compared with same-content-type peers using robust z-scores plus Isolation Forest; (4) rule-based recommendations cite the evidence that triggered them.

**Limitations.** Comments are a self-selected sample of the audience. Engagement is observational, so content-type differences are not proof of cause. The lexicon is hand-written and will miss slang. Brand names and numbers in the demo are fabricated.""")
    if "sim_anomaly" in posts:
        known = cf[cf.get("sim_topic", "general") != "general"] if "sim_topic" in cf else None
        st.markdown("**Validation on the simulated data (ground truth is known):**")
        v = anom[anom.is_anomaly]; planted = (anom.sim_anomaly != "")
        out = {"Planted unusual posts": int(planted.sum()), "Flagged posts": int(len(v)), "Planted posts found": f"{int((v.sim_anomaly != '').sum())} ({(v.sim_anomaly != '').sum() / max(planted.sum(), 1):.0%} recall)"}
        if known is not None: out["Keyword-tag accuracy"] = f"{(known.topic == known.sim_topic).mean():.0%}"; out["Language detection accuracy"] = f"{(cf.language_type == cf.sim_language).mean():.0%}"
        st.table(pd.Series({k: str(v) for k, v in out.items()}, name="value"))
