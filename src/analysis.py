"""Analysis engine: content performance, comment clustering, anomaly detection and recommendations."""
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize as l2norm
from nlp import TOPICS, tag_comments

FRIENDLY = {"price": "Price-sensitive shoppers", "discount": "Discount seekers", "delivery": "Delivery-concerned buyers",
            "availability": "Availability / city askers", "size": "Sizing askers", "fabric_quality": "Fabric & quality askers",
            "purchase_intent": "Ready to buy (inbox / DM)", "praise": "Admirers", "general": "General chatter"}

# ---------------- loading / enrichment ----------------
def prepare_posts(posts: pd.DataFrame) -> pd.DataFrame:
    p = posts.copy()
    for c in ["likes", "comments_count", "shares", "views", "days_since_post"]: p[c] = pd.to_numeric(p.get(c), errors="coerce")
    p["engagements"] = p["likes"].fillna(0) + p["comments_count"].fillna(0) + p["shares"].fillna(0)
    p["engagement_rate"] = p["engagements"] / p["views"].where(p["views"] > 0)
    p["like_rate"] = p["likes"] / p["views"].where(p["views"] > 0)
    p["comment_rate"] = p["comments_count"] / p["views"].where(p["views"] > 0)
    return p

def post_comment_profile(comments_tagged: pd.DataFrame) -> pd.DataFrame:
    g = comments_tagged.groupby("post_id")
    out = pd.DataFrame({"n_sampled": g.size(), "neg_share": g["sentiment"].apply(lambda s: (s == "negative").mean()),
                        "dup_share": g["text_norm"].apply(lambda s: 1 - s.nunique() / len(s) if len(s) else 0)})
    for t in TOPICS: out[f"{t}_share"] = g["topic"].apply(lambda s, t=t: (s == t).mean())
    return out.reset_index()

# ---------------- Q1: content performance ----------------
def content_performance(p: pd.DataFrame, by: str, min_age_days: int = 0) -> pd.DataFrame:
    d = p[(p["days_since_post"].fillna(9999) >= min_age_days) & p["engagement_rate"].notna()]
    g = d.groupby(by)
    t = g.agg(posts=("post_id", "size"), median_views=("views", "median"), median_engagement_rate=("engagement_rate", "median"),
              median_like_rate=("like_rate", "median"), median_comment_rate=("comment_rate", "median"),
              p25=("engagement_rate", lambda s: s.quantile(.25)), p75=("engagement_rate", lambda s: s.quantile(.75)))
    t["low_sample"] = t["posts"] < 10
    return t.sort_values("median_engagement_rate", ascending=False).reset_index()

# ---------------- Q2/Q4: clustering ----------------
def cluster_comments(c: pd.DataFrame, k: int = None, kmin: int = 8, kmax: int = 14, seed: int = 0, max_fit: int = 60000):
    """TF-IDF (char + word n-grams, robust to Roman-Urdu spelling) -> SVD -> KMeans. Returns (comments_with_cluster, summary, info)."""
    d = c[c["text_norm"].str.len() > 1].copy()
    fit = d.sample(min(len(d), max_fit), random_state=seed) if len(d) > max_fit else d
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=40000, sublinear_tf=True)
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    Xf = hstack([cv.fit_transform(fit["text_norm"]), wv.fit_transform(fit["text_norm"])]).tocsr()
    svd = TruncatedSVD(n_components=min(60, Xf.shape[1] - 1), random_state=seed); Z = l2norm(svd.fit_transform(Xf))
    scores = {}
    if k is None:
        samp = np.random.RandomState(seed).choice(len(Z), min(len(Z), 5000), replace=False)
        for kk in range(kmin, kmax + 1):
            lab = KMeans(kk, n_init=4, random_state=seed).fit_predict(Z)
            scores[kk] = float(silhouette_score(Z[samp], lab[samp]))
        k = max(scores, key=scores.get)
    km = KMeans(k, n_init=8, random_state=seed).fit(Z)
    Xall = hstack([cv.transform(d["text_norm"]), wv.transform(d["text_norm"])]).tocsr(); Zall = l2norm(svd.transform(Xall))
    d["cluster"] = km.predict(Zall); dist = np.linalg.norm(Zall - km.cluster_centers_[d["cluster"].values], axis=1); d["_dist"] = dist
    # top terms: word-level lift versus the whole corpus
    W = wv.transform(d["text_norm"]); terms = np.array(wv.get_feature_names_out()); glob = np.asarray(W.mean(0)).ravel(); rows = []
    for cl in sorted(d["cluster"].unique()):
        m = (d["cluster"] == cl).values; sub = d[m]; lift = np.asarray(W[m].mean(0)).ravel() - glob
        top = [t for t in terms[np.argsort(-lift)[:6]]]
        vc = sub["topic"].value_counts(); dom = vc.index[0]
        rows.append(dict(cluster=cl, size=len(sub), share=len(sub) / len(d), dominant_topic=dom, purity=vc.iloc[0] / len(sub),
                         neg_share=(sub["sentiment"] == "negative").mean(), top_terms=", ".join(top),
                         languages=", ".join(f"{k_} {v:.0%}" for k_, v in sub["language_type"].value_counts(normalize=True).head(3).items()),
                         examples=" | ".join(sub.nsmallest(3, "_dist")["comment_text"].str.slice(0, 70))))
    s = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    names, seen = [], {}
    for _, r in s.iterrows():
        lang = r.languages.split(",")[0]; n = (f"{FRIENDLY[r.dominant_topic]} · " if r.purity >= .4 else "Mixed topics · ") + (lang.rsplit(" ", 1)[0] if float(lang.rsplit(" ", 1)[1].strip("%")) >= 60 else "mixed language")
        seen[n] = seen.get(n, 0) + 1; names.append(n if seen[n] == 1 else f"{n} ({r.top_terms.split(',')[0]})")
    s.insert(1, "segment", names); d = d.merge(s[["cluster", "segment"]], on="cluster").drop(columns="_dist")
    from sklearn.metrics import adjusted_rand_score
    known = d["topic"] != "general"
    info = dict(k=k, silhouette_by_k=scores, ari_vs_lexicon=float(adjusted_rand_score(d.loc[known, "topic"], d.loc[known, "cluster"])),
                mean_purity=float(np.average(s.purity, weights=s["size"])))
    return d, s, info

# ---------------- Q3: anomalies ----------------
SHARE_FEATS = {"neg_share", "delivery_share", "price_share", "dup_share"}
FEATS = {"log_views": "relative views", "like_rate": "like rate", "comment_rate": "comment rate", "neg_share": "negative comments",
         "delivery_share": "delivery comments", "price_share": "price comments", "dup_share": "duplicate/spam-like comments"}
def detect_anomalies(p: pd.DataFrame, prof: pd.DataFrame, contamination: float = 0.03, min_z: float = 3.0, seed: int = 0):
    d = p.merge(prof, on="post_id", how="left"); d["log_views"] = np.log1p(d["views"]) - np.log1p(d.groupby("brand")["views"].transform("median"))  # views relative to the brand's typical post
    d["peer_group"] = d["content_type"].astype(str); small = d["peer_group"].map(d["peer_group"].value_counts()) < 12
    d.loc[small, "peer_group"] = "All posts"
    Z = pd.DataFrame(index=d.index, columns=list(FEATS), dtype=float); med = {}
    n = d["n_sampled"].fillna(0).clip(lower=1)
    for f in FEATS:
        x = d[f].astype(float); g = d.groupby("peer_group")[f]; m = g.transform("median"); med[f] = m
        if f in SHARE_FEATS:   # proportions from a few sampled comments are noisy: use a binomial standard error plus a floor
            mm = m.clip(.02, .98); scale = np.sqrt(mm * (1 - mm) / n) + .05
            Z[f] = ((x - m) / scale).fillna(0)
            Z.loc[d["n_sampled"].fillna(0) < 5, f] = 0
        else:
            mad = g.transform(lambda s: (s - s.median()).abs().median()); scale = 1.4826 * mad + 1e-3 * (np.abs(m) + 1e-6)
            Z[f] = ((x - m) / scale).clip(-12, 12).fillna(0)
    iso = IsolationForest(n_estimators=200, random_state=seed, contamination="auto").fit(Z.values)
    d["iso_score"] = -iso.score_samples(Z.values); d["max_abs_z"] = Z.abs().max(axis=1)
    d["anomaly_score"] = 0.5 * d["iso_score"].rank(pct=True) + 0.5 * d["max_abs_z"].rank(pct=True)
    cut = d["anomaly_score"].quantile(1 - contamination)
    d["is_anomaly"] = (d["anomaly_score"] >= cut) & (d["max_abs_z"] >= min_z)
    def why(i):
        parts = []
        for f in Z.loc[i].abs().sort_values(ascending=False).index[:2]:
            if abs(Z.loc[i, f]) < min_z: continue
            v, m = d.at[i, f], med[f][i]
            if f.endswith("share"): parts.append(f"{FEATS[f]} {v:.0%} vs peer median {m:.0%}")
            elif f == "log_views": parts.append(f"views {np.exp(v - m):.1f}× the brand's typical post")
            else: parts.append(f"{FEATS[f]} {v/max(m,1e-9):.1f}× peer median")
        return "; ".join(parts)
    d["why_unusual"] = [why(i) if d.at[i, "is_anomaly"] else "" for i in d.index]
    return d

# ---------------- Q5: recommendations ----------------
def recommend(p, ctab, seg, anomalies, comments, min_age_days=0):
    R = []; n_c = len(comments); tshare = comments["topic"].value_counts(normalize=True)
    def add(area, action, evidence, priority): R.append(dict(area=area, priority=priority, action=action, evidence=evidence))
    ok = ctab[~ctab["low_sample"]] if "low_sample" in ctab else ctab
    if len(ok) >= 2:
        b, w = ok.iloc[0], ok.iloc[-1]; ratio = b.median_engagement_rate / max(w.median_engagement_rate, 1e-9)
        add("Content", f"Shift production toward '{b.iloc[0]}' and rework '{w.iloc[0]}'.",
            f"Median engagement rate {b.median_engagement_rate:.2%} for '{b.iloc[0]}' ({int(b.posts)} posts) vs {w.median_engagement_rate:.2%} for '{w.iloc[0]}' ({int(w.posts)} posts): {ratio:.1f}x gap.", "High" if ratio > 1.5 else "Medium")
    pr, di = tshare.get("price", 0), tshare.get("discount", 0)
    if pr + di > .15:
        add("Promotions", "Put price in the caption/first line and test a bundle or tiered offer (e.g. 2 suits) instead of blanket discounts; pin a 'sale ends' comment.",
            f"{pr:.0%} of comments ask or complain about price and {di:.0%} ask for discounts ({(pr+di)*n_c:,.0f} of {n_c:,} comments).", "High" if pr + di > .3 else "Medium")
    de = tshare.get("delivery", 0)
    dneg = comments.loc[comments.topic == "delivery", "sentiment"].eq("negative").mean() if de else 0
    if de > .05:
        worst = comments[comments.topic == "delivery"].merge(p[["post_id", "campaign"]], on="post_id").campaign.value_counts(normalize=True)
        add("Customer service", f"Publish realistic delivery times, add courier tracking messages and assign someone to reply to order-status comments, starting with the '{worst.index[0]}' posts.",
            f"{de:.0%} of comments are delivery questions ({dneg:.0%} of them negative); {worst.iloc[0]:.0%} come from '{worst.index[0]}' posts.", "High" if dneg > .4 else "Medium")
    av = tshare.get("availability", 0)
    if av > .05:
        add("Content", "State stock status and delivery cities in captions/pinned comment; add a store-locator link to the description.", f"{av:.0%} of comments ask where or whether it is available.", "Medium")
    pi = tshare.get("purchase_intent", 0)
    if pi > .04:
        add("Customer service", "Put the WhatsApp/order link in the first line of the description and reply to 'inbox/DM' comments within a few hours; these are warm leads.",
            f"{pi:.0%} of comments ask to be contacted or ready to buy ({pi*n_c:,.0f} comments).", "High")
    sz = tshare.get("size", 0) + tshare.get("fabric_quality", 0)
    if sz > .08:
        add("Content", "Add a size chart and a fabric close-up/wash-care shot to product videos.", f"{sz:.0%} of comments ask about size, fit, fabric or quality.", "Medium")
    big = seg.sort_values("neg_share", ascending=False).iloc[0]
    if big.neg_share > .25: add("Customer service", f"Review the '{big.segment}' segment first: it has the highest share of negative comments.", f"{big.neg_share:.0%} negative across {int(big['size']):,} comments (e.g. {big.examples.split(' | ')[0]}).", "Medium")
    a = anomalies[anomalies.is_anomaly].sort_values("anomaly_score", ascending=False)
    if len(a): add("Investigation", f"Review the {len(a)} flagged posts (top: {a.iloc[0].post_id}, {a.iloc[0].brand}) before copying or dropping their format.", a.iloc[0].why_unusual or "Unusual engagement profile.", "Medium")
    if len(ok) >= 2:
        add("Campaign testing", f"A/B test '{ok.iloc[0].iloc[0]}' against '{ok.iloc[1].iloc[0]}' for the next launch, keeping brand, posting time and product constant; judge on comment rate and DM requests, not likes alone.",
            "Observational data cannot prove causality; a controlled test is the next step.", "Medium")
    order = {"High": 0, "Medium": 1, "Low": 2}
    return pd.DataFrame(R).sort_values("priority", key=lambda s: s.map(order)).reset_index(drop=True)
