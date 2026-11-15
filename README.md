# Pakistan Social Audience & Campaign Intelligence Engine (BDA Project 3)

A Streamlit web app that answers the five marketing questions in the brief for Pakistani lawn-suit brands, using
**public YouTube data** (YouTube Data API v3) and **clustering** of comments.

| Question | Tab | Method |
|---|---|---|
| Which content attracts attention? | ① Content | Engagement rate by content type / campaign / brand, IQR whiskers, sample-size warnings |
| What does the audience want? | ② What they want | Keyword lexicon (English, Roman Urdu, Urdu script): topic, language, sentiment, purchase intent |
| Which patterns are unusual? | ③ Unusual posts | Peer-group robust z-scores + Isolation Forest, with a plain-language reason per post |
| Which audience needs should we address? | ④ Audience segments | TF-IDF (char + word n-grams) → SVD → KMeans; k chosen by silhouette |
| What should the team do next? | ⑤ What to do next | Rule-based recommendations, each citing its evidence; export as Markdown / CSV |

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py            # opens with the bundled SIMULATED demo data
```

## Use real YouTube data
1. Google Cloud Console → create a project → enable **YouTube Data API v3** → Credentials → **API key**.
2. Edit `brands.txt` (one brand per line), then:
```bash
export YOUTUBE_API_KEY=your_key          # PowerShell: $env:YOUTUBE_API_KEY="your_key"
python src/collect_youtube.py --brands brands.txt --max-videos 100 --max-comments 100 --lawn-only
streamlit run app.py
```
The script writes `data/posts.csv` and `data/comments.csv` (Tables A and B, plus `source_url`, `collected_at`, `data_provenance`).
Check the channel it prints for each brand is the official one. Quota is 10,000 units/day; a search costs 100 units, so channel lookups are cached in `data/channels.json`.
YouTube gives views, likes and comment counts but **no shares**; some channels hide likes or disable comments. Follow the YouTube API Terms and do not republish comments.

## Files
- `app.py` front end · `src/nlp.py` tagging · `src/analysis.py` clustering, anomalies, recommendations
- `src/collect_youtube.py` real data · `src/simulate.py` demo data (brands anonymised, `data_provenance=simulated`)
- `tests/test_nlp.py` checks the example comments from the brief (`python tests/test_nlp.py`)

## Honest limits
- The bundled data is simulated, so its findings only prove the pipeline works. On it, the detector recovers 64 of 90 planted unusual posts and the keyword tags are ~89% accurate against the generator's labels. Real comments will be noisier.
- Clusters split by wording *and* language, so one need can appear as several clusters (mean purity vs keyword tags is about 50%).
- Engagement comparisons are observational, not causal; use the A/B test recommendation to confirm.
