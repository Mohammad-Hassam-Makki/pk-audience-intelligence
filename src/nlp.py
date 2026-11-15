"""Rule-based comment tagging for English / Roman Urdu / Urdu-script comments.

These are *weak labels*: a transparent keyword lexicon (topic, language, sentiment, purchase intent). The clustering
step then discovers audience groups from the text itself, and compares them to these labels.
"""
import re
import numpy as np
import pandas as pd

TOPICS = ["price", "discount", "delivery", "availability", "size", "fabric_quality", "purchase_intent", "praise"]
LEX = {
    "price": r"\bprice|\brate\b|\bcost|kitne ka|kitni|qeemat|kimat|\bmehng|mahng|expensive|bohat zyada|too much|\bpkr\b|\brs\.?\s?\d|قیمت|کتنے کا|مہنگ",
    "discount": r"discount|\bsale\b|\boff\b|\boffer|promo|coupon|\bdeal\b|sasta|sasti|kam kar|\bcheap|clearance|voucher|ڈسکاؤنٹ|سیل|سستا",
    "delivery": r"deliver|courier|\border(ed)?\b.*\b(wait|week|days|late|not|nahi)|still waiting|tracking|parcel|dispatch|shipp|not received|nahi (aya|ayi|mila|mili)|abhi tak|kitne din|\bleopard|\btcs\b|ڈیلیوری|آرڈر|پارسل",
    "availability": r"availab|in stock|out of stock|\bstock\b|restock|\bstore\b|\bbranch|outlet|kahan|kahaan|milega|milegi|mil jaye|\bkarachi|\blahore|\bislamabad|\bpeshawar|\bquetta|\bmultan|\bfaisalabad|دستیاب|کہاں|ملے گا|ملے گی",
    "size": r"\bsize|sizing|\blength|\bxl\b|\bxxl\b|\bmedium|\bsmall\b|\blarge\b|\bchest|\bfit\b|\bfitting|chota|chhota|bada|\bshalwar|سائز",
    "fabric_quality": r"fabric|\blawn\b|cotton|quality|kapra|kapray|kaprey|material|colou?r fade|\bfade|\bshrink|khaddar|silk|stitching|print quality|\bdupatta|کپڑا|کوالٹی",
    "purchase_intent": r"\binbox|\bdm\b|\bdms\b|details|want to buy|i want|i need|interested|order kar|lena hai|lena chahta|chahiye|chahie|whatsapp|\bwtsp|contact|\bnumber\b|\bpm\b|send me|bhej|ان باکس|چاہیے",
    "praise": r"beautiful|gorgeous|stunning|lovely|love (it|this|the)|\bnice\b|pretty|elegant|masha ?allah|khoobsurat|khubsurat|pyari|pyara|bohat acha|bohot acha|very good|amazing|awesome|خوبصورت|ماشاءاللہ|زبردست",
}
NEG = r"still waiting|not received|nahi (aya|ayi|mila|mili)|worst|\bbad\b|poor|fraud|scam|late|delay|disappoint|waste|bohat zyada|too (much|expensive)|mehng|mahng|fade|cheated|no response|horrible|مہنگ|دھوکہ"
POS = LEX["praise"] + r"|thank|shukriya|jazak|great|best|good quality|perfect|happy"
INTENT_HIGH = r"\binbox|\bdm\b|\bdms\b|want to buy|i want|lena hai|order kar|chahiye|chahie|whatsapp|\bwtsp|contact|\bnumber\b|send me|bhej"
ROMAN_MARKERS = set("""hai hain kya nahi nahin bohat bahut bohot kitne kitna kitni karo kar dein chahiye acha achi abhi tak mujhe aap
apka apki apke wali wala kab kahan milega milegi shukriya jazakallah mashallah bhai behan lena dena kaisay kese kia zyada
bhej dain karein krdo krdein hoga hogi tha thi mein se""".split())
_URDU = re.compile(r"[؀-ۿ]")
_COMP = {k: re.compile(v, re.I) for k, v in LEX.items()}
_NEG, _POS, _INT = re.compile(NEG, re.I), re.compile(POS, re.I), re.compile(INTENT_HIGH, re.I)

def normalize(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"https?://\S+|@\w+", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)          # pleaseee -> pleasee
    t = re.sub(r"[^\w\s؀-ۿ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def language_of(t: str) -> str:
    if len(_URDU.findall(t)) >= max(2, 0.25 * len(t)): return "Urdu"
    toks = normalize(t).split()
    if sum(w in ROMAN_MARKERS for w in toks) >= 1 and len(toks) > 0: return "Roman Urdu"
    return "English"

def tag_one(t: str):
    n = normalize(t)
    scores = {k: len(c.findall(n)) for k, c in _COMP.items()}
    best = max(TOPICS, key=lambda k: (scores[k], -TOPICS.index(k)))   # ties -> earlier topic
    topic = best if scores[best] > 0 else "general"
    neg, pos = bool(_NEG.search(n)), bool(_POS.search(n))
    sentiment = "negative" if neg else ("positive" if pos else "neutral")
    if _INT.search(n): intent = "High"
    elif scores["price"] or scores["availability"] or scores["size"] or scores["discount"]: intent = "Possible inquiry"
    else: intent = "None"
    return topic, sentiment, intent, language_of(t)

def tag_comments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); res = [tag_one(t) for t in df["comment_text"].fillna("")]
    df["topic"], df["sentiment"], df["purchase_intent"], df["language_type"] = zip(*res) if res else ([], [], [], [])
    df["text_norm"] = df["comment_text"].fillna("").map(normalize)
    return df

# ---- post-level classification (used by the YouTube collector) ----
CAMPAIGNS = [("Eid Collection", r"\beid|ramzan|ramadan|chand raat"), ("11.11 Sale", r"11\.?11"), ("12.12 Sale", r"12\.?12"),
             ("Winter Collection", r"winter|khaddar|marina|velvet"), ("Summer Lawn", r"summer|lawn|spring|festive lawn"),
             ("Wedding / Bridal", r"bridal|wedding|shaadi|luxury formal"), ("Sale", r"\bsale\b|% ?off|clearance|discount|flash")]
CONTENT = [("Sale announcement", r"\bsale\b|% ?off|discount|clearance|flash|offer|11\.?11|12\.?12"),
           ("Influencer collab", r"ambassador|collab|feat\.?|ft\.?|with [A-Z]\w+ [A-Z]\w+|influencer|vlog|haul|review|unboxing"),
           ("Styling video", r"styling|style|lookbook|how to wear|ootd|try[- ]?on|lookbook"),
           ("Campaign film", r"campaign|film|official video|tvc|teaser|ad film|commercial"),
           ("Product showcase", r"unstitched|stitched|embroidered|printed|collection|new arrival|lawn|suit|pret|ready to wear|3[- ]?piece")]
def classify_post(title: str, duration_s: float = None):
    t = str(title or ""); campaign = next((n for n, p in CAMPAIGNS if re.search(p, t, re.I)), "General")
    if duration_s is not None and duration_s == duration_s and duration_s <= 60: return "Short (reel-style)", campaign
    ctype = next((n for n, p in CONTENT if re.search(p, t, re.I)), "Other")
    return ctype, campaign
