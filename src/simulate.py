"""Simulated demo dataset (Tables A and B from the brief). Everything here is GENERATED, not observed:
data_provenance = 'simulated', brands are anonymised (Brand A..H). Used only so the app runs without an API key.
Ground-truth columns (sim_topic, sim_language, sim_anomaly) exist only to test the pipeline.
"""
import numpy as np, pandas as pd, random, re
R = np.random.default_rng(11); random.seed(11)
BRANDS = {"Brand A": 900_000, "Brand B": 700_000, "Brand C": 600_000, "Brand D": 350_000, "Brand E": 250_000, "Brand F": 180_000, "Brand G": 120_000, "Brand H": 60_000}
CITIES = ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", "Multan", "Faisalabad", "Rawalpindi"]
# content type: (view mult, like rate, comment rate, topic mix)
CT = {"Short (reel-style)": (1.6, .040, .0012, dict(praise=.35, purchase_intent=.15, price=.12, availability=.1, discount=.08, fabric_quality=.05, size=.03, general=.12)),
      "Product showcase": (.8, .030, .0016, dict(price=.2, fabric_quality=.15, availability=.15, size=.1, purchase_intent=.12, discount=.08, praise=.12, general=.08)),
      "Styling video": (1.0, .036, .0014, dict(praise=.3, fabric_quality=.15, size=.15, price=.1, availability=.08, purchase_intent=.08, discount=.04, general=.1)),
      "Influencer collab": (1.4, .028, .0020, dict(purchase_intent=.2, availability=.15, praise=.2, price=.12, discount=.08, fabric_quality=.05, size=.05, general=.15)),
      "Sale announcement": (1.1, .022, .0030, dict(discount=.25, price=.2, availability=.15, purchase_intent=.12, delivery=.08, size=.04, praise=.04, general=.12)),
      "Campaign film": (2.0, .034, .0008, dict(praise=.5, general=.15, price=.08, availability=.07, purchase_intent=.07, fabric_quality=.05, discount=.04, size=.04))}
CAMP = ["Eid Collection", "11.11 Sale", "12.12 Sale", "Summer Lawn", "Winter Collection", "Wedding / Bridal", "General"]
T = {  # template pools: English / Roman Urdu / Urdu script
 "price": (["Price please?", "What is the price of this suit?", "How much for the 3 piece?", "Price??", "Too expensive for lawn, price is crazy", "what's the rate"],
           ["Price bohat zyada hai", "Is ka rate kya hai?", "Kitne ka hai ye suit?", "Price kitni hai?", "Bohat mehngi hai yaar", "qeemat bata dein"], ["قیمت کیا ہے؟", "یہ کتنے کا ہے", "بہت مہنگا ہے"]),
 "discount": (["Any discount on this suit?", "When is the next sale?", "Is there any offer on this?", "Please give discount code", "sale ends when?", "any promo code?"],
              ["Discount kab aayega?", "Sale mein kitna off hai?", "Thora sasta kar do please", "Koi offer hai is pe?", "kam karo price thori"], ["ڈسکاؤنٹ ہے؟", "سیل کب لگے گی"]),
 "delivery": (["Ordered last week, still waiting.", "Order not received yet, no tracking update", "How many days for delivery?", "Delivery is so late, worst service", "Is there courier to my city? how long?", "my parcel is stuck since 10 days"],
              ["Delivery kitne din mein hogi?", "Order kiya tha abhi tak nahi aya", "Parcel abhi tak nahi mila, tracking bhi nahi", "Courier kab tak aayega?", "10 din ho gaye order nahi aya"], ["ڈیلیوری کتنے دن میں ہو گی", "آرڈر ابھی تک نہیں آیا"]),
 "availability": (["Is this available in {city}?", "Is it in stock in size M?", "Which store has this in {city}?", "Will you restock this?", "available in {city} branch?", "out of stock again?"],
                  ["{city} mein milega ye?", "Ye kahan se milegi?", "Stock mein hai ye suit?", "{city} branch mein available hai?", "Kab restock hoga?"], ["کراچی میں دستیاب ہے؟", "یہ کہاں سے ملے گا"]),
 "size": (["What is the size chart?", "Is it available in XL?", "Length of the kameez please", "will medium fit chest 38?", "sizing runs small?"],
          ["Size chart bhej dein", "XL size mein hai?", "Kameez ki length kitni hai?", "Medium chota to nahi hoga?"], ["سائز کیا ہے", "ایکس ایل ہے؟"]),
 "fabric_quality": (["Is this pure lawn?", "Does the color fade after wash?", "Quality was great last time, same fabric?", "fabric looks thin, how is quality?", "cotton or lawn? shrink hota hai?"],
                    ["Kapra kaisa hai?", "Lawn quality kaisi hai, colour fade to nahi hota?", "Fabric patla lag raha hai", "Stitching quality kesi hai?"], ["کپڑا کیسا ہے", "کوالٹی کیسی ہے"]),
 "purchase_intent": (["Inbox me details", "I want to buy this, DM please", "Interested, how to order?", "Please send me WhatsApp number", "Want 2 pieces, how to place order"],
                     ["Inbox me details send kar dein", "Mujhe ye lena hai, whatsapp number dein", "Order kaise karna hai? inbox kar do", "Isay chahiye 2 pieces, dm karein"], ["ان باکس کریں", "مجھے چاہیے"]),
 "praise": (["Very beautiful dress!", "Absolutely stunning ❤️", "Love this collection", "Gorgeous colours masha Allah", "so elegant and pretty", "Amazing print!"],
            ["Bohat khoobsurat hai", "Mashallah bohat acha suit hai", "Kitni pyari design hai", "Bohat acha collection hai 😍"], ["بہت خوبصورت", "ماشاءاللہ زبردست"]),
 "general": (["First 🔥", "Nice video", "Which song is this?", "Who is the model?", "notification squad", "🔥🔥🔥"], ["Ye song konsa hai?", "Model kon hai?", "Video acha hai bhai"], ["واہ", "اچھا"]),
}
LANG_P = np.array([.40, .52, .08])
def make_comment(topic, post_ctx=None):
    lang = R.choice(3, p=LANG_P); tpl = random.choice(T[topic][lang]).replace("{city}", random.choice(CITIES))
    if lang != 2 and R.random() < .25: tpl = tpl.lower()
    if R.random() < .12: tpl = re.sub(r"([aeiou])", r"\1\1", tpl, count=1)       # pleaseee-style stretching
    if R.random() < .08: tpl += " " + random.choice(["🙏", "😍", "🔥", "!!", "??"])
    return tpl, ["English", "Roman Urdu", "Urdu"][lang]

def build(n_posts=3000, mean_comments=11, collected="2026-11-15"):
    posts, comments = [], []; ctypes = list(CT); anomaly_ids = R.choice(n_posts, 90, replace=False)
    kinds = {int(i): ["viral", "delivery_storm", "spam", "flop"][j % 4] for j, i in enumerate(anomaly_ids)}
    for i in range(n_posts):
        brand = random.choices(list(BRANDS), weights=list(BRANDS.values()))[0]; ct = random.choices(ctypes, weights=[.18, .26, .14, .14, .14, .14])[0]
        camp = random.choices(CAMP, weights=[.2, .1, .1, .22, .14, .1, .14])[0]; age = int(R.integers(3, 420))
        vm, lr, cr, mix = CT[ct]; views = BRANDS[brand] * .06 * vm * R.lognormal(0, .8)
        likes = views * lr * R.lognormal(0, .25); cc = max(0, int(views * cr * R.lognormal(0, .3))); kind = kinds.get(i)
        if kind == "viral": views *= 18; likes *= 14; cc = int(cc * 12)
        if kind == "flop": views *= .15; likes *= .3; cc = int(cc * .15)
        if kind == "spam": cc = int(cc * 9) + 300
        pid = f"P{i+1:04d}"; mix_p = dict(mix)
        if camp in ("11.11 Sale", "12.12 Sale"): mix_p["delivery"] = mix_p.get("delivery", 0) + .07
        if kind == "delivery_storm": mix_p = {"delivery": .7, "price": .05, "general": .25}
        topics, w = list(mix_p), np.array(list(mix_p.values())); w = w / w.sum()
        n = min(cc, max(0, int(R.poisson(mean_comments)))) if kind != "spam" else min(cc, 25)
        for j in range(n):
            if kind == "spam" and R.random() < .8:
                txt, lang, tp = random.choice(["Follow me for free followers", "Nice pic follow back", "Earn money from home click link"]) + " 🔥", "English", "general"
            else:
                tp = R.choice(topics, p=w); txt, lang = make_comment(tp)
            comments.append(dict(comment_id=f"C{len(comments)+1:06d}", post_id=pid, comment_text=txt, sim_topic=tp, sim_language=lang))
        posts.append(dict(post_id=pid, brand=brand, platform="YouTube", campaign=camp, content_type=ct, product_category="Lawn suit", likes=int(likes),
                          comments_count=cc, shares=np.nan, views=int(views), days_since_post=age, source_url="", collected_at=collected,
                          data_provenance="simulated", sim_anomaly=kind or ""))
    return pd.DataFrame(posts), pd.DataFrame(comments)

if __name__ == "__main__":
    p, c = build(); p.to_csv("data/posts.csv", index=False); c.to_csv("data/comments.csv", index=False)
    print(len(p), "posts", len(c), "comments;", (p.sim_anomaly != "").sum(), "planted anomalies")
