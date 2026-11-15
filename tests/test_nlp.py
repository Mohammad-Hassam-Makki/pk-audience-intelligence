"""Run: python -m pytest tests -q   (or python tests/test_nlp.py). Uses the example comments from the project brief."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nlp import tag_one, classify_post
CASES = [("Price please?", "price", "English"), ("Is this available in Karachi?", "availability", "English"), ("Very beautiful dress!", "praise", "English"),
         ("Price bohat zyada hai.", "price", "Roman Urdu"), ("Delivery kitne din mein hogi?", "delivery", "Roman Urdu"), ("Any discount on this suit?", "discount", "English"),
         ("Ordered last week, still waiting.", "delivery", "English"), ("Inbox me details send kar dein.", "purchase_intent", "Roman Urdu"), ("قیمت کیا ہے؟", "price", "Urdu")]
def test_brief_examples():
    for text, topic, lang in CASES:
        t, _, _, l = tag_one(text); assert (t, l) == (topic, lang), (text, t, l)
def test_negative_sentiment():
    assert tag_one("Ordered last week, still waiting.")[1] == "negative" and tag_one("Price bohat zyada hai.")[1] == "negative"
def test_post_classifier():
    assert classify_post("Flash SALE 50% off", 120)[0] == "Sale announcement" and classify_post("Eid Collection 2026", 30) == ("Short (reel-style)", "Eid Collection")
if __name__ == "__main__": test_brief_examples(); test_negative_sentiment(); test_post_classifier(); print("all passed")
