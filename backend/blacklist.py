import json, os

BLACKLIST_FILE = os.path.join(os.path.dirname(__file__), "data", "blacklist.json")

DEFAULT = {
    "numbers": ["9999999999","8888888888","7777777777","1234567890"],
    "keywords": ["OTP","lottery","click here","free recharge",
                 "account blocked","KYC expired","verify now",
                 "limited offer","win","prize","urgent","phishing",
                 "UPI PIN","deactivated","customs fee","release fee",
                 "government scheme","rupee","lakh","suspicious login"]
}

def _load():
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE) as f:
            return json.load(f)
    return DEFAULT.copy()

def _save(data):
    os.makedirs(os.path.dirname(BLACKLIST_FILE), exist_ok=True)
    with open(BLACKLIST_FILE, "w") as f:
        json.dump(data, f, indent=2)

def is_blacklisted_number(number: str) -> bool:
    return number.strip() in _load()["numbers"]

def keyword_risk_score(text: str) -> float:
    bl   = _load()
    hits = sum(1 for kw in bl["keywords"] if kw.lower() in text.lower())
    return round(min(hits / max(len(bl["keywords"]), 1), 1.0), 3)

def add_number(number: str):
    data = _load()
    if number not in data["numbers"]:
        data["numbers"].append(number)
        _save(data)

def get_blacklist():
    return _load()