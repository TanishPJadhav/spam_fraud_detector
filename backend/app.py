from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, os, re, json
import numpy as np
from blacklist import is_blacklisted_number, keyword_risk_score, add_number, get_blacklist

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

MODELS = {}
MODEL_NAMES  = ['rf_model', 'gb_model', 'nb_model', 'lr_model']
MODEL_LABELS = {
    'rf_model': 'Random Forest',
    'gb_model': 'Gradient Boosting',
    'nb_model': 'Naive Bayes',
    'lr_model': 'Logistic Regression',
}

for name in MODEL_NAMES:
    path = os.path.join(os.path.dirname(__file__), 'models', f'{name}.pkl')
    if os.path.exists(path):
        MODELS[name] = joblib.load(path)
        print(f"✅ Loaded {MODEL_LABELS[name]}")
    else:
        print(f"⚠️  {name} not found — run train_model.py first")

def extract_features(text):
    return {
        'len':         len(text),
        'caps_ratio':  round(sum(1 for c in text if c.isupper()) / max(len(text),1), 3),
        'digit_ratio': round(sum(1 for c in text if c.isdigit()) / max(len(text),1), 3),
        'exclamation': text.count('!'),
        'has_url':     bool(re.search(r'http|www|\.com|\.xyz', text, re.I)),
        'has_phone':   bool(re.search(r'\d{10}', text)),
        'has_money':   bool(re.search(r'rs\.?|rupee|lakh|₹|\$', text, re.I)),
        'has_otp':     bool(re.search(r'\botp\b|\bpin\b', text, re.I)),
        'urgency':     bool(re.search(r'urgent|immediately|now|hurry|expire|block|suspend', text, re.I)),
        'prize':       bool(re.search(r'win|won|prize|lottery|congrat|free|offer', text, re.I)),
    }

def analyse(text="", caller_number=""):
    has_msg = bool(text.strip())
    has_num = bool(caller_number.strip())

    # Must have at least one
    if not has_msg and not has_num:
        return {"error": "Provide at least a message or a caller number."}

    blacklisted  = is_blacklisted_number(caller_number) if has_num else False
    kw_score     = keyword_risk_score(text) if has_msg else 0.0
    features     = extract_features(text) if has_msg else {}
    model_results = {}
    votes_spam    = 0
    avg_spam_conf = 0.0
    mode          = "both" if has_msg and has_num else ("message" if has_msg else "number")

    if has_msg:
        for name, model in MODELS.items():
            try:
                prob      = model.predict_proba([text])[0]
                label     = int(model.predict([text])[0])
                spam_prob = float(prob[1])
                model_results[MODEL_LABELS[name]] = {
                    "label":      "Spam" if label == 1 else "Legit",
                    "confidence": round(spam_prob * 100, 1),
                }
                if label == 1:
                    votes_spam += 1
            except:
                model_results[MODEL_LABELS[name]] = {"label": "Error", "confidence": 0}

        avg_spam_conf = float(np.mean([v['confidence'] for v in model_results.values()])) / 100 if model_results else 0.0

    # Composite risk — weights adjust based on mode
    if mode == "both":
        composite = round((avg_spam_conf * 0.55) + (float(blacklisted) * 0.25) + (kw_score * 0.20), 3)
    elif mode == "message":
        composite = round((avg_spam_conf * 0.70) + (kw_score * 0.30), 3)
    else:  # number only
        composite = 1.0 if blacklisted else 0.0

    # Verdict
    if composite >= 0.5 or blacklisted:
        verdict   = "⚠️ SPAM / FRAUD DETECTED"
        alert_msg = "🚨 WARNING: Likely SPAM or FRAUD. Do NOT share OTP, bank details, or click any links!"
        alert     = True
    elif composite >= 0.3:
        verdict   = "⚠️ Suspicious"
        alert_msg = "Be cautious. This message shows suspicious patterns. Verify before acting."
        alert     = True
    else:
        verdict   = "✅ Safe"
        alert_msg = "This message/number appears to be legitimate."
        alert     = False

    # Number-only specific message
    if mode == "number" and not blacklisted:
        verdict   = "✅ Number Not Blacklisted"
        alert_msg = "This number is not in our blacklist. Stay cautious with unknown callers."

    return {
        "verdict":             verdict,
        "alert":               alert,
        "alert_message":       alert_msg,
        "mode":                mode,
        "risk_score":          composite,
        "ensemble_confidence": round(avg_spam_conf * 100, 1),
        "votes_spam":          votes_spam,
        "total_models":        len(MODELS),
        "blacklisted":         blacklisted,
        "keyword_score":       kw_score,
        "model_results":       model_results,
        "features":            features,
        "has_msg":             has_msg,
        "has_num":             has_num,
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyse", methods=["POST"])
def analyse_route():
    data   = request.get_json()
    text   = data.get("message", "").strip()
    number = data.get("caller_number", "").strip()
    if not text and not number:
        return jsonify({"error": "Provide at least a message or a caller number."}), 400
    return jsonify(analyse(text, number))

@app.route("/report", methods=["POST"])
def report():
    data   = request.get_json()
    number = data.get("number", "").strip()
    if not number:
        return jsonify({"error": "No number provided"}), 400
    add_number(number)
    return jsonify({"success": True, "message": f"{number} added to blacklist ✅"})

@app.route("/blacklist")
def blacklist():
    return jsonify(get_blacklist())

@app.route("/stats")
def stats():
    return jsonify({
        "models_loaded":   len(MODELS),
        "model_names":     list(MODEL_LABELS.values()),
        "blacklist_count": len(get_blacklist()["numbers"]),
        "keyword_count":   len(get_blacklist()["keywords"]),
    })

if __name__ == "__main__":
    print("🚀 Fraud Shield AI → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)