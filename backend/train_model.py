import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

spam_samples = [
    "Congratulations! You won a lottery of Rs 10 lakh. Click here to claim.",
    "Your bank account has been blocked. Verify now at fake-bank.com",
    "Free Recharge! Send your OTP to get Rs 500 cashback.",
    "URGENT: Your KYC is expired. Update immediately or account will be closed.",
    "You have been selected for a government scheme. Call 9999999999.",
    "Win iPhone 15 for free. Limited time offer. Click the link.",
    "Your debit card is blocked. Call us now to unblock.",
    "Dear customer, your SBI account will be suspended. Verify OTP.",
    "Earn Rs 5000 daily from home. No investment needed. Join now.",
    "Congratulations! BSNL offers free 4G upgrade. Send OTP to confirm.",
    "Loan approved Rs 2 lakh instantly. No documents needed. Call now.",
    "Alert: Suspicious login detected. Verify via this link immediately.",
    "You have a pending refund. Enter UPI PIN to receive money.",
    "Job offer: Earn 50000/month. Work from home. Apply now.",
    "Phishing link detected click here http://scam-site.xyz",
    "Dear user, your account will be deactivated. Provide OTP to save.",
    "You are selected for PM scheme. Click to get Rs 15000 benefit.",
    "Tax refund of Rs 9800 pending. Submit bank details to claim.",
    "Your Aadhaar card is linked to fraud. Call officer immediately.",
    "Win a car! You are our lucky customer. Verify details now.",
    "Investment opportunity! Double your money in 7 days guaranteed.",
    "TRAI: Your mobile number will be blocked in 24 hours. Call us.",
    "You have a courier at customs. Pay Rs 1200 to release package.",
    "Your Paytm wallet is blocked. Complete KYC to continue using.",
    "WhatsApp: Verify your account or it will be permanently suspended.",
    "Police cybercell: Your IP address flagged for illegal activity.",
    "Amazon: Your order is held at customs. Pay release fee now.",
    "IRDAI: Your insurance policy has expired. Renew to avoid penalties.",
    "Electricity department: Pay pending bill or connection will be cut.",
    "Fake job offer: HR from MNC company offering 1 lakh salary.",
]

legit_samples = [
    "Hey, are you coming to the meeting at 3pm today?",
    "Please find the attached report for your review.",
    "Can we reschedule our call to tomorrow morning?",
    "Happy Birthday! Hope you have a wonderful day.",
    "The project deadline has been extended to next Friday.",
    "Dinner at 7pm? Let me know if that works for you.",
    "Your order has been shipped. Expected delivery in 2 days.",
    "Reminder: Doctor appointment tomorrow at 11am.",
    "Great work on the presentation! The client loved it.",
    "Please share the updated code once you are done.",
    "Mom called, she wants you to call her back.",
    "The electricity bill is due by the 30th.",
    "Team lunch is confirmed for Wednesday at 1pm.",
    "Your OTP is 482736. Do not share with anyone.",
    "Interview scheduled for Monday 10am. Best of luck!",
    "The meeting has been rescheduled to 4pm.",
    "Please review the attached documents before Thursday.",
    "Your flight is confirmed for 6am tomorrow. Check-in online.",
    "Library books are due for return this week.",
    "Congratulations on your promotion! Well deserved.",
    "Your subscription to Netflix has been renewed.",
    "Reminder: Staff meeting in conference room at 2pm.",
    "Your package has been delivered at the front door.",
    "Please submit your timesheet by end of day Friday.",
    "Your blood test report is ready. Please collect from lab.",
    "The annual leave has been approved for your request.",
    "Good morning! Have a productive day ahead.",
    "Quiz results are out. Check the college portal.",
    "The seminar on AI is scheduled for next Saturday.",
    "Congratulations on completing your project successfully.",
]

texts  = spam_samples + legit_samples
labels = [1]*len(spam_samples) + [0]*len(legit_samples)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42)

# Model 1: Random Forest
rf = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,3), max_features=8000, sublinear_tf=True)),
    ('clf',   RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)),
])
rf.fit(X_train, y_train)
print("RF  Accuracy:", round(rf.score(X_test, y_test)*100, 1), "%")

# Model 2: Gradient Boosting
gb = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)),
    ('clf',   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)),
])
gb.fit(X_train, y_train)
print("GB  Accuracy:", round(gb.score(X_test, y_test)*100, 1), "%")

# Model 3: Naive Bayes
nb = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ('clf',   MultinomialNB(alpha=0.1)),
])
nb.fit(X_train, y_train)
print("NB  Accuracy:", round(nb.score(X_test, y_test)*100, 1), "%")

# Model 4: Logistic Regression
lr = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,3), max_features=8000, sublinear_tf=True)),
    ('clf',   LogisticRegression(C=5, max_iter=1000, random_state=42)),
])
lr.fit(X_train, y_train)
print("LR  Accuracy:", round(lr.score(X_test, y_test)*100, 1), "%")

print("\n=== Best Model Report ===")
print(classification_report(y_test, rf.predict(X_test), target_names=["Legit","Spam"]))

os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(gb, "models/gb_model.pkl")
joblib.dump(nb, "models/nb_model.pkl")
joblib.dump(lr, "models/lr_model.pkl")
print("✅ All 4 models saved to backend/models/")