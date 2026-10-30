"""
Spam vs Legitimate Message Classifier (prototype)
Course: Software Design in Intelligent Systems
Author: Maftuna Sharabbayeva (ID 380261200262)

Pipeline: dataset -> preprocessing -> TF-IDF -> Multinomial Naive Bayes
          -> evaluation -> prediction function
"""
import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

random.seed(42)

# ---------------------------------------------------------------- 1. DATA
# Synthetic labelled dataset built from hand-written message templates.
SPAM_T = [
    "WINNER! You have won a {prize}. Click {link} to claim now",
    "Congratulations! You were selected for a free {prize}. Reply YES to claim",
    "URGENT: your bank account is suspended. Verify your password at {link}",
    "Limited offer!!! Buy cheap {item} now with {disc} discount. Click {link}",
    "You have been pre-approved for a loan of {money}. Apply today at {link}",
    "Earn {money} per week working from home. No experience needed. Visit {link}",
    "FREE {item}! Act now, offer expires today. Call 0900-555-777",
    "Your parcel is held. Pay a small fee at {link} to release it",
    "Claim your {money} cash prize today, send your card number to {link}",
    "Hot singles in your area want to meet you. Click {link} now",
    "Final notice: your account will be closed. Confirm your details at {link}",
    "Make {money} fast with this secret investment trick. Join now {link}",
]
HAM_T = [
    "Hi {name}, are we still meeting at {time} for the project discussion?",
    "Please find attached the lecture notes for {subj}. See you in class",
    "Reminder: the {subj} assignment is due on Friday. Let me know if you have questions",
    "Thanks {name}, I will send the report by {time} tomorrow",
    "Can you pick up some bread on your way home? Thanks",
    "Happy birthday {name}! Hope you have a wonderful day with your family",
    "The {subj} exam schedule has been updated. Check the faculty notice board",
    "Dinner at my place on Saturday? Mom is cooking plov",
    "Meeting moved to {time} in room 204. Please confirm your attendance",
    "I finished the code review, left a few comments on the pull request",
    "Don't forget to bring your laptop to the {subj} lab today",
    "Sorry I missed your call, I was in a lecture. Will call you back at {time}",
]
FILL = dict(
    prize=["iPhone 15", "$1000 gift card", "luxury holiday", "new laptop"],
    link=["http://bit.ly/x9Kd2", "www.win-now.biz", "http://secure-login.top/acc", "http://free-gift.xyz"],
    item=["watches", "pills", "sneakers", "phones"],
    disc=["70%", "90%", "50%"],
    money=["$5000", "$20000", "$300", "$1500"],
    name=["Ali", "Dilnoza", "Jasur", "Madina", "Sardor"],
    time=["10:30", "2 pm", "9 am", "5:15 pm", "noon"],
    subj=["Databases", "AI Systems", "Networks", "Algorithms", "Statistics"],
)


def fill(t):
    return re.sub(r"\{(\w+)\}", lambda m: random.choice(FILL[m.group(1)]), t)


def build_dataset(n_each=100):
    rows = [(fill(random.choice(SPAM_T)), "spam") for _ in range(n_each)]
    rows += [(fill(random.choice(HAM_T)), "legitimate") for _ in range(n_each)]
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["message", "label"])


# ------------------------------------------------------- 2. PREPROCESSING
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)   # normalise links
    text = re.sub(r"\$\d+|\d{3,}", " numtoken ", text)        # normalise money/numbers
    text = re.sub(r"[^a-z\s]", " ", text)                     # drop punctuation
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------- 3. MODEL
def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess,
                                  stop_words="english", ngram_range=(1, 2))),
        ("nb", MultinomialNB(alpha=0.5)),
    ])


def classify(model, message: str):
    proba = model.predict_proba([message])[0]
    classes = list(model.classes_)
    label = classes[proba.argmax()]
    return label, round(float(proba.max()), 3)


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("messages_dataset.csv", index=False)
    print("Dataset:", df.shape, df.label.value_counts().to_dict())

    # Evaluate on messages whose exact text was never seen in training
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.message, df.label, test_size=0.25, stratify=df.label, random_state=42)
    model = make_model().fit(X_tr, y_tr)
    pred = model.predict(X_te)

    print("\n--- Evaluation on hold-out set ---")
    print("Test size :", len(y_te))
    print("Accuracy  :", round(accuracy_score(y_te, pred), 3))
    print("Precision :", round(precision_score(y_te, pred, pos_label="spam"), 3))
    print("Recall    :", round(recall_score(y_te, pred, pos_label="spam"), 3))
    print("F1-score  :", round(f1_score(y_te, pred, pos_label="spam"), 3))
    print("Confusion matrix [rows=true, cols=pred] order [legitimate, spam]:")
    print(confusion_matrix(y_te, pred, labels=["legitimate", "spam"]))

    # Final model trained on everything for the demo
    final = make_model().fit(df.message, df.label)

    print("\n--- Test cases (new, hand-written messages) ---")
    tests = [
        "Congratulations! You won a free iPhone. Click http://claim-prize.top now",
        "Hi Madina, can we move the AI Systems lab to 3 pm tomorrow?",
        "URGENT: verify your bank password at www.secure-bank-login.biz or account will be closed",
        "Please send me the slides from today's Databases lecture, thanks",
        "Earn $3000 per week from home, no experience needed!!!",
        "Mom says dinner is ready, come home soon",
        "Your invoice is attached. Let me know if the payment details are correct",
        "Limited offer: 80% discount on luxury watches. Buy now!",
    ]
    for i, t in enumerate(tests, 1):
        label, conf = classify(final, t)
        print(f"{i}. [{label.upper():10s} p={conf}] {t}")
