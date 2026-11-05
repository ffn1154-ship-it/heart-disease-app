"""Train the heart disease model and save model/heart_model.joblib.

Run once:  python train_model.py
Downloads heart.csv automatically if it is not already present.
"""
import os
import urllib.request

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

CSV_URL = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"

if not os.path.exists("heart.csv"):
    urllib.request.urlretrieve(CSV_URL, "heart.csv")

df = pd.read_csv("heart.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/heart_model.joblib")
print(f"Model saved. Test accuracy: {model.score(X_test, y_test):.3f}")
