import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("../data/Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = joblib.load("crop_model.pkl")
preds = model.predict(X)

print(classification_report(y, preds))
