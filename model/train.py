import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

print("Starting script...")  # Debug print

# Load dataset
print("Loading CSV...")  # Debug print
df = pd.read_csv("../data/Crop_recommendation.csv")
print(f"Data loaded: {df.shape}")  # Debug print

X = df.drop("label", axis=1)
y = df["label"]

# Stratified split
print("Splitting data...")  # Debug print
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
print("Training model...")  # Debug print
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Evaluating...")  # Debug print
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy: {acc}")
print(f"F1 Score: {f1}")

# Save model ONLY if metrics are good
if acc >= 0.90 and f1 >= 0.90:
    joblib.dump(model, "crop_model.pkl")
    print("Model saved successfully.")
else:
    print("Model did not meet required metrics.")