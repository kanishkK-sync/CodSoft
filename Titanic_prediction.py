# titanic_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load dataset from same folder as this script
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "titanic.csv")  # make sure titanic.csv is in Task 1 folder

df = pd.read_csv(file_path)

# Basic preprocessing
df = df.dropna(subset=["Age", "Fare", "Survived"])
X = df[["Age", "Fare", "Pclass"]]   # features
y = df["Survived"]                  # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
