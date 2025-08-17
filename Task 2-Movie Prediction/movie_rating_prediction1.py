# movie_rating_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ✅ Load dataset
file_path = r"D:\Internship\Task 2-Movie Prediction\58038e20-17b1-45b1-99a1-9e01b8f76fb5.csv"
df = pd.read_csv(file_path)

print("✅ Dataset loaded successfully!")
print("Shape of dataset:", df.shape)
print("Columns available:", df.columns)

# ✅ Handle missing values
df = df.dropna()   # remove rows with missing values
print("Shape after dropping NaN:", df.shape)

# ✅ Select features and target
# (Change column names if they are different in your dataset)
features = ["Genre", "Director", "Actors"]
target = "Rating"

# Encode categorical features
encoder = LabelEncoder()
for col in features:
    df[col] = encoder.fit_transform(df[col].astype(str))

X = df[features]
y = df[target]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Train model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Predictions
y_pred = model.predict(X_test)

# ✅ Evaluation
print("\n📊 Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ✅ Example prediction
sample = X_test.iloc[0].values.reshape(1, -1)
predicted_rating = model.predict(sample)[0]
print("\n🎬 Example Prediction:")
print("Movie features:", X_test.iloc[0].to_dict())
print("Predicted Rating:", round(predicted_rating, 2))
