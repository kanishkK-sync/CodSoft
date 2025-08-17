# 1. Import Necessary Libraries
# =============================
# We'll need pandas for data manipulation, matplotlib and seaborn for plotting,
# and scikit-learn for machine learning models and evaluation.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load and Explore the Data
# ============================
# We can load the Titanic dataset directly from Seaborn's library for convenience.
# If you have a train.csv file, you would use: df = pd.read_csv('train.csv')

print("--- Loading and Exploring Data ---")
df = sns.load_dataset('titanic')

# Get a first look at the data
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
df.info()

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# 3. Data Cleaning and Feature Engineering
# ========================================
# Before we can train a model, we need to handle missing data and convert
# non-numeric columns into a format the model can understand.

print("\n--- Cleaning Data and Engineering Features ---")

# --- Handle Missing Values ---
# Strategy 1: Fill 'age' with the median age. The median is less sensitive to outliers than the mean.
df['age'] = df['age'].fillna(df['age'].median())

# Strategy 2: Fill 'embarked' and 'embark_town' with the mode (most common value).
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

# Strategy 3: The 'deck' column has too many missing values (688 out of 891). It's best to drop it.
df.drop('deck', axis=1, inplace=True)

# Verify that there are no more missing values in the columns we're using
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# --- Convert Categorical Features to Numeric ---
# Machine learning models require all input features to be numeric.
# We'll use one-hot encoding for 'sex' and 'embarked' columns.
# This creates new columns for each category (e.g., 'sex_male', 'sex_female').

df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)
# drop_first=True avoids multicollinearity by dropping one of the new columns (e.g., 'sex_female').

# --- Drop Unnecessary Columns ---
# 'who', 'adult_male', 'alive', 'class', 'embark_town' are redundant or not useful for prediction.
# 'pclass' is the numerical version of 'class'. 'survived' is the numerical version of 'alive'.
df.drop(['who', 'adult_male', 'alive', 'class', 'embark_town'], axis=1, inplace=True)

# The 'name' and 'ticket' columns are not in the seaborn dataset, so we don't need to drop them.
# If you were using the Kaggle train.csv, you would need to drop 'name', 'ticket', and 'passengerId'.

print("\nFirst 5 rows of the processed data ready for modeling:")
print(df.head())


# 4. Prepare Data for Modeling
# ============================
# We need to separate our features (X) from our target variable (y).

print("\n--- Preparing Data for Modeling ---")
X = df.drop('survived', axis=1) # All columns except 'survived'
y = df['survived']               # Only the 'survived' column

# Split the data into training and testing sets.
# 80% for training, 20% for testing. random_state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} passengers")
print(f"Testing set size: {X_test.shape[0]} passengers")


# 5. Train the Classification Model
# =================================
# We'll start with Logistic Regression, a simple and effective model for binary classification.

print("\n--- Training the Logistic Regression Model ---")
model = LogisticRegression(max_iter=2000) # max_iter increased to ensure convergence
model.fit(X_train, y_train)
print("Model training complete.")


# 6. Evaluate the Model
# =====================
# Let's see how well our model performs on the unseen test data.

print("\n--- Evaluating Model Performance ---")
y_pred = model.predict(X_test)

# --- Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# --- Confusion Matrix ---
# Shows True Positives, False Positives, True Negatives, False Negatives.
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# --- Classification Report ---
# Provides precision, recall, and f1-score for each class.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
