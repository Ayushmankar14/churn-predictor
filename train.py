# train.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# STEP 1: Load the dataset
print("🔄 Loading data...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# STEP 2: Clean target column
print("🧹 Cleaning target column 'Churn'...")
df["Churn"] = df["Churn"].str.strip()
df = df[df["Churn"].isin(["Yes", "No"])]
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# STEP 3: Drop columns not useful or causing issues
print("🗑️ Dropping unused columns...")
df.drop(["customerID"], axis=1, inplace=True)

# STEP 4: Handle total charges (can have blanks or strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# STEP 5: Encode categorical features
print("🔤 Encoding categorical variables...")
df = pd.get_dummies(df, drop_first=True)

# STEP 6: Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# STEP 7: Train-test split
print("✂️ Splitting train/test data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 8: Scaling
print("📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 9: Model training
print("🚀 Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# STEP 10: Evaluation
print("📊 Model Evaluation:")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# STEP 11: Save model and scaler
print("💾 Saving model, scaler, and feature columns...")
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("churn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("churn_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("🎉 Training complete! Model, scaler, and feature list saved.")
