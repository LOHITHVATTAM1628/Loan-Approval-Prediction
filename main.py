# Loan Approval Prediction using Random Forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1️⃣ Load dataset
data = pd.read_csv("loan_data.csv")

# 2️⃣ Remove Loan_ID column (not useful for prediction)
if "Loan_ID" in data.columns:
    data = data.drop("Loan_ID", axis=1)

# 3️⃣ Handle missing values
data = data.dropna()

# 4️⃣ Convert text columns to numbers
encoder = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = encoder.fit_transform(data[column])

# 5️⃣ Separate features and target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# 6️⃣ Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# 8️⃣ Make predictions
predictions = model.predict(X_test)

# 9️⃣ Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
