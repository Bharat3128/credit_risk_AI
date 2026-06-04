
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("========== Credit Risk AI Project ==========")

# ----------------------------------
# Dataset
# ----------------------------------

data = {
    "income": [50000, 60000, 55000, 80000, 75000, 45000, 70000, 90000, 120000, 150000],

    "credit_score": [550, 400, 450, 500, 600, 700, 650, 720, 760, 780],

    "loan_amount": [20000, 25000, 22000, 30000, 28000, 18000, 26000, 35000, 50000, 60000],

    "defaulted": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
}

# Create DataFrame

df = pd.DataFrame(data)

print("\nTraining Dataset:\n")

print(df)

# ----------------------------------
# Features and Target
# ----------------------------------

X = df[["income", "credit_score", "loan_amount"]]

y = df["defaulted"]

# ----------------------------------
# Split Dataset
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------
# Train Model
# ----------------------------------

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# ----------------------------------
# Model Evaluation
# ----------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# ----------------------------------
# User Input
# ----------------------------------

print("\n========== Loan Risk Assessment ==========")

income = float(input("Enter Annual Income: "))

credit_score = float(input("Enter Credit Score: "))

loan_amount = float(input("Enter Loan Amount: "))

# ----------------------------------
# Create Input Data
# ----------------------------------

new_customer = pd.DataFrame({
    "income": [income],
    "credit_score": [credit_score],
    "loan_amount": [loan_amount]
})

# ----------------------------------
# Prediction
# ----------------------------------

result = model.predict(new_customer)

# ----------------------------------
# Risk Probability
# ----------------------------------

probability = model.predict_proba(new_customer)

risk_percentage = probability[0][1] * 100

print(f"\nRisk Probability: {risk_percentage:.2f}%")

# ----------------------------------
# Risk Level
# ----------------------------------

if risk_percentage >= 80:
    risk_level = "HIGH"

elif risk_percentage >= 50:
    risk_level = "MEDIUM"

else:
    risk_level = "LOW"

print("Risk Level:", risk_level)

# ----------------------------------
# Business Rules
# ----------------------------------

print("\n========== Loan Decision ==========")

reasons = []

if credit_score < 650:
    reasons.append("Credit score below recommended level")

if loan_amount > income:
    reasons.append("Loan amount exceeds annual income")

# ----------------------------------
# Final Decision
# ----------------------------------

if result[0] == 1:

    print("\nLoan Status: REJECTED")

    print("Sorry for the inconvenience.")

    if reasons:

        print("\nReason(s):")

        for reason in reasons:
            print("-", reason)

else:

    print("\nLoan Status: APPROVED")

    print("Congratulations!")

    print("Your loan application has been approved.")

# ----------------------------------
# End
# ----------------------------------

print("\n========== Thank You ==========")
