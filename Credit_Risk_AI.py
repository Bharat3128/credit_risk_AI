
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import os

print("========== Credit Risk AI Project ==========")

# ----------------------------------
# Dataset
# ----------------------------------

df = pd.read_csv("loan_data.csv")

print("\nTraining Dataset:\n")

print(df)
print("\nMissing Values Check:\n")
print(df.isnull().sum())

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

loan_years = float(input("Enter Loan Term (Years): "))


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


#----------------------------------------------------

# EMI Calculation

#---------------------------------------------------

annual_interest_rate = 10

monthly_interest_rate = annual_interest_rate / 12 / 100

months = loan_years * 12

emi = (
    loan_amount *
    monthly_interest_rate *
    (1 + monthly_interest_rate) ** months
) / (
    ((1 + monthly_interest_rate) ** months )- 1
)

print(f"\nEstimated Monthly EMI: {emi:.2f}")


#------------------------------------------------
# Affordability Check
#------------------------------------------------

monthly_income = income / 12

if emi > monthly_income * 0.5:
    affordability = "HIGH FINANCIAL BURDEN"

elif emi > monthly_income * 0.3:
    affordability = "MODERATE"

else:
    affordability = "AFFORDABLE"
print("Affordability Status:", affordability)
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

    # AI Recommendations
    recommendations = []
    
    if credit_score < 650:
        recommendations.append("Improve credit score above 650")
    
    if loan_amount > income:
        recommendations.append("Reduce loan amount below annual income")
    
    if risk_percentage >= 80:
        recommendations.append("High financial risk detected")
    
    recommendations.append("Maintain stable income and repayment history")
    
    if recommendations:
        print("\nAI Recommendations:")
        for rec in recommendations:
            print("-", rec)

else:

    print("\nLoan Status: APPROVED")

    print("Congratulations!")

    print("Your loan application has been approved.")

# ----------------------------------
# End
# ----------------------------------

#prediction logging
#----------------------------------

log_data = pd.DataFrame({
    "timestamp": [datetime.now()],
    "income": [income],
    "credit_score": [credit_score],
    "loan_amount": [loan_amount],
    "risk_percentage": [risk_percentage],
    "risk_level": [risk_level],
    "loan_status": [
        "REJECTED" if result[0] == 1 else "APPROVED"]
})

#save Predication History
log_data.to_csv(
    "prediction_history.csv", 
    mode="a", 
    header=not pd.io.common.file_exists("loan_prediction_log.csv"),
    index=False
    )
print("\nPrediction Saved successfully.")


print("\n========== Thank You ==========")
