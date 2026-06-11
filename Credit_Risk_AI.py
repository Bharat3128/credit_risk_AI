
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
import sys

# ----------------------------------
# LOGIN SYSTEM
# ----------------------------------

def login(max_attempts=3):
    """Simple login system with retry mechanism"""
    print("\n========== LOGIN SYSTEM ==========")
    
    valid_users = {"admin": "admin123", "user": "password"}
    
    for attempt in range(max_attempts):
        try:
            username = input("Enter Username: ").strip()
            password = input("Enter Password: ").strip()
            
            if not username or not password:
                print("⚠ Username and password cannot be empty.")
                print(f"Attempts remaining: {max_attempts - attempt - 1}\n")
                continue
            
            if username in valid_users and valid_users[username] == password:
                print(f"\n✓ Login Successful! Welcome, {username}!")
                return True
            else:
                print("\n✗ Invalid credentials. Please try again.")
                print(f"Attempts remaining: {max_attempts - attempt - 1}\n")
        
        except KeyboardInterrupt:
            print("\n\n✗ Login cancelled by user.")
            sys.exit(0)
    
    print("\n✗ Maximum login attempts exceeded. Access denied.")
    return False

# ----------------------------------
# ASK FOR LOGIN
# ----------------------------------

if not login():
    sys.exit(0)

print("\n========== Credit Risk AI Project ==========")

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

def get_valid_float(prompt, min_value=0):
    """Get and validate float input from user"""
    while True:
        try:
            value = float(input(prompt))
            if value < min_value:
                print(f"⚠ Error: Value must be at least {min_value}. Please try again.")
                continue
            return value
        except ValueError:
            print("⚠ Error: Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n✗ Input cancelled by user.")
            sys.exit(0)

print("\n========== Loan Risk Assessment ==========")

income = get_valid_float("Enter Annual Income (₹): ", min_value=1)
credit_score = get_valid_float("Enter Credit Score (300-850): ", min_value=1)
loan_amount = get_valid_float("Enter Loan Amount (₹): ", min_value=1)
loan_years = get_valid_float("Enter Loan Term (Years): ", min_value=0.5)


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

# Handle edge case where interest rate is 0
if monthly_interest_rate == 0:
    emi = loan_amount / months
else:
    emi = (
        loan_amount *
        monthly_interest_rate *
        (1 + monthly_interest_rate) ** months
    ) / (
        ((1 + monthly_interest_rate) ** months) - 1
    )

print(f"\nEstimated Monthly EMI: ₹{emi:,.2f}")


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

print(f"Affordability Status: {affordability}")
# ----------------------------------
# Business Rules
# ----------------------------------

print("\n========== Loan Decision ==========")

reasons = []

if credit_score < 650:
    reasons.append("Credit score below recommended level")

if loan_amount > income:
    reasons.append("Loan amount exceeds annual income")


#----------------------------------
# FRAUD DETECTION
#----------------------------------

# ----------------------------------
# Fraud Detection
# ----------------------------------

print("\n========== Fraud Analysis ==========")

fraud_warnings = []
fraud_score = 0

# Check 1: Loan to Income Ratio
if loan_amount > income * 3:
    fraud_warnings.append(
        "Loan amount is unusually high compared to income"
    )
    fraud_score += 40

# Check 2: Low Credit + Large Loan
if credit_score < 550 and loan_amount > income:
    fraud_warnings.append(
        "High risk loan request with low credit score and large loan amount"
    )
    fraud_score += 35

# Check 3: Very Low Credit Score
if credit_score < 400:
    fraud_warnings.append(
        "Extremely low credit score indicates high default risk"
    )
    fraud_score += 25

# Check 4: Extreme Loan Amount
if loan_amount > income * 5:
    fraud_warnings.append(
        "Loan amount exceeds 5x annual income - severely suspicious"
    )
    fraud_score += 30

# Determine Fraud Risk Level
if fraud_score >= 60:
    fraud_risk = "HIGH"
elif fraud_score >= 30:
    fraud_risk = "MEDIUM"
else:
    fraud_risk = "LOW"

# Display Fraud Analysis
print(f"\nFraud Risk Score: {fraud_risk}")
print(f"Risk Score Points: {fraud_score}/100")

if fraud_warnings:
    print("\nFraud Warning(s):")
    for warning in fraud_warnings:
        print(f"  ⚠ {warning}")
else:
    print("\n✓ No suspicious activity detected.")


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

# Prediction Logging
# ----------------------------------

try:
    log_data = pd.DataFrame({
        "timestamp": [datetime.now()],
        "income": [income],
        "credit_score": [credit_score],
        "loan_amount": [loan_amount],
        "risk_percentage": [risk_percentage],
        "risk_level": [risk_level],
        "fraud_risk": [fraud_risk],
        "loan_status": [
            "REJECTED" if result[0] == 1 else "APPROVED"]
    })

    # Save Prediction History
    log_data.to_csv(
        "prediction_history.csv",
        mode="a",
        header=not os.path.exists("prediction_history.csv"),
        index=False
    )
    print("\n✓ Prediction saved successfully.")

except Exception as e:
    print(f"\n✗ Error saving prediction: {str(e)}")

# ----------------------------------
# DASHBOARD
# ----------------------------------

print("\n" + "="*50)
print("   LOAN APPLICATION ASSESSMENT DASHBOARD")
print("="*50)

print("\n📊 APPLICANT INFORMATION:")
print(f"  Income: ${income:,.2f}")
print(f"  Credit Score: {credit_score}")
print(f"  Loan Amount: ${loan_amount:,.2f}")
print(f"  Loan Term: {loan_years} years")
print(f"  Monthly EMI: ${emi:,.2f}")

print("\n🔍 RISK ANALYSIS:")
print(f"  Model Prediction Risk: {risk_percentage:.2f}%")
print(f"  Risk Level: {risk_level}")
print(f"  Affordability Status: {affordability}")

print("\n⚠️  FRAUD ASSESSMENT:")
print(f"  Fraud Risk: {fraud_risk}")
print(f"  Fraud Score: {fraud_score}/100")

print("\n✅ FINAL DECISION:")
loan_status = "REJECTED" if result[0] == 1 else "APPROVED"
print(f"  Loan Status: {loan_status}")

print("\n" + "="*50)
print("========== Thank You ==========")
