print("Credit Risk AI Project")

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = {
    "income": [50000, 60000, 55000, 80000, 75000, 45000, 70000, 90000, 120000, 150000],
    "credit_score": [550, 400, 450, 500, 600, 700, 650, 720, 760, 780],
    "loan_amount": [20000, 25000, 22000, 30000, 28000, 18000, 26000, 35000, 50000, 60000],
    "defaulted": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
}
df = pd.DataFrame(data) 
print(df)

print("\nTraining Data Summary:")
print(df)

#features and Target variable
x = df[["income", "credit_score", "loan_amount"]]
y = df["defaulted"] 

#Split the data 

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y, 
    test_size=0.2, 
    random_state=42
    )
#Train the model

model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)
#evaluate the model

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("\nmodel Accuracy:", round(accuracy * 100,2), "%")

#user input for prediction
print("\n======== Loan Application Risk Assessment =======")
income = float(input("Enter your income: "))
credit_score = float(input("Enter your credit score: "))
loan_amount = float(input("Enter the loan amount: "))
user_data = [[income, credit_score, loan_amount]]
result = model.predict(user_data)

new_customer = pd.DataFrame({ 
"income": [income],
"credit_score": [credit_score],
"loan_amount": [loan_amount],
})

result = model.predict(new_customer)

#Risk probability
probability = model.predict_proba(new_customer)
risk_pertcentage = probability[0][1] * 100
print(f"\n Risk Probability: {risk_pertcentage:.2f}%")

risk_percentage = probability[0][1] * 100

print(f"\nRisk Probability: {risk_percentage:.2f}%")

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

print("\n===== Loan Decision =====")

reasons = []

if credit_score < 650:
    reasons.append("Credit score below recommended level")

if loan_amount > income:
    reasons.append("Loan amount exceeds annual income")

if result[0] == 1:

    print("Loan Status: REJECTED")
    print("Sorry for the inconvenience.")

    if reasons:
        print("\nReason(s):")

        for reason in reasons:
            print("-", reason)

else:

    print("Loan Status: APPROVED")
    print("Congratulations!")
    print("Your loan application has been approved.")

print("\n===== Thank You =====")
