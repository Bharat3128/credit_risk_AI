print("Credit Risk AI Project")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    "income": [50000, 60000, 55000, 80000, 75000],
    "credit_score": [700, 650, 720, 680, 710],
    "loan_amount": [20000, 25000, 22000, 30000, 28000],
    "defaulted": [0, 1, 0, 1, 0]
}
df = pd.DataFrame(data) 
print(df)

x = df[["income", "credit_score", "loan_amount"]]
y = df["defaulted"] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

new_prediction = model.predict([[60000, 680, 25000]])

print("New Prediction:", new_prediction)


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x, y)

prediction = model.predict([[60000, 680, 25000]])

print("Prediction:", prediction)
