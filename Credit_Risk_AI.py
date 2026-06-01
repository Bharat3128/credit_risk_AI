print("Credit Risk AI Project")

import pandas as pd

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

from sklearn.tree import DecisionTreeClassifier
model = Descisiontreeclassifier()
model.fit(x, y)

prediction = model.predict([[60000, 680, 25000]])

print(Predication)
