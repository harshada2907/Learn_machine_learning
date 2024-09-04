#fitting a line is called regression
#here the data is discrete hence Logistic Regression
#last were continuous data hence LinearRegression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("hrsep23.csv")
print(data)

print(data.isnull().sum())

feature = data[["hr"]]
target = data["result"]

x = data["hr"]
y = data["result"]

model = LogisticRegression()
model.fit(feature, target)

hr = float(input("Enter hours : "))
result = model.predict([[hr]])

print("Result = ", result[0])

plt.scatter(x, y, color = "black")
plt.xlabel("hour")
plt.ylabel("result")
plt.show()