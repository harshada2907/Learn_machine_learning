#visualizing the result

import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = pd.read_csv("hrsep23.csv")
print(data)

print(data.isnull().sum())

x = data["hr"]
y = data["result"]

feature = data[["hr"]]
target = data["result"]

model = LogisticRegression()
model.fit(feature, target)

plt.scatter(x, y, color = "black", label = "data points")
plt.plot(feature, model.predict(feature), color = "red", label = "sigmoid function")
plt.xlabel("Hours")
plt.ylabel("Result")
plt.show()