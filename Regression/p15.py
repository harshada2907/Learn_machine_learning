#Polynomial Regression

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("pssep2023.csv")
print(data)

feature = data[["Level"]]
target = data["Salary"]

pf = PolynomialFeatures(degree = 5)
nfeature = pf.fit_transform(feature)

model = LinearRegression()
model.fit(nfeature, target)

level = int(input("Enter the level : "))
nlevel = pf.fit_transform([[level]])

salary = model.predict(nlevel)
print("Salary = ", round(salary[0], 2))

plt.scatter(data["Level"], data["Salary"], color = "green")
plt.plot(data["Level"], model.predict(nfeature), color = "blue")
plt.show()