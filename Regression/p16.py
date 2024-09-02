#import the libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#load the data
data = pd.read_csv("tpsep2023.csv")
print(data)

#check for null data/clean the data
print(data.isnull().sum())

#features and target
feature = data[["temp"]]
target = data["pressure"]

#apply polynomial features
pf = PolynomialFeatures(degree = 5)
nfeature = pf.fit_transform(feature)

#model building
model = LinearRegression()
model.fit(nfeature, target)

#plotting the graph
plt.scatter(data["temp"], data["pressure"], color = "red")
plt.plot(data["temp"], model.predict(nfeature), color = "green")
plt.show()

#input from the user
temp = float(input("Enter the temperature : "))
ntemp = pf.fit_transform([[temp]])

#prediction
pressure = model.predict(ntemp)
print("Pressure = ", pressure)