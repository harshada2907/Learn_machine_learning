import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("papsep2023.csv")
print(data)

print(data.isnull().sum())

features = data[["place", "area"]]
target = data["price"]

nfeatures = pd.get_dummies(features)

#pd.get_dummies() is a function of pandas that is used to convert the categorical values into numerical values
#since the ML model understands only numerical data so we need to convert the categorical values into numerical values

print(features)
print(nfeatures)

model = LinearRegression()
model.fit(nfeatures, target)

area = float(input("Enter the area : "))
op = int(input("1 for karjat 2 for khandala and 3 for Lonavala : "))

if op == 1:
	d = [[area, 1, 0, 0]]
elif op == 2:
	d = [[area, 0, 1, 0]]
else:
	d = [[area, 0, 0, 1]]

price = model.predict(d)
print("Price = ",round(price[0], 2))