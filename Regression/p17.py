#import the libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")


#load the dataset
data = pd.read_csv("medical_cost.csv")
print(data)


#check for null data
print(data.isnull().sum())


#feature and target
features = data[["age", "bmi", "sex", "children", "smoker", "region"]]
target = data["charges"]


#handle categorical data
nfeatures = pd.get_dummies(features)

print(features)
print(nfeatures)


#Polynomial Features
pf = PolynomialFeatures(degree = 2)
nnfeatures = pf.fit_transform(nfeatures)


#train and test
x_train, x_test, y_train, y_test = train_test_split(nnfeatures, target, random_state = 42)


#model building
model = LinearRegression()
model.fit(x_train, y_train)


#score
s1 = model.score(x_train, y_train)
print("Training Score = ", s1)

s2 = model.score(x_test, y_test)
print("Testing Score = ", s2)


#Prediction
age = int(input("Enter your age : "))
d = [age]

bmi = float(input("Enter your bmi : "))
d = d + [bmi]

sex = int(input("Enter 1 for female and 2 for male : "))
if sex == 1:
	d = d + [1, 0]
else:
	d = d + [0, 1]

children = int(input("Enter number of childrens you have : "))
d = d + [children]

smoker = int(input("Enter 1 if you dont smoke and 2 if you smoke : "))
if smoker == 1:
	d = d + [1, 0]
else:
	d = d + [0, 1]

region = int(input("Enter 1 if the region is northeast, 2 if region is northwest, 3 if region is southeast and 4 if region is southwest :"))
if region == 1:
	d = d + [1, 0, 0, 0]
elif region == 2:
	d = d + [0, 1, 0, 0]
elif region == 3:
	d = d + [0, 0, 1, 0]
else:
	d = d + [0, 0, 0, 1]

nd = pf.fit_transform([d])
charges = model.predict(nd)
print()

print("Charges are : ", round(charges[0], 2))



