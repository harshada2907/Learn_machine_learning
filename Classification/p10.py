#Social Network adds

#import libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

#load the data
data = pd.read_csv("snasep2023.csv")
print(data)

#check and handle null data
print(data.isnull().sum())

#features and target
features = data[["Gender", "Age", "EstimatedSalary"]]
target = data["Purchased"]

#check and handle categorical data
nfeatures = pd.get_dummies(features)

print(features)
print(nfeatures)

#train and test
x_train, x_test, y_train, y_test = train_test_split(nfeatures, target)

#model building
model = GaussianNB()
model.fit(x_train, y_train)

#classification report
cr = classification_report(y_test, model.predict(x_test))
print(cr)

#prediction
age = float(input("Enter youe age : "))
es = float(input("Enter the salary : "))
gen = int(input("Enter 1 for Female and 2 for Male : "))
if gen == 1:
	d = [[age, es, 1, 0]]
else:
	d = [[age, es, 0, 1]]

pur = model.predict(d)

print("Purchased= ", pur[0])