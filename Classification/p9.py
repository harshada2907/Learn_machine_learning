#Email Classifier

#import libraries
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#load the dataset
data = pd.read_csv("edsep2023.csv")
print(data)

#check and handle null data
print(data.isnull().sum())

#features and target
features = data[["Dear", "Friend", "Lunch", "Money"]]
target = data["Result"]

print(features)
print(target)

#check and handle categorical data
nfeatures = pd.get_dummies(features)

print(features)
print(nfeatures)

#train and test
x_train, x_test, y_train, y_test = train_test_split(nfeatures, target)

#model building
model = BernoulliNB()
model.fit(x_train, y_train)

#classification report
cr = classification_report(y_test, model.predict(x_test))
print(cr)

#prediction
dear = int(input("For dear: 1 No 2 Yes : "))
if dear == 1:
	d = [1, 0]
else:
	d = [0, 1]

friend = int(input("For friend : 1 No 2 Yes : "))
if friend == 1:
	d = d + [1, 0]
else:
	d = d + [0, 1]

lunch = int(input("For lunch : 1 No 2 Yes : "))
if lunch == 1:
	d = d + [1, 0]
else:
	d = d + [0, 1]

money = int(input("For money : 1 No 2 Yes : "))
if money == 1:
	d = d + [0, 1]
else:
	d = d + [1, 0]

result = model.predict([d])
print(result)

#internal working
res = model.predict_proba([d])
info = res.ravel().tolist()
print(info)

normal = round(info[0] * 100, 2)
spam = round(info[1] * 100, 2)

print("Normal = ", normal)
print("Spam = ", spam)