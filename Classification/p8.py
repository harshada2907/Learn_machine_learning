#import the libraries
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import warnings
warnings.filterwarnings("ignore")

#load the data
data = pd.read_csv("gssep2023.csv")
print(data)

#check and handle the null data
print(data.isnull().sum())

#features and target 
features = data[["Weather", "Car"]]
target = data["Result"]

#check and handle cat data
nfeatures = pd.get_dummies(features)

print(features)
print(nfeatures)

#model building
model = BernoulliNB()
model.fit(nfeatures, target)

#prediction
we = int(input("Enter 1 for Rainy and 2 for Sunny : "))
if we == 1:
	d = [1, 0]
else:
	d = [0, 1]

cc = int(input("Enter 1 for Broken and 2 for Working : "))
if cc == 1:
	d = d + [1, 0]
else:
	d = d + [0, 1]

ans = model.predict([d])
print(ans[0])

#internal working
res = model.predict_proba([d])
info = res.ravel().tolist()
g = round(info[0] * 100, 2)
s = round(info[1] * 100, 2)
print("go-out = ", g, "%")
print("Stay-in = ", s, "%")

