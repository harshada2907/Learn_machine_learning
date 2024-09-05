#import the libraries
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import warnings
warnings.filterwarnings("ignore")

#load the data
data = pd.read_csv("pdsep2023.csv")
print(data)

#check the null data
print(data.isnull().sum())

#feature and target
feature = data[["Weather"]]
target = data["Play"]

#check and handle cat data
nfeature = pd.get_dummies(feature)
print(feature)
print(nfeature)


#model building
model = BernoulliNB()
model.fit(nfeature, target)

#prediction
we = int(input("1 Overcast 2 Rainy 3 Sunny : "))
if we == 1:
	d = [[1, 0, 0]]
elif we == 2:
	d = [[0, 1, 0]]
else:
	d = [[0, 0, 1]]

play = model.predict(d)
print(play)

#internal working
res = model.predict_proba(d)
info = res.ravel().tolist()
print(info)

pno = round(info[0] * 100, 2)
pyes = round(info[1] * 100, 2)
print("Probability of no = ", pno)
print("probability of yes = ", pyes)