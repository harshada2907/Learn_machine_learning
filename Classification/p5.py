#import the libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

#load the data
data = pd.read_csv("vdsep2023.csv")
print(data)

#check for null data
print(data.isnull().sum())

#feature and target
feature = data[["Age"]]
target = data["Vehicle"]

#train and test
x_train, x_test, y_train, y_test = train_test_split(feature, target)

#model building
model = LogisticRegression()
model.fit(feature, target)

#prediction
#age = float(input("Enter your age : "))
#res = model.predict_proba([[age]])
#print(res)


#info = res.ravel().tolist()
#print(info)

#bi_info = round(info[0] * 100, 2)
#ca_info = round(info[1] * 100, 2)
#cy_info = round(info[2] * 100, 2)

#print("Bike = ", bi_info)
#print("car = ", ca_info)
#print("cycle = ", cy_info)

#print test and predict
print(x_test)
print(y_test)
y_pred = model.predict(x_test)
print(model.predict(x_test))

input()

#classification report
cr = classification_report(y_test, y_pred)
print(cr)
