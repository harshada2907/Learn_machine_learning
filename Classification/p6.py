#import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#load the dataset
data = pd.read_csv("heart.csv")
print(data)

#check for null data
print(data.isnull().sum())

#features and target
features = data.drop("output", axis = "columns")
target = data["output"]

#train and test
x_train, x_test, y_train, y_test = train_test_split(features, target)

#model building
model = LogisticRegression(max_iter = 3000)
model.fit(x_train, y_train)

#classification report
cr = classification_report(y_test, model.predict(x_test))
print(cr)

