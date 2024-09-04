import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

data = pd.read_csv("hrsep23.csv")
print(data)

print(data.isnull().sum())

feature = data[["hr"]]
target = data["result"]

x_train, x_test, y_train, y_test = train_test_split(feature, target)

print(x_train)
print(y_train)

print(x_test)
print(y_test)

model = LogisticRegression()
model.fit(x_train, y_train)
print(model.predict(x_test))

input()

disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
print(disp.confusion_matrix)

input()

cr = classification_report(y_test, model.predict(x_test))
print(cr)