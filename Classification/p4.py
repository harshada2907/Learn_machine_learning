#import the libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

#load the data
data = pd.read_csv("ahsep23.csv")
print(data)

#check for null data
print(data.isnull().sum())

#feature and target
feature = data[["age"]]
target = data["have_insurance"]

#train and test
x_train, x_test, y_train, y_test = train_test_split(feature, target)

print(x_train)
print(y_train)

print(x_test)
print(y_test)

#model building
model = LogisticRegression()
model.fit(x_train, y_train)

#confusion matrix display
disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
print(disp.confusion_matrix)

#classification report
cr = classification_report(y_test, model.predict(x_test))
print(cr)

#prediction
age = float(input("Enter your age : "))
ha = model.predict([[age]])
print(ha[0])