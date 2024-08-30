#import the library
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#load the dataset
data = pd.read_csv("esmsep23.csv")
print(data)

#feature and target
feature = data[["exp"]]

target = data["sal"]

print(feature)
print(target)

#train and test 
x_train, x_test, y_train, y_test = train_test_split(feature, target)

print("Training data is:")
print(x_train)
print(y_train)

print("Testing data is : ")
print(x_test)
print(y_test)

#model
model = LinearRegression()
model.fit(x_train, y_train)

#score 
s1 = model.score(x_train, y_train)
print("Training score = ", round(s1 * 100, 2), "%")

s2 = model.score(x_test, y_test)
print("Testing score = ", round(s2 * 100, 2) , "%")

#prediction
exp = float(input("Enter the experience : "))
salary = model.predict([[exp]])

print("Salary is : ", salary[0])