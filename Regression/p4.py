#import library
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

#load the data
data = pd.read_csv("area.csv")
print(data)

#feature and target
feature = data[["area"]]

target = data["price"]

#train test spliting of dataset
x_train, x_test, y_train, y_test = train_test_split(feature, target)

print(x_train)
print(y_train)

print(x_test)
print(y_test)

#fit the model
model = LinearRegression()
model.fit(x_train, y_train)

#score
score = model.score(x_test, y_test)
print(score)

#prediction
area = float(input("Enter the area : "))
price = model.predict([[area]])
print("Price is : ", round(price[0], 2))