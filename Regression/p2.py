import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("area.csv")

#independent variable
feature = data[["area"]] #this is the feature that is the independent variable

#dependent variable
target = data["price"] #this is the target that is dependent variable

#to know what actually feature and target stores
print(feature)
print(target)

model = LinearRegression()  #using the linear Regression model from sklearn library
model.fit(feature, target)  #giving the model the feature and target so that it can learn from the them


area = float(input("Enter the area : ")) #taking the area as an input from the user
price = model.predict([[area]]) # predicting the price of the inputted area
print("Price is : ", round(price[0], 2)) #output the predicted price