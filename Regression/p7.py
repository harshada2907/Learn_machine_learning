#import the library
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#load the data
data = pd.read_csv("applesep2023.csv")
print(data)

#feature and target
feature = data[["qty"]]
target = data["price"]

model = LinearRegression(random_state = 42)
model.fit(feature, target)

#train and test
x_train, x_test, y_train, y_test = train_test_split(feature, target)

#training score 

s1 = model.score(x_train, y_train)
print("Training score is: ", s1)

s2 = model.score(x_test, y_test)
print("Testing score is: ", s2)

plt.scatter(feature, target)
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.title("Price Prediction")
plt.show()

qty = float(input("Enter the quantity : "))
price = model.predict([[qty]])

print("The price is : ", round(price[0], 2))