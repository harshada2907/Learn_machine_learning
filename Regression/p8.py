#import the library

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("applesep2023.csv")
print(data)

print(data.isnull().sum())
print()

#in the last practical we dropped the null values 
#but in case if the dataset is small and we drop the null values the size of the dataset will reduce and the model will not be able to train properly as the dataset is small

#so in such cases we can fill those null values
#now these null values can be filled manually too but if the dataset is large then it will consume time and not worthy
#so we can fill those values code.
#now for filling those values there are 3 statistical methods
#mean  median  mode

#mean and median can be used if the values are numerical
#median and mode can be used if the values are categorical

ndata = data.fillna({"price" : data["price"].mean()})

print(ndata.isnull().sum())
print(ndata)

#feature and target
feature = ndata[["qty"]]
target = ndata["price"]

#train and test
x_train, x_test, y_train, y_test = train_test_split(feature, target, random_state = 42)


#model making
model = LinearRegression()
model.fit(x_train, y_train)

#scores

s1 = model.score(x_train, y_train)
print("Training score is : ", round(s1 * 100, 2))

s2 = model.score(x_test, y_test)
print("Testing score is : ", round(s2 * 100, 2))

#prediction
qty = float(input("Enter the quantity : "))
price = model.predict([[qty]])

print("Price is : ", price)