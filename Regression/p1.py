#This code is just used to show the graphical representation of the datapoints
#how the points are plotted (scattered)

import pandas as pd    #this is a library in pandas that consists of functions like pd.read_csv() as used here

import matplotlib.pyplot as plt #library in python that is used to plot the dataset

data = pd.read_csv("area.csv") #reading the dataset using pandas library
print(data)

x = data["area"] #independent variable

y = data["price"] #dependent variable

plt.scatter(x, y) #this creates a scatter plot
plt.xlabel("Area") #naming the x-axis
plt.ylabel("Price") #naming the y-axis
plt.title("Lonavala Price")
plt.show()
