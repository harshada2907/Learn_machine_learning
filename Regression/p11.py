import pandas as pd

data = pd.read_csv("datasep2023.csv")
print(data)


#replacing the null value with constant
d1 = data.fillna({"Salary" : 8000})
print(d1)

#replacing the null value with mean
d2 = data.fillna({"Salary" : data["Salary"].mean()})
print(d2)

#replacing the null value with median
d3 = data.fillna({"Salary" : data["Salary"].median()})
print(d3)

#replacing the null values with mode
d4 = data.fillna({"Position" : data["Position"].mode()[0]})
print(d4)

data.fillna({
	"Age" : data["Age"].mean(),
	"Position" : "Unallocated",
	"Experience" : data["Experience"].median(),
	"Salary" : data["Salary"].mode()[0]

}, inplace = True)

print(data)

