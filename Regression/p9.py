import pandas as pd

data = pd.read_csv("datasep2023.csv")
print(data)

#column wise count of null data
r1 = data.isnull().sum()
print(r1)
print()

#salary is null
r2 = data[data.Salary.isnull()]
print(r2)
print()

#experience is null
r3 = data[data.Experience.isnull()]
print(r3)
print()

#age is null
r4 = data[data.Age.isnull()]
print(r4)
print()

#age and position is null
r5 = data[(data.Age.isnull()) & (data.Position.isnull())]
print(r5)