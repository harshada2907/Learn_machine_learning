#drop null data

import pandas as pd

data = pd.read_csv("datasep2023.csv")
print(data)
print()

#drop is any value is null
d1 = data.dropna(how = "any")
print(d1)
print()

#drop is all values are null
d2 = data.dropna(how = "all")
print(d2)
print()

#drop if salary is null
d2 = data.dropna(subset = ["Salary"])
print(d2)
print()

#drop is age is null
d3 = data.dropna(subset = ["Age"])
print(d3)
print()

#drop if age and position is null
d4 = data.dropna(subset = ["Age", "Position"], how = "all")
print(d4)
print()

d5 = data.dropna(thresh = 3)
print(d5)
print()

data.dropna(how = "any", inplace = True) #inplace = True will make the changes in the same dataset itself rather than addding it to a new variable
print(data)