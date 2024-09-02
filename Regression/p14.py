#sometimes LinearRegression doesn't suit the dataset as the data is varying in polynomial #equation form and not linearly so the model will not work properly with just linear model
#so lets see how to handle it

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("pssep2023.csv")
print(data)

feature = data[["Level"]]
target = data["Salary"]

model = LinearRegression()
model.fit(feature, target)

plt.scatter(data["Level"], data["Salary"], color = "red")
plt.plot(data["Level"], model.predict(feature), color = "blue")
plt.show()

#when we plot this graph we can see that the line is not fitted properly because of which the model will not predict accurately
#so we have some steps to handle that
#so how to handle is shown in p15.py python file