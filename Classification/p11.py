#import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

#load the data
data = pd.read_csv("tdsep23.csv")
print(data)

#features and target
features = data[["Height(cm)", "Weight(kg)"]]
target = data["T-Shirt Size"]

mms = MinMaxScaler()
nfeatures = mms.fit_transform(features)

print(features)
print(nfeatures)

N = int(len(data) ** 0.5)
if N % 2 == 0:
	N = N + 1

print(N)

model = KNeighborsClassifier(n_neighbors = N, metric = "euclidean")
model.fit(nfeatures, target)

ht = float(input("Enter your height : "))
wt = float(input("Enter your weight : "))
d = [[ht, wt]]
nd = mms.transform(d)
ts = model.predict(nd)

print(ts[0])

#internal working
nn = model.kneighbors(nd, n_neighbors = N)
print(nn)