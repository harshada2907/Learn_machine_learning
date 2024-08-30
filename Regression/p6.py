from tkinter import Label, Entry, Button, Tk
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

root = Tk()
root.title("Salary Predictor")
root.geometry("700x600+50+50")
f = ("Arial", 30, "bold")

lab_header = Label(root, text = "Salary Predictor", font = f)
lab_header.pack(pady = 20)

def find():
	try:
		data = pd.read_csv("esmsep23.csv")
		feature = data[["exp"]]
		target = data["sal"]
		x_train, x_test, y_train, y_tets = train_test_split(feature, target)
		model = LinearRegression()
		model.fit(x_train, y_train)
		exp = float(ent_exp.get())
		sal = model.predict([[exp]])
		msg = "Salary = " + str(round(sal[0], 2)) + "K"
		lab_sal.configure(text = msg)
	except ValueError:
		msg = "You should enter numbers only"
		lab_sal.configure(text = msg)

lab_exp = Label(root, text = "Enter experience", font = f)
ent_exp = Entry(root, font = f)

btn_predict = Button(root, text = "Predict Salary", font = f, comman = find)
lab_sal = Label(root, font = f)
lab_exp.pack(pady = 10)
ent_exp.pack(pady = 10)
btn_predict.pack(pady = 10)
lab_sal.pack(pady = 10)

root.mainloop()