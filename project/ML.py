from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib


X = np.array([10, 20, 30, 40, 50, 10, 25, 30, 45, 50, 15, 20, 35, 40, 50, 50, 45, 30, 25, 20, 10])
Y = np.array([30, 55, 75, 95, 115, 30, 65, 75, 105, 115, 45, 85, 150, 200, 250, 300, 320, 270, 100, 80, 35])

data = pd.DataFrame({'Ads/Month':X, 'Paid/Month':Y})
print(data.head())

sns.regplot(x='Ads/Month', y='Paid/Month', data=data)
plt.title("Regression Plot Monthly")
plt.show()

model = LinearRegression().fit(X.reshape(-1, 1), Y)

filename = "model.sav"
joblib.dump(model, filename)

loaded_model = joblib.load(filename)
print(loaded_model.predict([[20]]))
