from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Anmo2\\Desktop\\city_temperature.csv")

df = df.dropna(subset=["AvgTemperature", "Year"])

x= df[["AvgTemperature"]]   
y = df["Year"]

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)


plt.scatter(x, y, color='blue', alpha=0.3, label='Actual Data')
plt.plot(x, y_pred, color='red', label='Linear Regression Line')
plt.xlabel("Average Temperature (°F or °C)")
plt.ylabel("Year")
plt.title("Linear Regression: Year vs Average Temperature")
plt.legend()
plt.show()
