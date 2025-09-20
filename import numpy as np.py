import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
p = pd.read_csv("C:\\Users\\Anmo2\\Downloads\\archive (3)\\polynomial-regression.csv")
X = p[["araba_max_hiz"]].values
y = p["araba_fiyat"].values

# Polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Scale features for gradient descent
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Gradient Descent setup
m, n = X_poly_scaled.shape
theta = np.zeros(n)   # initialize weights
alpha = 0.01          # learning rate (tune if needed)
epochs = 2000

# Store errors for visualization
mse_list = []

# Gradient Descent loop
for _ in range(epochs):
    predictions = X_poly_scaled.dot(theta)
    errors = predictions - y
    gradients = (1/m) * X_poly_scaled.T.dot(errors)
    theta -= alpha * gradients
    
    mse = np.mean(errors**2)
    mse_list.append(mse)

print("Learned parameters:", theta)

# Predictions
y_pred = X_poly_scaled.dot(theta)

# Sort X for smooth plotting
sorted_idx = X[:, 0].argsort()
X_sorted = X[sorted_idx]
y_sorted = y_pred[sorted_idx]

# Plot regression fit
plt.scatter(X, y, color='red', label='Raw Data')
plt.plot(X_sorted, y_sorted, color='blue', linewidth=2, label='Polynomial Regression (GD)')
plt.xlabel('Max Speed')
plt.ylabel('Car Price')
plt.title('Polynomial Regression with Gradient Descent')
plt.legend()
plt.show()

# Plot MSE convergence
plt.plot(mse_list)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Convergence of Gradient Descent")
plt.show()

# --------------------
# Error metrics
# --------------------
#

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
