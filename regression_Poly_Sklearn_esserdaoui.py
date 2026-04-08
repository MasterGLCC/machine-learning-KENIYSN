import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

#On charge les données 
dataset = np.genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(46, 80))

X = dataset[:, 0].reshape(-1, 1) # On stocke la colonne de surface
y = dataset[:, 1]               # On stocke les valeurs de prix

#Cet outil va créer lui-même les colonnes X et X^2
poly_transformer = PolynomialFeatures(degree=2)
X_poly = poly_transformer.fit_transform(X)

#On crée et on entraîne le modèle de régression
model = LinearRegression()
model.fit(X_poly, y)

#On demande au modèle de faire des prédictions
y_pred = model.predict(X_poly)

#On calcule l'erreur (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Erreur Moyenne (MSE) avec Sklearn : {mse}")

# ----------------Visualisation ------------------
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.3, label='Données réelles')
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_range_poly = poly_transformer.transform(x_range)
y_curve = model.predict(x_range_poly)
plt.plot(x_range, y_curve, color='red', linewidth=3, label='Courbe Polynomiale')
plt.xlabel("Surface")
plt.ylabel("Prix")
plt.title("Régression Polynomiale avec sklearn")
plt.legend()
plt.grid(True)
plt.show()
