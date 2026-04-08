import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#On charge les données avec genfromtxt
dataset = np.genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(46, 47, 48, 80))

#On sépare les caractéristiques (X) et le prix (y)
X = dataset[:, :-1] # On stocke toutes les colonnes sauf le prix
y = dataset[:, -1]  # On stocke les valeurs de prix (Sklearn accepte les vecteurs plats)

#On crée l'outil de régression (l'objet modèle)
model = LinearRegression()

#On entraîne le modèle (Calcul automatique des coefficients)
model.fit(X, y)

#On demande au modèle de faire des prédictions sur nos données
y_pred = model.predict(X)

#On affiche les résultats trouvés par la bibliothèque
print(f"Valeur de départ (b) : {model.intercept_}")
print(f"Coefficients (a1, a2, a3) : {model.coef_}")

#On calcule l'erreur avec la fonction de la bibliothèque
mse = mean_squared_error(y, y_pred)
print(f"Erreur moyenne (MSE) : {mse}")

# ---------------------- Visualisation --------------------
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='purple', alpha=0.5, label='Prédictions Sklearn')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=3, label='Ligne Idéale')
plt.xlabel("Vrai Prix ")
plt.ylabel("Prix Deviné ")
plt.title("Régression Multiple avec Sklearn")
plt.legend()
plt.grid(True)
plt.show()
