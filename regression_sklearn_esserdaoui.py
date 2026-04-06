import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = np.genfromtxt('train.csv',delimiter=',', skip_header=1, usecols=(46, 80))
X=dataset[:,0].reshape(-1,1)
Y=X=dataset[:,1]

model = LinearRegression()
model.fit(X, Y)

y_pred = model.predict(X)

# 4. Affichage des résultats
plt.scatter(X, Y, color='blue', alpha=0.3, label='Données réelles')
plt.plot(X, y_pred, color='green', linewidth=3, label='Modèle Sklearn')
plt.title("Régression Linéaire avec Scikit-Learn")
plt.xlabel("Surface")
plt.ylabel("Prix")
plt.legend()
plt.show()

print(f"Coefficient (a): {model.coef_[0]}")
print(f"Interception (b): {model.intercept_}")
