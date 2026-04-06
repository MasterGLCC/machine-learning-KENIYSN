import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = np.genfromtxt('train.csv',delimiter=',', skip_header=1, usecols=(46, 80))
X=dataset[:,0].reshape(-1,1)
Y=dataset[:,1]
  #Creation d une instance de l objet LinearRegression
model = LinearRegression()
  #on utilisant l'algorithme des Moindres Carrés Ordinaires la méthode fit prend les donner d entrer et les resultat attendus pour apprendre la relation mathématique qui les lie.
model.fit(X, Y)

y_pred = model.predict(X)

  #Affichage des resultats
plt.scatter(X, Y, color='blue', alpha=0.3, label='Données réelles')
plt.plot(X, y_pred, color='green', linewidth=3, label='Modèle Sklearn')
plt.title("Régression Linéaire avec Scikit-Learn")
plt.xlabel("Surface")
plt.ylabel("Prix")
plt.legend()
plt.show()

print(f"Coefficient (a): {model.coef_[0]}")
print(f"Interception (b): {model.intercept_}")
