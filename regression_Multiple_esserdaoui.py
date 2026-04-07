import numpy as np 
import matplotlib.pyplot as plt

dataset = np.genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(46, 47, 48, 80))
X = dataset[:, :-1] # On essaie de stocker toutes les colonnes sauf la dernière (le prix)
y = dataset[:, -1].reshape(-1, 1)# On stocke les valeurs de prix dans une liste d une colonne
m=len(y) 
X_b = np.c_[np.ones((m, 1)), X] #On ajoute une colonne de 1 devant la matrice X
XT = X_b.T #On calcule la transposée de la matrice
XR=XT.dot(X_b) # On calcule la multiplication matricielle entre la matrice et sa transposée
X_inv= np.linalg.inv(XR) #Ici on essaie de trouver l'inverse de la matrice
X_Y=XT.dot(y)
W=X_inv.dot(X_Y) # On calcule les coefficients finaux

y_pred = X_b.dot(W) #On calcule les prix devinés
erreur = y_pred - y
mse = (1 / (2 * m)) * np.sum(erreur**2)

#----------------------Visualisation--------------------
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='blue', alpha=0.5, label='Prédictions du modèle')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=3, label='Ligne de Perfection (0 erreur)')
plt.xlabel("Vrai Prix (Vérité)")
plt.ylabel("Prix Deviné par ton Code")
plt.title("Régression Multiple : Performance du modèle")
plt.legend()
plt.grid(True)
plt.show()
