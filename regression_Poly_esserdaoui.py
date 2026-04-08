import numpy as np 
import matplotlib.pyplot as plt

#On charge les données (Surface et Prix)
dataset = np.genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(46, 80))

X_single = dataset[:, 0].reshape(-1, 1) # On stocke la colonne de surface
y = dataset[:, 1].reshape(-1, 1)        # On stocke les valeurs de prix 

#On crée une matrice avec X et X au carré (X^2)
X_poly = np.c_[X_single, X_single**2] 

m = len(y) 

#On ajoute une colonne de 1 devant la matrice pour la valeur de départ
X_b = np.c_[np.ones((m, 1)), X_poly] 

XT = X_b.T 
#On calcule la transposée de la matrice 

XR = XT.dot(X_b) 
#On calcule la multiplication matricielle entre la matrice et sa transposée

X_inv = np.linalg.inv(XR) 
#Ici on essaye de trouver l'inverse de la matrice 

X_Y = XT.dot(y)

W = X_inv.dot(X_Y) 
#On calcule les coefficients finaux 

# --------------- Visualisation-------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_single, y, color='blue', alpha=0.3, label='Données réelles')
x_line = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
x_line_b = np.c_[np.ones((100, 1)), x_line, x_line**2]
y_line = x_line_b.dot(W)
plt.plot(x_line, y_line, color='red', linewidth=3, label='Courbe Polynomiale')
plt.xlabel("Surface")
plt.ylabel("Prix")
plt.title("Régression Polynomiale (from scratch)")
plt.legend()
plt.grid(True)
plt.show()
