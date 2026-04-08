import numpy as np
import matplotlib.pyplot as plt

def RegreL(dataS,a,b,alpha,it=1000):
  history_J = []
  X=dataS[:,0] #pour prend tous les ligne de premier colone
  Y=dataS[:,1] # pour prend tous les ligne de 2eme colone
  m=len(Y)
  x_max = np.max(X)
  y_max = np.max(Y)
  # On divise par le maximum pour que toutes les valeurs soient entre 0 et 1
  X_=X/np.max(X)
  Y_=Y/np.max(Y)
  for i in range(it):
    #On calcul de la prediction
    Yp=a*X_ +b
    #On calcul le cout J MSE
    cost = (1 / (2 * m)) * np.sum((Yp - Y_)**2)
    history_J.append(cost)
    #on calcule le derive partiale de a et b
    deriv_a= (1/m) * np.sum((Yp - Y_) * X_)
    deriv_b= (1/m) * np.sum(Yp - Y_)

    a=a-alpha * deriv_a
    b=b-alpha * deriv_b

  return a,b, history_J, x_max,y_max
#ici j ai importer une dataSet d apres KAGGLE et On charge le fichier directement dans une matrice de chiffres ou on choisi les colone que je vais utilise
dataset = np.genfromtxt('train.csv', delimiter=',', skip_header=1, usecols=(46, 80))
#appel de la fonction Regression
a_fin, b_fin, erreurs, mx, my = RegreL(dataset, 0.0, 0.0, 0.1)

X_plot = dataset[:, 0] / mx
Y_plot = dataset[:, 1] / my
plt.scatter(X_plot, Y_plot, color='blue', alpha=0.3, label='Maisons réelles (Points)')
Y_predite = a_fin * X_plot + b_fin
plt.plot(X_plot, Y_predite, color='red', linewidth=3, label='Ta Droite de Régression')
plt.title("Résultat de ma Régression 'From Scratch'")
plt.xlabel("Surface (Normalisée)")
plt.ylabel("Prix (Normalisé)")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()

