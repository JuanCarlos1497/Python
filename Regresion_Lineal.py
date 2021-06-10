
# −∗− coding: utf−8−∗−
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.random.seed (42)
#definimos la cantidad de muestras
m = 100
#generamos una serie d enumeros aleatorios entre −3 y 3
X = 6 * np.random.rand(m, 1)-3
#obtenemos los valores de Y segun una formula dada en este caso
y = 0.5 *  X**2 + X + 2 + np.random.randn(m, 1)
plt.plot(X, y, "b.")
plt.xlabel("$x$", fontsize =18)
plt.ylabel("$y$", rotation =0, fontsize =18)
plt.axis([-3, 3, 0, 10])
plt.show ()
#creamos el objeto para realizar la regresion lineal
lin_reg = LinearRegression()
#entrenamos con nuestros datos de entrada y salida
lin_reg.fit(X, y)
#hacemos el print the alpha y beta en este caso beta solo es un numero ya que solo tenemos un dato por cada muestra
print( lin_reg.intercept_ , lin_reg.coef_)
#creamos una muestra de 100 valores igualmente separados entre −3 y 3
X_new =np.linspace(-3, 3, 100). reshape (100 , 1)
y_new = lin_reg.predict( X_new )
#ploteamos la linea de la regresion
plt.plot( X_new , y_new , "r-", linewidth =2, label="Predictions")
plt.xlabel("$x 1$", fontsize =18)
plt.ylabel("$y$", rotation =0, fontsize =18)
plt.legend(loc="upper left", fontsize =14)
plt.axis([-3, 3, 0, 10])
plt.show()