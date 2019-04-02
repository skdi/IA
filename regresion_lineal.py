import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
#base de datos para el ejemplo
boston = datasets.load_boston() 
#print(boston)
#informacion de los datos
print("informacion de datos")
print(boston.keys())
print()
print("cantidad de datos")
print(boston.data.shape)
print()
print("nombres de columnas")
print(boston.feature_names)

#seleccionamos la variable dependiente tomando una columna de los datos dados
X = boston.data[:,np.newaxis,5]
#print(X)
#seleccionamos la variable independiente tomando la columna de etiquetas de target
Y = boston.target

#graficando datos 
plt.scatter(X,Y)
plt.xlabel("numero de habitaciones")
plt.ylabel("Valor medio")
plt.show()

### REGRESION LINEAL 
X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

#algoritmo a usar
regresion_lineal = linear_model.LinearRegression()

#entrenamiento
regresion_lineal.fit(X_train,Y_train)

#prediccion
Y_pred = regresion_lineal.predict(X_test)


#datos de la regresion
print()
print("Datos de la regresion_lineal")
print("pendiente:",regresion_lineal.coef_," intercepto:",regresion_lineal.intercept_)
print("Precision:",regresion_lineal.score(X_train,Y_train))

#grafica de los datos y el modelo
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred,color="red",linewidth=3)
plt.title("regresion_lineal")
plt.xlabel("numero de habitaciones")
plt.ylabel("Valor medio")
plt.show()
