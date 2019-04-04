import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
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

cant_muestras=506

#X
x_multiple= boston.data[:,5:8]
y_multiple = boston.target
#y_multiple.reshape(cant_muestras,1)
#r = np.sqrt(np.power(x_multiple,2)+np.power(y_multiple,2))

#inicializamos las variables de entrenamiento usando el 20% de los datos
x_train, x_test, y_train, y_test = train_test_split(x_multiple, y_multiple, test_size=0.2)
#definimos el algoritmo a utilizar
regre_multiple = linear_model.LinearRegression()

#entrenando el modelo

regre_multiple.fit(x_train,y_train)

#realizando la prediccion
y_pred_multiple = regre_multiple.predict(x_test)

#datos de la regresion
print()
print("Datos de la regresion lineal")
print("pendiente:",regre_multiple.coef_," intercepto:",regre_multiple.intercept_)
print("Precision:",regre_multiple.score(x_train,y_train))


#grafica de los datos y el modelo
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred_multiple,color="red",linewidth=3)
plt.title("regresion_lineal")
plt.xlabel("numero de habitaciones")
plt.ylabel("Valor medio")
plt.show()

