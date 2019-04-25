import numpy as np
import pandas as pd
from pandas import DataFrame
def inicializar(porcentaje,nombre):
	datos=pd.read_csv(nombre,header=0)
	#print(datos)
	#print(datos.ix[0:3])#lectura por filas
    #datos.sort_values(by="n1",ascending=False)#ordenamiento segun ni
    #datos[datos.ix[:,5]]#columna 5
    #tel=datos['n1']#almacenado de columna en tel
    #datos.size()
	x=datos.ix[0:porcentaje,0:4]#sub matriz X
	y=datos.ix[0:porcentaje,4]
	x2=datos.ix[porcentaje:100,0:4]#sub matriz X
	y2=datos.ix[porcentaje:100,4]
	return x,y,x2,y2



def h(thetas , vc):
	return sum(e[0]*e[i] for e in zip(np.traspose(thetas), vc))

def s( thetas , vc):
	return 1/(1+np.exp(-h(thetas,vc)))

def error( x , y ):
	n= len(x)
	sumi = 0 
	for i in range(n):
		sumi = sumi +y[i]*np.log(s(x[i])+1-y[i]*np.log(1-s(x[i])))
	return -1/100*sumi

def changethetas(thetas,alfa,x,y):
	for j in range (len(x[0])-1) :
		sumi = 0 
		for i in range(x):
			sumi=sumi+ s(x[i])-y[i]*x[i][j]
		thethas[j] = thethas[j]-alfa*sumi/len(x)

def prediccion(nom_archivo,x,y,thetas):
	##Falta completar
	y_pred=s(tethas,x)

	
def aprendizaje(x,y,nombre):
	alfa= 0.07
	umbral= 0.01
	
	for i in range(0,4):
		thetas[i]= np.random.rand(0,10)
	
	error_rl = error(x,y)
	while(error_rl>umbral):
		changethetas(thetas,alfa,x,y)
		error_rl = error(x,y)
	return thetas

def metodo():
	x,y,x2,y2=inicializar(70,'test.csv')
	#fase de aprendizaje
	tethas=aprendizaje(x,y,'test.csv')
	#fase de testing
	prediccion('test.csv',x2,y2,tethas)
	