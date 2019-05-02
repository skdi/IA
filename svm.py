import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib import style


def inicializar(porcentaje,nombre,col):
	datos=pd.read_csv(nombre,header=0)
	#print(datos)
	#print(datos.ix[0:3])#lectura por filas
    #datos.sort_values(by="n1",ascending=False)#ordenamiento segun ni
    #datos[datos.ix[:,5]]#columna 5
    #tel=datos['n1']#almacenado de columna en tel
    #datos.size()
	x=np.array(datos.ix[0:porcentaje,0:col])#sub matriz X
	y=np.array(datos.ix[0:porcentaje,4])
	x2=np.array(datos.ix[porcentaje:100,0:4])#sub matriz X
	y2=np.array(datos.ix[porcentaje:100,col])
	return x,y,x2,y2



def h(thetas , vc):
	return sum(e[0]*e[i] for e in zip(np.traspose(thetas), vc))

def sigmoidea( thetas , vc):
	return 1/(1+np.exp(-h(thetas,vc)))

def error( x , y ):
	n= len(x)
	sumi = 0 
	for i in range(n):
		sumi = sumi +y[i]*np.log(sigmoidea(x[i])+1-y[i]*np.log(1-sigmoidea(x[i])))
	return -1/100*sumi

def changethetas(thetas,alfa,x,y):
	for j in range (len(x[0])-1) :
		sumi = 0 
		for i in range(x):
			sumi=sumi+ sigmoidea(x[i])-y[i]*x[i][j]
		thethas[j] = thethas[j]-alfa*sumi/len(x)

def prediccion(nom_archivo,x,y,thetas):
	##Falta completar
	y_pred=sigmoidea(tethas,x)

	
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
	x,y,x2,y2=inicializar(70,'test.csv',3)
	#fase de aprendizaje
	tethas=aprendizaje(x,y,'test.csv')
	#fase de testing
	prediccion('test.csv',x2,y2,tethas)

	#L(w,b) = 0.5*w-Sum(lambda[i]*y[i]*(b+trans(w)*x[i]-1))
	#w=sum(lambda[i]*x[i]*y[i])
	#b=-sum(lambda[i]*y[i]*(kernel))
	#kernel=x[j]*x[i]
	#L(lambda[i])=sum(lambda[i])-0.5*sum(lamda[i]*lambda[j]*y[i]*y[j](kernel))
	#donde sum(lamda[i]*y[i]=0) y Lamda[i]>=0

def SupportVectorMachinne():
	x,y,x_test,y_test=inicializar(70,'iris.csv',4)
	
	for val, inp in enumerate(x):
	    if y[val] == 0:
	        plt.scatter(inp[0], inp[1], s=100, marker='_', linewidths=5)

	    else:
	        plt.scatter(inp[0], inp[1], s=100, marker='+', linewidths=5)

	#plt.plot([-2,6],[6,1])
	print(y)
	plt.show()


	w,out = svm_function(x,y)
	print('Pesos calculados')
	print(w)
	#print('salida predicha')
	#print(out)   
	    
	for val, inp in enumerate(x):
	    if y[val] == 0:
	        plt.scatter(inp[0], inp[1], s=100, marker='_', linewidths=5)
	    else:
	        plt.scatter(inp[0], inp[1], s=100, marker='+', linewidths=5)

	
	x1=[w[0],w[1],-w[1],w[0]]
	x2=[w[0],w[1],w[1],-w[0]]

	x1x2 =np.array([x1,x2])
	X,Y,U,V = zip(*x1x2)
	ax = plt.gca()
	ax.quiver(X,Y,U,V,scale=1, color='pink')
	
	result = []
	#Producto punto matriz x_test por w
	for i, val in enumerate(x_test):
	        result.append(np.dot(x_test[i], w))

	print('test result')
	print(result)
	
	plt.show()


def svm_function(x,y):
    #reservo memoria
    w = np.zeros(len(x[0]))
    #learning rate
    l_rate = 1
    #epoch
    epoch = 100000
    #output list
    out = []
    #Fase de entrenamiento
    for e in range(epoch):
        for i, val in enumerate(x):
            val1 = np.dot(x[i], w)
            if (y[i]*val1 < 1):
                w = w + l_rate * ((y[i]*x[i]) - (2*(1/epoch)*w))
            else:
                w = w + l_rate * (-2*(1/epoch)*w)
    
    for i, val in enumerate(x):
        out.append(np.dot(x[i], w))
    
    return w, out


SupportVectorMachinne()
