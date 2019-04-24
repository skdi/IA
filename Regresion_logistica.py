import numpy as np


def inicializar():
	f= open("test.csv")
	linea=f.readline()
	i=0
	x=np.arange(70,5)

	while(linea!= "" nd i<70):
		valor=""
		for j in linea:
			if(j!=","):
				valor=valor+j
			else:
				x[i][j]=float(valor)
				#x.append(float(valor))
		i=i+1		
		linea=f.readline() 
    


def h(thetas , vc):
	return sum(e[0]*e[i] for e in zip(np.traspose(thetas), vc))

def s( thetas , vc):
	return 1/(1+np.exp(-h(thetas,vc)))

def error( x , y ):
	n= len(x)
	sum = 0 
	for i in range(n):
		sum = sum +y[i]*np.log(s(x[i])+1-y[i]*np.log(1-s(x[i])))
	return -1/100*sum

def chanchethetas(thetas,alfa,x,y):
	for j in range (len(x[0])-1) :
		sum = 0 
		for i in range(x):
			sum=sum+ s(x[i])-y[i]*x[i][j]
		thethas[j] = thethas[j]-alfa*sum/len(x)

def testear(nom_archivo,x,y,thetas):

	for i in range():
		errores = [ s(e) for e in x[i] ]

	
def rl(x,y):
	alfa= 0.07
	umbral= 0.01

	for i in range(0,4):
		thetas[i]= np.random.rand(0,10)
	
	error_rl = error(x,y)
	while(error_rl>umbral):
		chanchethetas(thetas,alfa,x,y)
		error_rl = error(x,y)
	testear("test.csv",x,y,thetas)

inicializar()
