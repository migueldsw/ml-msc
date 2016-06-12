import numpy as np
import scipy as sc
from scipy.spatial import distance as dist
from datasets import DATA
import random as rd 
from numpy.linalg import inv
from numpy.linalg import det

DATASETNAMES = ['fac','fou','kar','mor','pix','zer']
x0,y0 = DATA[DATASETNAMES[0]]
x1,y1 = DATA[DATASETNAMES[1]]
x2,y2 = DATA[DATASETNAMES[2]]
x3,y3 = DATA[DATASETNAMES[3]]
x4,y4 = DATA[DATASETNAMES[4]]
x5,y5 = DATA[DATASETNAMES[5]]

toy1 = np.array([[0, 2,9], [1, 1,10], [2, 0,0]]).T
ff = np.array([[1,3],[3,4]])
f2 = np.array([[4.,2.,.6],[4.2,2.1,.59],[3.9,2.,.58],[4.3,2.1,.62],[4.1,2.2,.63]])
det(ff)

#matrix product -> np.dot(m1,m2)
#inverse matrix -> inv(m)
#transpose matrix -> m.T
#a = np.array([5,4])[np.newaxis]
#print a
#print a.T
#matrix det ->  det(m)

def meanVector(M):
	#estimativa de maxima verossimilhanca do vetor de media
	#np.mean(M, axis=0) |or axis=1
	return sum(M)/float(len(M))

def estCov(M):
	mu = meanVector(M)
	out = []
	for i in M:
		v = (i - mu)[np.newaxis]
		out.append(np.dot(v.T,v))
	return ((len(M)-1)**-1)*(sum(out))

def mahalanobisDist(x,M):
	estCovMat = estCov(M)
	iEstCovMat = inv(estCovMat)
	mu = meanVector(M)
	dif = x - mu [np.newaxis]
	return(np.dot(np.dot(dif,iEstCovMat),dif.T)[0][0])

def dens(x,M): #estimador de maxveriss de p(xi|wl) } funcao densidade
	mahalanobis = mahalanobisDist(x,M)
	p = len(x)
	ecov = estCov(M)
	return ((((2*np.pi)**(p/2.)) * (det(ecov)**.5) )**-1) * np.exp(-.5*mahalanobis)

def priori(Y,c): #estimativa maxveriss de p(wl) da classe c a priori
	out = []
	for i in Y:
		if (i == c ):
			out.append(1)
	return sum(out)/float(len(Y))


def totalProb(X,Y,xi):
	terms = []
	classes = np.unique(Y)
	for c in classes:
		v = dens(xi,getXc(X,Y,c)) * priori(Y,c)
		terms.append(v)
	return sum(terms)

def posteriori(X,Y,c,xi): #est. maxveriss. de p(wl|xi) 
	numerator = dens(xi,getXc(X,Y,c)) * priori(Y,c)
	denominator = totalProb(X,Y,xi)
	return (numerator/denominator)


def getXc(X,Y,c): #subset of X -> class 'c'
	Xc = []
	for index in range (len(X)):
		if (Y[index] == c):
			Xc.append(X[index])
	return np.array(Xc)

class BC:
	def __init__(self):
		self.X = []
		self.y = []
		self.classes = []

	def fit(self,X,y):
		self.X = X
		self.y = y
		self.classes = np.unique(self.y)

	def predict(self, x):
		probs = []
		for c in self.classes:
			probs.append(posteriori(self.X,self.y,c,x))
		return self.classes[np.argmax(probs)]






