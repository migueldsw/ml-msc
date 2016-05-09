import numpy as np
import scipy as sc
from scipy.spatial import distance as dist
from datasets import DATA
import random as rd 

dist = sc.spatial.distance.euclidean #set global function 'dist' as the euclidean distance, from scipy implementation

DATASETNAMES = ['fac','fou','kar','mor','pix','zer']

xfac,yfac = DATA["fac"]
xfou,yfou = DATA["fou"]
xkar,ykar = DATA["kar"]

toy = np.array([[1,0,0],[0,1,0],[0,0,0]]) #toy dataset, just for test and validation

print "Multiple Features Data Set loaded..."

def datasetdetails(dataset,name):
	#print "%s -> instances: %d | dimensions: %d | max/min frst column: %f / %f"%(name,len(dataset), len(dataset[0]), max(dataset[:,0]),min(dataset[:,0]))
	print "%s -> instances: %d | dimensions: %d "%(name,len(dataset), len(dataset[0]))

def dissimilarityMatrix(dataset,dist): # returns the : D(nxn) from the dataset of n instances:
	print "...Calculating Dissimilarity Matrix..."
	mat = []
	for i in dataset :
		row = []
		for j in dataset:
			row += [dist(i,j)] 
			#print i, j ,dist(i,j)
		mat.append(row)
	mat=np.array(mat)
	#print mat
	return mat
getdm = dissimilarityMatrix # alias

def verifyMatrix(mat): 
	symmetry = np.allclose(m,np.transpose(m)) #verifies the matrix symmetry
	zeroDiag = np.allclose(np.diag(m),np.zeros(len(m))) #verifies whether all main diagonal values are equals zero
	return symmetry and zeroDiag 
vmat = verifyMatrix #alias

#------ MVFCMddV implementation-------
def getInitialVectorOfMedoidsVector(E,K):
	G=[]
	for k in range(K):
		randIndex = rd.randint(0,len(E)-1)
		#print yfac[randIndex]
		g_k = E[randIndex]
		G.append([g_k])
	return np.array(G)
vmv = getInitialVectorOfMedoidsVector #alias

def getInitialVectorOfRelevanceWeightVectors(K):
	LAMBDA = []
	for k in range(K):
		lambda_k = [1]
		LAMBDA.append(lambda_k)
	return np.array(LAMBDA)
gbgl = getInitialVectorOfRelevanceWeightVectors #alias

def getInitialVectorOfMembershipDegreeVectors(E,K): #eq. (6)
	U = []
	n = len(E)
	for i in range(n):
		u_i = []
		for k in range(K):
			argSum = 0
			for h in range(K):
				argSum += (( LAMBDA[k][0] * dist(E[i], G[k][0] ) )/(  LAMBDA[h][0] * dist(E[i], G[h][0] ) + 1e-25 ))
			u_i_k = ( argSum ** (1./(m-1.)) ) ** -1.
			u_i.append(u_i_k)
		U.append(u_i)
	return np.array(U)

#------ MVFCMddV init
E = xfou
K = 10
m = 1.6
p = 1
D1 = dissimilarityMatrix(E,dist)
G = getInitialVectorOfMedoidsVector(E,K)
LAMBDA = getInitialVectorOfRelevanceWeightVectors(K)
U = getInitialVectorOfMembershipDegreeVectors(E,K)
#--------------------------------------

def checkdata():
	for n in DATASETNAMES:
		x,y = DATA[n]
		datasetdetails(x,n)
	print("----------------")







