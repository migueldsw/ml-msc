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
xsmall,ysmall = DATA["small"]

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
	vlist = []
	for k in range(K):
		randIndex = rd.randint(0,len(E)-1)
		while (randIndex in vlist):
			randIndex = rd.randint(0,len(E)-1)
		vlist.append(randIndex)
		#print randIndex, yfac[randIndex]
#>>		g_k = E[randIndex] #G <- instance
		g_k = randIndex # G <- instance's index
		G.append([g_k])
	return np.array(G)
vmv = getInitialVectorOfMedoidsVector #alias

def getInitialVectorOfRelevanceWeightVectors(K):
	L = []
	for k in range(K):
		L_k = [1.]
		L.append(L_k)
	return np.array(L)
gbgl = getInitialVectorOfRelevanceWeightVectors #alias

def getInitialVectorOfMembershipDegreeVectors(E,K): #eq. (6)
	U = []
	n = len(E)
	for i in range(n):
		u_i = []
		for k in range(K):
			argSum = 0
			for h in range(K):
				numerator = 0
				denominator = 0 
				for j in range(p):
#					numerator += L[k][j] * dist(E[i],G[k][j])
					numerator += L[k][j] * D1[i][G[k][j]]
				for j in range(p):
#					denominator+= L[h][j] * dist(E[i],G[h][j])
					denominator+= L[h][j] * D1[i][G[h][j]]
				argSum += ( numerator /(denominator + 1e-25 ))
			u_i_k = (( argSum ** (1./(m-1.)) ) + 1e-25  ) ** -1.
			u_i.append(u_i_k)
		U.append(u_i)
	return np.array(U)
getu = getInitialVectorOfMembershipDegreeVectors #alias

def J(G,L,U,K): #eq. (1) -> objective function
	n = len(U)
	val = 0
	for k in range(K):
		for i in range(n):
			summ = 0
			for j in range(p):
				summ += L[k][j] * D1[i][G[k][j]]
			val += (U[i][k] ** m) * summ
	return val

def step1(E,K,G,L,U): #search for the best medoid vectors -> returns G (updates cluster medoids vector) 
	#Eq. (4)
	nG=[] #new G -> G(t) updated
	n = len(E)
	for k in range(K):
		arglist = []
		for h in range (n):
			summ = 0
			for i in range (n):
				summ += ((U[i][k] ** m) * (D1[i][h]))
			arglist.append(summ)
		l = argminIndex(arglist)
		nG.append([l])
	return np.array(nG)

def step2(E,K,G,L,U): #computes the vector of relevance weights => Eq. (5)
	nL = []
	n = len(D1)
	for k in range(K):
		l_k = []
		for j in range(p):
			prod = 1 
			for h in range(p):
				summ = 0
				for i in range(n):
					summ += ( (U[i][k]**m) * D1[i][G[k][h]] )
				prod *= summ
			numerator = prod
			denominator = 0 
			for i in range(n):
				denominator += ( (U[i][k]**m) * D1[i][G[k][j]] )
			l_k_j = (numerator ** (1./p))/denominator
			l_k.append(l_k_j)
		nL.append(l_k)
	return np.array(nL)



def Ut0(K,n,):
	U = []
	for i in range (n):
		u_i = []
		for k in range(K):
			u_i_k = 1./K 
			u_i.append(u_i_k)
		U.append(u_i)
	return np.array(U)

def argminIndex(list):
	argmin = min(list)
	index = []
	for i in range(len(list)):
		if (list[i] == argmin) :
			index = i
	return index



#------ MVFCMddV init
#def MVFCMddV_init():
#INIT----
E = xsmall
K = 3 #10
m = 1.6
p = 1
D1 = dissimilarityMatrix(E,dist)
G = getInitialVectorOfMedoidsVector(E,K)
L = getInitialVectorOfRelevanceWeightVectors(K)
U = getInitialVectorOfMembershipDegreeVectors(E,K)
#REPEAT----
nG = step1(E,K,G,L,U)
nL = step2(E,K,G,L,U)




#--------------------------------------

def checkdata():
	for n in DATASETNAMES:
		x,y = DATA[n]
		datasetdetails(x,n)
	print("----------------")







