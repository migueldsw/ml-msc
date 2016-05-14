import numpy as np
import scipy as sc
from scipy.spatial import distance as dist
from datasets import DATA
import random as rd 
from datetime import datetime as dt

dist = sc.spatial.distance.euclidean #set global function 'dist' as the euclidean distance, from scipy implementation

DATASETNAMES = ['fac','fou','kar','mor','pix','zer']

def datasetdetails(dataset,name):
	#print "%s -> instances: %d | dimensions: %d | max/min frst column: %f / %f"%(name,len(dataset), len(dataset[0]), max(dataset[:,0]),min(dataset[:,0]))
	print "%s -> instances: %d | dimensions: %d "%(name,len(dataset), len(dataset[0]))

def dissimilarityMatrix(dataset,dist): # returns the : D(nxn) from the dataset of n instances:
	print "Calculating Dissimilarity Matrix..."
	mat = []
	for i in dataset :
		row = []
		for j in dataset:
			row += [dist(i,j)] 
			#print i, j ,dist(i,j)
		mat.append(row)
	mat=np.array(mat)
	#print mat
	print "...Done!"
	return mat
getdm = dissimilarityMatrix # alias

def verifyMatrix(mat): 
	symmetry = np.allclose(m,np.transpose(m)) #verifies the matrix symmetry
	zeroDiag = np.allclose(np.diag(m),np.zeros(len(m))) #verifies whether all main diagonal values are equals zero
	return symmetry and zeroDiag 
vmat = verifyMatrix #alias

#------ MVFCMddV implementation-------
def getInitialVectorOfMedoidsVector(D,K):
	G=[]
	p = len(D)
	for k in range(K):
		g_k = []
		vlist = []
		for j in range(p):
			randIndex = rd.randint(0,len(D[j])-1)
			while (randIndex in vlist):
				randIndex = rd.randint(0,len(D[j])-1)
			vlist.append(randIndex)
#>>			g_k = E[randIndex] #G <- instance
			g_k_j = randIndex # G <- instance's index
			g_k.append(g_k_j)
		G.append(g_k)
	#print "G INIT"
	return np.array(G)
vmv = getInitialVectorOfMedoidsVector #alias

def getInitialVectorOfRelevanceWeightVectors(D,K):
	L = []
	p = len(D)
	for k in range(K):
		l_k = []
		for h in range(p):
			l_k.append(1.)
		L.append(l_k)
	#print "L INIT"
	return np.array(L)
gbgl = getInitialVectorOfRelevanceWeightVectors #alias

def getInitialVectorOfMembershipDegreeVectors(m,D,K,G,L): #eq. (6)
	U = []
	n = len(D[0])
	p = len(D)
	for i in range(n):
		u_i = []
		for k in range(K):
			summ = 0
			for h in range(K):
				numerator = 0
				denominator = 0 
				for j in range(p):
#					numerator += L[k][j] * dist(E[i],G[k][j])
					numerator += L[k][j] * D[j][i][G[k][j]]
				for j in range(p):
#					denominator+= L[h][j] * dist(E[i],G[h][j])
					denominator+= L[h][j] * D[j][i][G[h][j]]
				summ += ( numerator /(denominator + 1e-25 )) ** (1./(m-1.))
			u_i_k = (( summ  ) + 1e-25  ) ** -1.
			u_i.append(u_i_k)
		U.append(u_i)
	#print "U INIT"
	return np.array(U)
getu = getInitialVectorOfMembershipDegreeVectors #alias

def J(m,D,K,G,L,U): #eq. (1) -> objective function
	n = len(D[0])
	p = len(D)
	val = 0
	for k in range(K):
		for i in range(n):
			summ = 0
			for j in range(p):
				summ += L[k][j] * D[j][i][G[k][j]]
			val += (U[i][k] ** m) * summ
	return val

def step1(m,D,K,G,L,U): #search for the best medoid vectors -> returns G (updates cluster medoids vector) 
	#Eq. (4)
	nG=[] #new G -> G(t) updated
	n = len(D[0])
	p = len(D)
	for k in range(K):
		g_k = []
		for j in range(p):
			arglist = [] 
			for h in range (n):
				summ = 0
				for i in range (n):
					summ += ((U[i][k] ** m) * (D[j][i][h]))
				arglist.append(summ)
			l = argminIndex(arglist,g_k)
			g_k_j = l
			g_k.append(g_k_j)
		nG.append(g_k)
	return np.array(nG)

def step2(m,D,K,G,L,U): #computes the vector of relevance weights => Eq. (5)
	nL = []
	n = len(D[0])
	p = len(D)
	for k in range(K):
		l_k = []
		for j in range(p):
			prod = 1 
			for h in range(p):
				summ = 0
				for i in range(n):
					summ += ( (U[i][k]**m) * D[h][i][G[k][h]] )
				prod *= summ
			numerator = prod
			denominator = 0 
			for i in range(n):
				denominator += ( (U[i][k]**m) * D[j][i][G[k][j]] )
			l_k_j = (float(numerator) ** (1./p))/denominator
			l_k.append(l_k_j)
		nL.append(l_k)
	return np.array(nL)

def step3(m,D,K,G,L,U): #best fuzzy partition -> Eq. (6) 
	nU = []
	n = len(D[0])
	p = len(D)
	for i in range(n):
		u_i = []
		for k in range(K):
			summ = 0
			for h in range(K):
				numerator = 0
				denominator = 0 
				for j in range(p):
					numerator += L[k][j] * D[j][i][G[k][j]]
				for j in range(p):
					denominator+= L[h][j] * D[j][i][G[h][j]]
				summ += ( float(numerator) /(denominator + 1e-25 )) ** (1./(m-1.))
			u_i_k = ( summ  + 1e-25  ) ** -1.
			u_i.append(u_i_k)
		nU.append(u_i)
	return np.array(nU)

def Ut0(K,n,):
	U = []
	for i in range (n):
		u_i = []
		for k in range(K):
			u_i_k = 1./K 
			u_i.append(u_i_k)
		U.append(u_i)
	return np.array(U)

def argminIndex(list,exclude):
	argmin = min(list)
	argmax = max(list)
	indexList = []
	for i in range(len(list)):
		if (list[i] == argmin) :
			if i in exclude:
				list[i] = argmax
				i = argminIndex(list,exclude)	
			indexList.append(i) 
	return indexList[0]

#time cost evaluation
TIME = [] #[t_init,t_final]
def startCrono():
	TIME.append(dt.now())
def getCrono(): # returns delta t in microseconds
	TIME.append(dt.now())
	deltat = TIME[-1]-TIME[-2]
	return deltat.seconds

def isUok(U):
	isok = True
	for i in U:
		if (sum(i)>1.00001):
			isok = False
	return isok

#--------------------------------
#datasets 
xfac,yfac = DATA["fac"]
xfou,yfou = DATA["fou"]
xkar,ykar = DATA["kar"]
xsmall,ysmall = DATA["small"]
xsmall2,ysmall2 = DATA["small2"]
xsmall3,ysmall3 = DATA["small3"]
toy = np.array([[1,0,0],[0,1,0],[0,0,0]]) #toy dataset, just for test and validation
E = xsmall
E2 = xsmall2
E3 = xsmall3
# E = xfac
# E2 = xfou
# E3 = xkar
print "Multiple Features Data Set loaded..."

#------ Calculate Dissimilarity Matrices
startCrono()
D1 = dissimilarityMatrix(E,dist)
D2 = dissimilarityMatrix(E2,dist)
D3 = dissimilarityMatrix(E3,dist)
print "matrices calculated in %d s"%(getCrono())

def runMVFCMddV():
	#INIT---- t=0
	startCrono()
	K = 3 #10
	m = 1.6 
	p = 3 
	e = 10 ** -10
	T = 150
	D = np.array([D1,D2,D3])
	G = getInitialVectorOfMedoidsVector(D,K)
	L = getInitialVectorOfRelevanceWeightVectors(D,K)
	U = getInitialVectorOfMembershipDegreeVectors(m,D,K,G,L)
	u_t = J(m,D,K,G,L,U)
	u_t_m1 = np.inf 
	print "t=0: J(v) = %f"%(u_t)
	#REPEAT... UNTIL t>T
	for t in range(T):
		nG = step1(m,D,K,G,L,U)
		nL = step2(m,D,K,nG,L,U)
		nU = step3(m,D,K,nG,nL,U)
		u_t_m1 = u_t
		u_t = J(m,D,K,nG,nL,nU)
		udif = u_t - u_t_m1
		print "t=%d: J(v) = %f | dif= %f"%(t+1,u_t,udif)
		G = nG
		L = nL
		U = nU
		# UNTIL (OR) |u_t - u_t-1| < e 
		if (abs(udif)<e):
			print "     |u_t - u_t-1| < e"
			break

	print "done in %d s"%(getCrono())
	#print "t=%d: J = %f"%(t+1,J(m,D,K,G,L,U))
	verifyU(U)
	return G,L,U



#--------------------------------------

def checkdata():
	for n in DATASETNAMES:
		x,y = DATA[n]
		datasetdetails(x,n)
	print("----------------")

def verifyU(U):
	if (isUok(U)):
		print "U OK! "
	else :
		print fU 
		print U
		print fG
		print G
		print " FAIL U!!!!"


print "RUNNING MVFCMddV! "
G,L,U = runMVFCMddV()
print "END EXECUTION! "

def clusters(U):
	mapped = []
	index = 0
	for i in U:
		mapped.append((index,argminIndex(i,[])))
		index += 1
	clusters = []
	for k in range(len(U[0])):
		clusters.append([])
	for i in mapped:
		clusters[i[-1]].append(i[0])
	return clusters	

C = clusters(U)


