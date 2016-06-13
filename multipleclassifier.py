import numpy as np
import scipy as sc
from scipy.spatial import distance as dist
from datasets import DATA
import random as rd 
from numpy.linalg import inv
from numpy.linalg import det
from collections import Counter as ctr
from sklearn.cross_validation import StratifiedKFold


DATASETNAMES = ['fac','fou','kar','mor','pix','zer']
x0,y0 = DATA[DATASETNAMES[0]]
x1,y1 = DATA[DATASETNAMES[1]]
x2,y2 = DATA[DATASETNAMES[2]]
x3,y3 = DATA[DATASETNAMES[3]]
x4,y4 = DATA[DATASETNAMES[4]]
x5,y5 = DATA[DATASETNAMES[5]]
s1,s2 = DATA['small2']

def apureVotes(li):
	return ctr(li).most_common(1)[0][0]

#-----------------
#------
def classOfIndex(index,indsz):
	#return index/200
	return index/indsz
def printFolds(c,sz):
	lst = create(c,sz)
	for v in lst:
		print "%d   C:  %d"%(v,classOfIndex(v,sz))
def printClasses(y):
	for i in y:
		print classOfIndex(i)
def create(c,it,folds):
	#out = np.array([])
	out = []
	for i in range (c):
		a = np.array(range(it))+(it*i)
		np.random.shuffle(a)
		#out = np.concatenate((out,a))
		out.append(a)
	folds = []	
	for f in range(folds):
		fold = []
		for g in out:
			train = g[:f]
			test = g[f:]

	return out
def getFold(lst,kthFold,numFolds):
	factor = len(lst)/numFolds
	return lst[(kthFold*factor):(kthFold*factor)+factor] #train ,test
#-----
# ----------

def k10foldCrossValidation(DATA,TARGET):
	skf = StratifiedKFold(s2, n_folds = 10, shuffle = True)
	for trainIndexes, testIndexes in skf:
		np.random.shuffle(trainIndexes)
		np.random.shuffle(testIndexes)
		print("%s %s" % (trainIndexes, testIndexes))
		trainX, trainY = listToData(trainIndexes,DATA,TARGET)
		testX, testY = listToData(testIndexes,DATA,TARGET)
		print trainX, trainY, testX, testY


def listToData(lst,DATA,TARGET):
	Xout = []
	Yout = []
	for index in lst:
		Xout.append(DATA[index])
		Yout.append(TARGET[index])
	return np.array(Xout),np.array(Yout)




