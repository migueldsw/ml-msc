import numpy as np
import scipy as sc
from scipy.spatial import distance as dist
from datasets import DATA
import random as rd 
from datetime import datetime as dt
from report import *
from sklearn.metrics.cluster import adjusted_rand_score

def getHardPartitionList(U):
	instances = []
	for i in U:
		instances.append(argmaxIndex(i,[]))
	return instances
def argmaxIndex(list,exclude):
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

x,y = DATA['fac']
U = np.genfromtxt('./U')
hard = getHardPartitionList(U)

ars = adjusted_rand_score(hard,y)
print "adjusted rand score: %.6f"%ars