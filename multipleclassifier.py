import numpy as np
import scipy as sc
from scipy.spatial import distance as dist
from datasets import DATA
import random as rd 
from numpy.linalg import inv
from numpy.linalg import det
from collections import Counter as ctr
from sklearn.cross_validation import StratifiedKFold
from bayesianclassifier import BC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from report import *
from datetime import datetime as dt
import warnings
warnings.filterwarnings("ignore")


DATASETNAMES = ['fac','fou','kar','mor','pix','zer']
#x0,y0 = DATA[DATASETNAMES[0]]
x1,y1 = DATA[DATASETNAMES[1]]
x2,y2 = DATA[DATASETNAMES[2]]
#x3,y3 = DATA[DATASETNAMES[3]]
#x4,y4 = DATA[DATASETNAMES[4]]
x5,y5 = DATA[DATASETNAMES[5]]
#s1,s2 = DATA['small2']

#time cost evaluation
TIME = [] #[t_init,t_final]
def startCrono():
	TIME.append(dt.now())
def getCrono(): # returns delta t in microseconds
	TIME.append(dt.now())
	deltat = TIME[-1]-TIME[-2]
	return deltat.seconds

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
	#create 10 folds
	skf = StratifiedKFold(s2, n_folds = 10, shuffle = True)
	numFold = 1
	totalError = 0
	for trainIndexes, testIndexes in skf:
		print "FOLD %d --------"%(numFold)
		numFold += 1
		np.random.shuffle(trainIndexes)
		np.random.shuffle(testIndexes)
		##print("%s %s" % (trainIndexes, testIndexes))
		trainX, trainY = listToData(trainIndexes,DATA,TARGET)
		testX, testY = listToData(testIndexes,DATA,TARGET)
		##print trainX, trainY, testX, testY
		#classifiers
		bc = BC()
		totalError += evaluate(trainX, trainY, testX, testY,bc)
	print("-------------------------")
	print ("the TOTAL error was %d")%totalError


def listToData(lst,DATA,TARGET):
	Xout = []
	Yout = []
	for index in lst:
		Xout.append(DATA[index])
		Yout.append(TARGET[index])
	return np.array(Xout),np.array(Yout)

def evaluate(trainX,trainY,testX,testY,classifier):
	errors = 0
	print("Training classifier...")
	classifier.fit(trainX,trainY)
	print("Testing classifier...")
	for i in range(len(testX)):
		py = classifier.predict(testX[i])[0]
		if (testY[i] != py):
			errors += 1
	print ("the error was %d in %d tests")%(errors,len(testY))
	return errors

def fitBCs(trainIndexes, datasets):
	classifiers = []
	for (x,y) in datasets:
		cl = BC()
		data, target = listToData(trainIndexes, x, y)
		cl.fit(data, target)
		classifiers.append(cl)
	return classifiers  

def fitSVMs(trainIndexes, datasets):
	classifiers = []
	for (x,y) in datasets:
		cl = SVC(C=1.0, cache_size=200, kernel='rbf')
		data, target = listToData(trainIndexes, x, y)
		cl.fit(data, target)
		classifiers.append(cl)
	return classifiers 

def fitMLPs(trainIndexes, datasets):
	classifiers = []
	for (x,y) in datasets:
		cl =  MLPClassifier(algorithm='l-bfgs', alpha=1e-4, hidden_layer_sizes=(76, 30), random_state=1, momentum=0.8)
		data, target = listToData(trainIndexes, x, y)
		cl.fit(data, target)
		classifiers.append(cl)
	return classifiers 

def votesForInstance(testIndex, classifierList,datasets):
	votes = []
	for i in range(len(classifierList)):
		x,y = datasets[i]
		cl = classifierList[i]
		instance, target = listToData([testIndex],x,y)
		votes.append(cl.predict(instance[0])[0])
	return votes



#k10foldCrossValidation(x1,y1)
#cls = BC()
#cls = SVC(C=1.0, cache_size=200, kernel='rbf')
#cls = MLPClassifier(algorithm='l-bfgs', alpha=1e-4, hidden_layer_sizes=(76, 30), random_state=1, momentum=0.8)
#evaluate(x5,y5,x5,y5,cls)

def evaluateMultiple(trainIndexes,testIndexes,datasetsList,fitClassifiersFunction):
	cll = fitClassifiersFunction(trainIndexes,datasetsList)
	error = 0
	for ind in testIndexes:
		votes = votesForInstance(ind,cll,datasetsList)
		print "x ID: %d(CLASS: %d)  VOTES: %s  APURED VOTE: %d" %(ind,classOfIndex(ind,200),votes,apureVotes(votes))
		selectedClass = apureVotes(votes)
		correctClass = classOfIndex(ind,200)
		if (correctClass != selectedClass):
			print "ERROR!"
			error += 1
	print "--------------------\n  ERROR WAS %d"%error
	return error

#evaluateMultiple([1,2,3,4,5,200,201,203,204],[9,210,11,220])

def k10foldCrossValidation(fitClassifiersFunction):
	#create 10 folds
	skf = StratifiedKFold(y1, n_folds = 10, shuffle = True)
	numFold = 1
	totalError = 0
	datasetsList=[ (x1,y1), (x2,y2), (x5,y5)] ##'fou','kar','zer'
	for trainIndexes, testIndexes in skf:
		np.random.shuffle(trainIndexes)
		np.random.shuffle(testIndexes)
		print "FOLD %d --------"%(numFold)
		numFold += 1
		#
		#clfList = fitBCs(trainIndexes,datasetsList)
		totalError += evaluateMultiple(trainIndexes,testIndexes,datasetsList,fitClassifiersFunction)
	print("-------------------------")
	print ("the TOTAL error was %d")%totalError
	return totalError

def evaluateMultipleAllClassifiers(trainIndexes,testIndexes,datasetsList):
	cllBC = fitBCs(trainIndexes,datasetsList)
	cllSVM = fitSVMs(trainIndexes,datasetsList)
	cllMLP = fitMLPs(trainIndexes,datasetsList)
	error = 0
	for ind in testIndexes:
		votesBC = votesForInstance(ind,cllBC,datasetsList)
		votesSVM = votesForInstance(ind,cllSVM,datasetsList)
		votesMLP = votesForInstance(ind,cllMLP,datasetsList)
		selectedClass = apureVotes([apureVotes(votesBC),apureVotes(votesSVM),apureVotes(votesMLP)])
		correctClass = classOfIndex(ind,200)
		print "x ID: %d(CLASS: %d)  VOTES(BC,SVM,MLP): %s  APURED VOTE: %d" %(ind,correctClass,[votesBC,votesSVM,votesMLP],selectedClass)
		if (correctClass != selectedClass):
			print "ERROR!"
			error += 1
	print "--------------------\n  ERROR WAS %d"%error
	return error
def k10foldCrossValidationAllClassifiers():
	#create 10 folds
	skf = StratifiedKFold(y1, n_folds = 10, shuffle = True)
	numFold = 1
	totalError = 0
	datasetsList=[ (x1,y1), (x2,y2), (x5,y5)] ##'fou','kar','zer'
	for trainIndexes, testIndexes in skf:
		np.random.shuffle(trainIndexes)
		np.random.shuffle(testIndexes)
		print "FOLD %d --------"%(numFold)
		numFold += 1
		#
		#clfList = fitBCs(trainIndexes,datasetsList)
		totalError += evaluateMultipleAllClassifiers(trainIndexes,testIndexes,datasetsList)
	print("-------------------------")
	print ("the TOTAL error was %d")%totalError
	return totalError

def errorToSuccessRate(err):
	return (2000-err)/2000.

###########
###RUN

def RUN(execNum):
	print "######### EXEC.: %d #########"%execNum
	startCrono()
	print("Stratified 10 Fold Cross Validation for Bayesian Classifier") 
	BCError = k10foldCrossValidation(fitBCs)
	print "BC ERROR WAS: %d"%BCError
	timeBC = getCrono()
	print "DONE! in %d seconds"%timeBC

	startCrono()
	print("Stratified 10 Fold Cross Validation for SVM") 
	SVMError = k10foldCrossValidation(fitSVMs)
	print "SVM ERROR WAS: %d"%SVMError
	timeSVM = getCrono()
	print "DONE! in %d seconds"%timeSVM


	startCrono()
	print("Stratified 10 Fold Cross Validation for MLP") 
	MLPError = k10foldCrossValidation(fitMLPs)
	print "MLP ERROR WAS: %d"%MLPError
	timeMLP = getCrono()
	print "DONE! in %d seconds"%timeMLP


	#success rates:
	print "success rates (BC,SVM,MLP)" 
	print (2000-BCError)/2000., (2000-SVMError)/2000., (2000-MLPError)/2000.

	# #>>0.906 0.8925 0.9145
	##>>> print BCError, SVMError, MLPError
	##>>188 215 171

	startCrono()
	print("Stratified 10 Fold Cross Validation for MLP") 
	ALLError = k10foldCrossValidationAllClassifiers()
	print "FINAL ERROR WAS: %d"%ALLError
	timeALL = getCrono()
	print "DONE! in %d seconds"%timeALL
	#>>FINAL ERROR WAS: 157

	print "------------"
	print "ALL TIMES(secs) (%s) SUM= %d secs" %([timeBC,timeSVM,timeMLP,timeALL],sum([timeBC,timeSVM,timeMLP,timeALL]))
	print "------------"
	print "Success Rates (BC,SVM,MLP,ALL): "
	print errorToSuccessRate(BCError)
	print errorToSuccessRate(SVMError)
	print errorToSuccessRate(MLPError)
	print errorToSuccessRate(ALLError)
	appendFile('outQ2/SucRate_BC',[str(errorToSuccessRate(BCError))])
	appendFile('outQ2/SucRate_SVM',[str(errorToSuccessRate(SVMError))])
	appendFile('outQ2/SucRate_MLP',[str(errorToSuccessRate(MLPError))])
	appendFile('outQ2/SucRate_ALL',[str(errorToSuccessRate(ALLError))])

os.system('rm -r outQ2')
os.system('mkdir outQ2')
###RUN TESTS
print"Q2 RUN !"

for e in range(2): #NUM EXECUTIONS
	RUN(e+1)

print "-------------END EXEC."

#>> python multipleclassifier.py > Q2_EXEC_LOG.txt 2> Q2_stderr.txt
