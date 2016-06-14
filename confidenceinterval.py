import numpy as np
import scipy as sp
import scipy.stats

test = [2,3,3,4,5,5,2,3,4,5] #means sequence

def meanConfidenceInterval(data, confidenceLevel=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, s = np.mean(a), scipy.stats.sem(a)
    #print sp.stats.t._ppf((1+confidenceLevel)/2., n-1)
    h = s * sp.stats.t._ppf((1+confidenceLevel)/2., n-1)
    return (m, m-h, m+h)
mci = meanConfidenceInterval

def printInterval((estimation, lowvalue, upvalue)):
	print "Estimated Mean: %.6f | INTERVAL: [%.6f,%.6f]"%(estimation, lowvalue, upvalue)
#test
#print meanConfidenceInterval(data1)

#Success Rates Values from the executions
#BC
BC = [0.9025,0.902,0.9025,0.905,0.9005,0.9105,0.899,0.911,0.906,0.9075,0.9085,0.9035,0.898,0.9055,0.899,0.9065,0.903,0.904,0.9055,0.9065,0.9035,0.9005,0.9115,0.9045,0.905]

#SVM
SVM = [0.8905,0.8905,0.8895,0.89,0.8905,0.8905,0.8905,0.891,0.8885,0.893,0.888,0.8885,0.8895,0.8875,0.895,0.8935,0.8865,0.886,0.8875,0.887,0.8925,0.8875,0.8865,0.884,0.889]

#MLP
MLP = [0.9195,0.914,0.925,0.92,0.9225,0.9125,0.913,0.9095,0.916,0.916,0.915,0.9195,0.9135,0.915,0.921,0.919,0.9145,0.9135,0.9205,0.9185,0.916,0.9205,0.9125,0.921,0.924]

#ALL 
ALL = [0.9295,0.9205,0.923,0.9235,0.926,0.9205,0.916,0.927,0.926,0.9245,0.9235,0.925,0.9185,0.9205,0.922,0.9235,0.923,0.925,0.922,0.9185,0.922,0.9225,0.924,0.916,0.9225]

print"For Success Rate Mean: "
print"BC:"
printInterval(meanConfidenceInterval(BC))
print"SVM:"
printInterval(meanConfidenceInterval(SVM))
print"MLP:"
printInterval(meanConfidenceInterval(MLP))
print"ALL:"
printInterval(meanConfidenceInterval(ALL))
print"confidence Level = 95%"