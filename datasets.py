import numpy as np
import math

def normalize_columns(data):
    rows, cols = data.shape
    for col in range(0,cols):
        minimo = data[:,col].min()
        maximo = data[:,col].max()
        
        if(minimo != maximo):
            denominador = maximo - minimo
            normazu = (data[:,col] - minimo) / denominador
            data[:,col] = normazu # [max,min] -> [0,1]
            #data[:,col] = (normazu*2) - 1 # [max,min] -> [-1,1]
        else:
            data[:,col] = 0 #column 'col' - numpyarray notation

def generateTargetsArray():
    instances = 200
    out = np.zeros(instances)
    for i in range(1,10):
        out = np.append(out, np.ones(instances) * i )
    return out

#read dataset files: 6 flies: "mfeat-fac", "mfeat-fou", "mfeat-kar", "mfeat-mor", "mfeat-pix" and "mfeat-zer" 
##  READ mfeat-fac  ##
fac = np.genfromtxt('./mfeat/mfeat-fac')
##  READ mfeat-fou  ##
fou = np.genfromtxt('./mfeat/mfeat-fou')
##  READ mfeat-kar  ##
kar = np.genfromtxt('./mfeat/mfeat-kar')
##  READ mfeat-mor  ##
mor = np.genfromtxt('./mfeat/mfeat-mor')
##  READ mfeat-pix  ##
pix = np.genfromtxt('./mfeat/mfeat-pix')
##  READ mfeat-zer  ##
zer = np.genfromtxt('./mfeat/mfeat-zer')


normalize_columns(fac)
normalize_columns(fou)
normalize_columns(kar)
normalize_columns(mor)
normalize_columns(pix)
normalize_columns(zer)

target = generateTargetsArray()

#-------------small data sets - reduced for a faster programming
#small data set: 30 instances -> 3 first classes, 10 instances of each class 
small_mor = np.concatenate([mor[0:10],mor[200:210],mor[400:410]])
small_fou = np.concatenate([fou[0:10],fou[200:210],fou[400:410]])
small_target = np.concatenate([target[0:10],target[200:210],target[400:410]])

#DATASETS: "(x,y)" like, where x = features vector(np.array) and y = targets(classes). 
#all features values normilized into a [0,1] interval
DATA = {
    'fac': (fac,target),
    'fou': (fou,target),
    'kar': (kar,target),
    'mor': (mor,target),
    'pix': (pix,target),
    'zer': (zer,target),
    'small': (small_mor,small_target)
    'small2': (small_fou,small_target)
    }

