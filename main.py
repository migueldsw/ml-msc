import numpy as np
from datasets import DATA

DATASETNAMES = ['fac','fou','kar','mor','pix','zer']

xfac,yfac = DATA["fac"]
xfou,yfou = DATA["fou"]

print "data loaded..."

def datasetdetails(dataset,name):
	#print "%s -> instances: %d | dimensions: %d | max/min frst column: %f / %f"%(name,len(dataset), len(dataset[0]), max(dataset[:,0]),min(dataset[:,0]))
	print "%s -> instances: %d | dimensions: %d "%(name,len(dataset), len(dataset[0]))


for n in DATASETNAMES:
	x,y = DATA[n]
	datasetdetails(x,n)

print("----------------")