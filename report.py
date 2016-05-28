import matplotlib.pyplot as plt
import numpy as np
import os

#pretty printing
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


def checkDir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

#graph plotting
def plotLine(x,y,path,fileName):
	line, = plt.plot(x, y, '-', linewidth=2, color='blue', label='J')
	
	plt.plot(x, y, 'bo')

	#tick no eixo x
	#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
	#tick no eixo y
	#plt.yticks(np.arange(min(x), max(x)+1, .05))

	#defines fixed x y  range
#	plt.axis([0,5,1,2])

	# # draw vertical line from [xfinal,xinicial][yfinal,yinicial]
	# for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:	
	# 	plt.plot([i, i], [2, 1], 'k--')

	#plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
	plt.legend( loc=1, borderaxespad=0.)

	#dashes = [10, 5, 100, 5] # 10 points on, 5 off, 100 on, 5 off
	#line.set_dashes(dashes)
	#plt.show()

	#stub
	for i,j in zip(x,y):
		plt.annotate(str(j),xy=(i,j))
	#end stub


	checkDir(path)
	plt.savefig(path +"/"+fileName)
	plt.clf()
	plt.close()

def plotValuesLine(y,path,fileName):
	plotLine(range(len(y)),y,path,fileName)

def plotDots(y,path,fileName):
	x = range(len(y))
	plt.plot(x, y, 'bo', label= 'classes')
	plt.legend( loc=1, borderaxespad=0.)
	#defines fixed x y  range
	plt.axis([0,len(y),min(y)-1,max(y)+1])
	# for i,j in zip(x,y):
	# 	plt.annotate(str(j),xy=(i,j))
	checkDir(path)
	plt.savefig(path +"/"+fileName)
	plt.clf()
	plt.close()

#file writing
def writeFile(fileName,lines):
	f = open(fileName,'a')
	for l in lines:
		f.write(l + "\n")
	f.close()

x = [0,1,2,3,4,5]
y = [95.858585,57.22,55.5,54.3,54.0,51.82]
y1 = [1,2,3,1,2,1,4,2,3,1,1,1,3,4,1,1,6,2,1,1,7,8,9,0,7]

plotLine(x,y,"out","outtest")
plotValuesLine(y,"out","outtestValues")
plotDots(y1,"out","classes")

writeFile('out/results.txt',['lorem','ipsum','dolor','sit','amet'])