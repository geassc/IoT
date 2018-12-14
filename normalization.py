import sys
import numpy as np

###############
###read data###
###############
datafile = sys.argv[1]
data = np.loadtxt(datafile,dtype=np.float32)
rows = len(data)
print(rows)
cols = len(data[0])
print(cols)

###########################
###normalization z-score###
###########################

mean = (np.mean(data,axis=0))
print(mean)
standard_deviation=(np.std(data,axis=0))
print(standard_deviation)
dataN = np.zeros((rows,cols))
for i in range(0,rows,1):
    for j in range(0,cols,1):
        dataN[i][j] = (data[i][j] - mean[j])/standard_deviation[j]

np.savetxt('trainext_normalization.txt',dataN,fmt='%0.8f')
#np.savetxt('testssext_normalization.txt',dataN,fmt='%0.8f')