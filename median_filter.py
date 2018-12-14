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

##print data

###################
###median filter###
###################
##window = 5, step = 1##
dataM1 = np.zeros((rows,cols))
for j in range(0,cols,1):
    dataM1[0][j] = (3 * data[0][j] + data[1][j] + data[2][j]) / 5
    dataM1[1][j] = (2 * data[0][j] + data[1][j] + data[2][j] + data[3][j]) / 5
    dataM1[rows-2][j] = (data[rows-4][j] + data[rows-3][j] + data[rows-2][j] + 2 * data[rows-1][j]) / 5
    dataM1[rows-1][j] = (data[rows-3][j] + data[rows-2][j] + 3 * data[rows-1][j]) / 5
    for i in range(2,rows-2,1):
        dataM1[i][j] = (data[i-2][j] + data[i-1][j] + data[i][j] + data[i+1][j] + data[i+2][j])/5

np.savetxt('trainext_median_filter_1.txt',dataM1,fmt='%0.8f')
#np.savetxt('testssext_median_filter_1.txt',dataM1,fmt='%0.8f')
##window = 5, step = 5##
dataM2 = np.zeros((rows//5,cols))
k=0
for j in range(0,cols,1):
    for i in range(0,rows//5,1):
        if i==0:
            dataM2[0][j] = (3 * data[0][j] + data[1][j] + data[2][j]) / 5
        else:
            dataM2[i][j] = (data[5*i-2][j] + data[5*i-1][j] + data[5*i][j] + data[5*i+1][j] + data[5*i+2][j])/5

#np.savetxt('train_median_filter_5.txt',dataM2,fmt='%0.8f')


