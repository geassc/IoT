import sys
import numpy as np

########################
###read training data###
########################
datafile = sys.argv[1]
data = np.loadtxt(datafile,dtype=np.float32)
rows = len(data)
print(rows)
cols = len(data[0])
print(cols)
print(data)

############################
###label initialization#####
############################
trainlabels = {}
for i in range(0,rows,1):
    if i < rows/2:
        trainlabels[i] = -1
    else:
        trainlabels[i] = 1
n = [rows/2,rows/2]

#############################
###compute means&deviation###
#############################
m0 = []
for j in range(0,cols,1):
    m0.append(1)
m1 = []
for j in range(0,cols,1):
    m1.append(1)

for i in range(0, rows, 1):
    if trainlabels[i] == -1:
        for j in range(0, cols, 1):
            m0[j] = m0[j] + float(data[i][j])
    if trainlabels[i] == 1:
        for j in range(0, cols, 1):
            m1[j] = m1[j] + float(data[i][j])
for j in range(0,cols,1):
    m0[j] = m0[j]/n[0]
    m1[j] = m1[j]/n[1]

s0 = []
for j in range(0, cols, 1):
    s0.append(0)
s1 = []
for j in range(0,cols,1):
    s1.append(0)

for i in range(0, rows, 1):
    if trainlabels[i] == -1:
        for j in range(0, cols, 1):
            s0[j] = s0[j] + (data[i][j]-m0[j])**2
    if trainlabels[i] == 1:
        for j in range(0, cols, 1):
            s1[j] = s1[j] + (data[i][j]-m1[j])**2

for j in range(0,cols,1):
    s0[j] = s0[j]/n[0]
    s1[j] = s1[j]/n[1]

########################
###read testing data###
########################
datafile2 = sys.argv[2]
datatest = np.loadtxt(datafile2,dtype=np.float32)
rowstest = len(datatest)
print(rowstest)
colstest = len(datatest[0])
print(colstest)

###################
###classify data###
###################
d0 = 0;
d1 = 0;
pre_f = open('predicted_labels_NB.txt', 'w')
for i in range(0,rowstest,1):
    for j in range(0,colstest,1):
        d0 += (datatest[i][j]-m0[j])**2/s0[j]
        d1 += (datatest[i][j]-m1[j])**2/s1[j]
    if(d0 < d1):
        print("-1 ",i)
        pre_f.write(str(-1)+' '+str(i)+'\n')
    else:
        print("1 ",i)
        pre_f.write(str(1)+' '+str(i)+'\n')
    d0 = 0;
    d1 = 0;
pre_f.close()