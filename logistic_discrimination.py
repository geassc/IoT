import sys
import numpy as np
import random
import math
from math import sqrt

#################
###dot product###
#################
def dotProduct(x, w):
    l = len(x)
    v = 0
    for i in range(0, l, 1):
        v += x[i] * w[i]
    return v

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

################
###initialize###
################
w = []
for j in range(0,cols,1):
    w.append(0.02*random.random()-0.01)

#################################
###gradient descent iterations###
#################################
##median filter & normalization & feature selection##
##eta = 0.00000001
##stopcondition = 0.0286
eta = 0.00000001
stopcondition = 0.002
stop_condition = 10
k = 0
prev = 1
while(stop_condition > stopcondition):
###################
###compute dellf###
###################
    dellf = []
    for j in range(0, cols, 1):
        dellf.append(0)
    for i in range(0, rows, 1):
        if (trainlabels.get(i) != None):
            w0 = w[cols - 1]
            e = math.exp(-dotProduct(data[i], w) - w0)
            for j in range(0, cols - 1, 1):
                dellf[j] += data[i][j] * (trainlabels.get(i) - 1 / (1 + e))
            dellf[cols - 1] += trainlabels.get(i) - (1 / (1 + e))

##############
###update w###
##############
    for j in range(0,cols,1):
        w[j] = w[j] + eta * dellf[j]

#################################
###compute Logistic regression###
#################################
    Logistic = 0
    for i in range(0, rows, 1):
        if (trainlabels.get(i) != None):
            w0 = w[cols - 1]
            e = math.exp(-dotProduct(data[i], w) - w0)
            Logistic += -trainlabels.get(i) * math.log(1 / (1 + e)) - (1 - trainlabels.get(i)) * math.log(e / (1 + e))
    stop_condition = abs(prev - Logistic)
    prev = Logistic
    k += 1
    print("Logistic : ", Logistic, "delta Logistic:", stop_condition)
w[cols-1] = w[cols-1]*2
print("when eta =",eta,"stop condition =",stopcondition)
print("the run time is ",k)
print("w = ",w)
normw = 0
for j in range(0,cols-1,1):
    normw += w[j]**2
normw = abs(sqrt(normw))
print("||w|| = ",normw)
d_origin = abs(w[cols-1]/normw)
print("distance to origin = ",d_origin)

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
pre_f = open('predicted_labels_LD.txt', 'w')
for i in range(0,rowstest,1):
    dp = 0
    for m in range(0,colstest,1):
        dp = dp + w[m] * datatest[i][m]
    if(dp<0):
        pre_f.write(str(-1)+' '+str(i)+'\n')
    else:
        pre_f.write(str(1)+' '+str(i)+'\n')
    d0 = 0;
    d1 = 0;
pre_f.close()
