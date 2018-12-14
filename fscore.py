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

############################
###label initialization#####
############################
trainlabels = {}
for i in range(0,rows,1):
    if i < rows/2:
        trainlabels[i] = -1
    else:
        trainlabels[i] = 1


###############
###f1score#####
###############
f_score = []
for i in range(0,cols,1):
    f_score.append(1)
mean_x = (np.mean(data,axis=0))


#k = 0
for i in range(0,cols,1):
    NP = 0
    NN = 0
    SP = 0
    SN = 0
    MP = 0
    MN = 0
# k += 1
    for j in range(0,rows,1):
        if(trainlabels.get(j)!= None and trainlabels.get(j) == 1):
            NP +=1
            SP += (data[j][i] - mean_x[i])**2
            MP += data[j][i]
        if(trainlabels.get(j)!= None and trainlabels.get(j) == -1):
            NN +=1
            SN += (data[j][i] - mean_x[i])**2
            MN += data[j][i]
    MP = MP/NP
    MN = MN/NN
# print(k,np,nn,sp,sn)
    f_score[i] = ((MP-mean_x[i])**2 + (MN-mean_x[i])**2)/(SP/(NP-1) + SN/(NN-1))

print(f_score)


index = []

for j in range(0,cols,1):
    if f_score[j] < 0.1:
        index.append(j)

l = len(index)
##print(data)
dataF = np.delete(data,index, axis=1)
print(dataF)
np.savetxt('trainext_feature_selection.txt',dataF,fmt='%0.8f')
##np.savetxt('test_feature_selection.txt',dataF,fmt='%0.8f')