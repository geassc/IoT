import sys
###########################
###read predicted labels###
###########################
predictedlabelfile = sys.argv[1]
f=open(predictedlabelfile,'r')
predicted_labels={}
l=f.readline()
while(l!=''):
    a = l.split()
    predicted_labels[int(a[1])] = int(a[0])
    l = f.readline()
print(predicted_labels)
###print('\n')
rows = len(predicted_labels)
print(rows)
#########################
###correct labels###
#########################
correct_labels={}
for i in range(0,rows,1):
    if i < rows/2:
        correct_labels[i] = -1
    else:
        correct_labels[i] = 1
###########################
###compute balance error###
###########################
a = 0
b = 0
c = 0
d = 0
for i in range(0,rows,1):
    if predicted_labels[i] == -1:
        if(correct_labels[i] == -1):
            a +=1
        else:
            c +=1
    if predicted_labels[i] == 1:
        if(correct_labels[i] == -1):
            b +=1
        else:
            d +=1
ber = ((b/(a+b))+(c/(c+d)))/2
print("a = ",a)
print("b = ",b)
print("c = ",c)
print("d = ",d)
print("balance error rate is ", ber)