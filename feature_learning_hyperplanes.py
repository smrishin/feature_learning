import sys
import random
import math
#from datetime import datetime
#import os

eta = 0.001
stop = 0.001

###################
#### Supplementary function to compute dot product
###################
def dotproduct(u, v):
    assert len(u) == len(v), "dotproduct: u and v must be of same length"
    dp = 0
    for i in range(0, len(u), 1):
        dp += u[i]*v[i]
    return dp

###################
## Standardize the code here: divide each feature of each 
## datapoint by the length of each column in the training data
## return [traindata, testdata]
###################
def standardize_data(traindata, testdata):
 
    for i in range(0, len(traindata[0]), 1):
        fact = 0
        for j in range(0, len(traindata), 1):
            fact += math.pow(traindata[j][i], 2)
        fact = math.sqrt(fact)
        if fact != 0:
            for a in range(0, len(traindata), 1):
                traindata[a][i] /= fact
            for b in range(0, len(testdata), 1):
                testdata[b][i] /= fact
    return [traindata, testdata]

###################
## Solver for hinge loss
## return [w, w0]
###################
def hinge_loss(traindata, trainlabels):
    w = []
    w0 = []
    for i in range(0, len(traindata[0]), 1):
        w.append(random.uniform(-0.01,0.01)) 
    product = 0
    old_error = len(traindata) * 10
    while True:      
        delta = []
        er = 0
        for j in range(0, len(traindata[0]), 1):
            delta.append(0)
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w, traindata[i])
                check = (trainlabels[i]) * product
                for j in range(0, len(traindata[0]), 1):
                    if(check < 1):
                        q = (trainlabels[i])*(traindata[i][j])
                        delta[j] += -1 * q
        for j in range(0, len(traindata[0]), 1):
            w[j] -= eta * delta[j]                  # w
        for i in range(0, len(traindata), 1):
            if (trainlabels[i] != None):
                product = dotproduct(w,traindata[i])
                p = trainlabels[i] * product
                er += max(0, 1 - p)                 #error/hinge loss
        w0 = [len(w) - 1]                           # w0
        if(abs(old_error - er) <= stop):
            break
        old_error = er
    #print("Min Error",str(er)) 
    return[w,w0]

###################################
#### MAIN ################
#######################
#k = int(sys.argv[3])
k=10
###################
#### Code to read train data and train labels
###################
#trainfile = sys.argv[1]
#testfile = sys.argv[2]
trainfile = "ion.train.0.txt"
testfile = "ion.test.0.txt"
traindata = []
trainlabels = []
testdata = []
testlabels = []

dataset=trainfile.split(".")[0].upper()


f = open(trainfile) 
line = f.readline()
while (line != ''):
    values = line.split()
    y = []
    for val in range(1, len(values)):
        y.append(float(values[val]))
    traindata.append(y)
    trainlabels.append(float(values[0]))
    line = f.readline()

###################
#### Code to test data and test labels
#### The test labels are to be used
#### only for evaluation and nowhere else.
#### When your project is being graded we
#### will use 0 label for all test points
###################
f = open(testfile) 
line = f.readline()
while (line != ''):
    values = line.split()
    y = []
    for val in range(1, len(values)):
        y.append(float(values[val]))
    testdata.append(y)
    testlabels.append(float(values[0]))
    line = f.readline()




#standardization
[traindata, testdata] = standardize_data(traindata, testdata)

zest = []
zrain = []
for i in range(0, len(testdata), 1):
    p = []
    for j in range(k):
        p.append(0)
    zest.append(p)

for i in range(0, len(traindata), 1):
    q = []
    for j in range(k):
        q.append(0)
    zrain.append(q)
    
############# For i = 0 to k do:

for i in range(k):
    z = []
    z_ = []
    w = []
    for q in range(0, len(traindata[0]), 1):
        w.append(random.uniform(-1, 1))

    for j in range(len(traindata)):
        z.append(dotproduct(traindata[j], w))
    w[0] = random.uniform(min(z), max(z))
    z = []
    for j in range(len(traindata)):
        z.append(dotproduct(traindata[j], w))
    for j in range(len(traindata)):
        if (z[j] > 0.0):
            zrain[j][i] = 1
        else:
            zrain[j][i] = 0
    for j in range(len(testdata)):
        z_.append(dotproduct(testdata[j], w))
    for j in range(len(testdata)):
        if (z_[j] > 0.0):
            zest[j][i] = 1
        else:
            zest[j][i] = 0

[zrain, zest] = standardize_data(zrain, zest)

################################
#Hinge Loss predictions for Z and Z'
################################
print("Computing w,w0 : Hinge Loss.......for z")
print("Dataset:" , dataset)
[w,w0] = hinge_loss(zrain, trainlabels)
print("Printing Prediction")
#printOutput(w, w0, testdata, "hinge_predictions.txt")
#=============================================================================
### PRINTING OUTPUT for z dataset TO A FILE
predfile = 'Project2_' + dataset + "_" + str(k) + '.txt'
OUT = open(predfile, 'w')
rows = len(zest)
x=0
for i in range(0, rows, 1):
    product = dotproduct(w, zest[i]);
    if (product < 0):
        print("0", i, file = OUT);
        #print(testlabels[i])
        if testlabels[i] == -1:
            x += 1
    else:
        print("1",i, file = OUT);
        #print(testlabels[i])
        if testlabels[i] == 1:
            x += 1
#print(str(x),str(rows))
print("DONE : Hinge Loss k=",str(k),"accuracy=",str(x*100/rows)+"%")
OUT.close()
#=============================================================================
###########Hinge loss for original dataset
# print("Computing w,w0 : Hinge Loss.......for original")
# print("Dataset:" , dataset)
# [w,w0] = hinge_loss(traindata, trainlabels)
# print("Printing Prediction")
#printOutput(w, w0, testdata, "hinge_predictions.txt")
#=============================================================================
### PRINTING OUTPUT for original dataset TO A FILE
# predfile = 'Project2_' + dataset + "_original"+ '.txt'
# OUT = open(predfile, 'w')
# rows = len(testdata)
# m=0
# for i in range(0, rows, 1):
#     product = dotproduct(w, testdata[i]);
#     if (product < 0):
#         print("0", i, file = OUT);
#         #print(testlabels[i])
#         if testlabels[i] == -1:
#             m += 1
#     else:
#         print("1",i, file = OUT);
#         #print(testlabels[i])
#         if testlabels[i] == 1:
#             m += 1
# print(str(m),str(rows))
# print("DONE : Hinge Loss original data","accuracy=",str(m*100/rows)+"%")
# OUT.close()

