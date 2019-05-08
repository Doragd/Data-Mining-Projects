import numpy as np
import pandas as pd
import random


def kernelTrans(X, A, kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
    
def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = np.abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0: return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)
        if (np.abs(oS.alphas[j] - alphaJold) < 0.00001): return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True  
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas

def norm(df):
    dat = df.iloc[:,1:].copy()
    label = df.iloc[:,0].copy()
    dat /= 255
    return pd.concat([label, dat], axis=1)

def trainSVM(trainData, C=200, toler=0.0001, maxIter=10000, kTup=('rbf', 10)):
    dataArr = trainData[2][:,1:].copy()
    labelArr = list(trainData[2][:,0].copy())
    b,alphas = smoP(dataArr, labelArr, C, toler, maxIter, kTup)
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors" % np.shape(sVs)[0])
    model = {'alphas':alphas, 'b':b, 'svInd':svInd, 'sVs':sVs, 'labelSV':labelSV, 'label_x':trainData[0], 'label_y':trainData[1], 'kTup':kTup}
    return model

def predict(kTup, C):
    models = []
    for i in range(len(trainSet)):
        print("Training Model1: %d" % i)
        models.append(trainSVM(trainSet[i], C=C, kTup=kTup))
    print("Train and Test..")
    datMat = np.array(train_set.iloc[:,1:]); labelMat = np.array(train_set.iloc[:,0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        count = [0]*classNum
        for j in range(len(models)):
            sVs = models[j]['sVs']; kTup = models[j]['kTup']; labelSV = models[j]['labelSV']
            svInd = models[j]['svInd']; alphas = models[j]['alphas']; b = models[j]['b']
            label_x = models[j]['label_x']; label_y = models[j]['label_y']
            kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
            predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
            if(np.sign(predict) == 1):
                count[label_x] += 1
            else:
                count[label_y] += 1
        if(np.argmax(count) != labelMat[i]):
            errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))
    
    datMat = np.array(test_set.iloc[:,1:]); labelMat = np.array(test_set.iloc[:,0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        count = [0]*classNum
        for j in range(len(models)):
            sVs = models[j]['sVs']; kTup = models[j]['kTup']; labelSV = models[j]['labelSV']
            svInd = models[j]['svInd']; alphas = models[j]['alphas']; b = models[j]['b']
            label_x = models[j]['label_x']; label_y = models[j]['label_y']
            kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
            predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
            if(np.sign(predict) == 1):
                count[label_x] += 1
            else:
                count[label_y] += 1
        if(np.argmax(count) != labelMat[i]):
            errorCount += 1
    print ("the testing error rate is: %f" % (float(errorCount)/m))

    
df = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
train_set = pd.read_csv('train.csv')

df = norm(df)
test_set = norm(test_set)
train_set = norm(train_set)

train = []
classNum = len(df['label'].drop_duplicates())
for i in range(classNum):
    train.append(df[df['label'] == i])
trainSet = []
for i in range(classNum-1):
    for j in range(i+1, classNum): 
        temp_i = np.array(train[i]).copy()
        temp_j = np.array(train[j]).copy()
        temp_i[:,0] = 1.0
        temp_j[:,0] = -1.0
        temp = np.concatenate([temp_i,temp_j])
        trainSet.append((i,j, temp))    
    
predict(kTup=('rbf', 10), C=100)