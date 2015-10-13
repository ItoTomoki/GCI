# encoding: utf-8

import sys
import os
import re
import codecs
import csv
import MeCab
import re
import unicodedata
import sys
from gensim.models import word2vec
import numpy as np
#model = word2vec.Word2Vec.load("2013_kai.model")
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from gensim.models import word2vec
import numpy as np
import json
import os
from ast import literal_eval
import re
import sys
import MeCab
from collections import defaultdict
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sklearn.decomposition
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#単語出現回数と、共起単語出現回数をカウント

#tagger = MeCab.Tagger('-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/ruiter-keyword.dic') 
#tfidfTextList = {}
number = 0
textinfo = {}
freqwords = {}
freqpair = {}
max = 0
textlist = {}
freqwords = defaultdict(int)

g = file("textinfo0000-0006.dump","r")
textinfo = pickle.load(g)
g.close()
g = file("thread0000-0006.dump","r")
thread = pickle.load(g)
g.close()

#modelnico = word2vec.Word2Vec.load("allcomment2_kai0000_0006.model")
modelnico = word2vec.Word2Vec.load("allcomment0000-0006_200.model")
model = modelnico

def makekyoukilist(sentvoc):
    freqwords = {}
    freqpair = {}
    for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        print ID
        for video_id in textinfo[ID].keys():
            filename = ("comment_kai" + ID + "/" + str(video_id) + ".txt")
            try:
                f = open(filename)
            except:
                continue
            text = f.read()
            f.close()
            datalines = text.split('\n')
            #print len(datalines)
            for wordlist in datalines:
                wordlist = wordlist.split(" ")
                for w in wordlist:
                        try:
                            freqwords[w] += 1
                        except:
                            freqwords[w] = defaultdict(int)
                            freqwords[w] = 1
                        if w in sentvoc:
                            print w
                            for w2 in wordlist:
                                try:
                                    freqpair[w][w2] += 1
                                except:
                                    freqpair[w] = defaultdict(int)
                                    freqpair[w][w2] = 1
    return freqpair,freqwords


#共起頻度のみからpos/neg を判定
in_file = csv.reader(open('../../../../../Mypython/ruiternews/sentimintdic/sentimentdic/pn.csv',"rb"))
pne = {}
wordsarray = []
errorwordsarray = []
for line in in_file:
    line = line[0].split("\t")
    s =line[1]
    t = line[0].decode("utf-8")
    if s == "p":
        score = 1.0
    elif s == "n":
        score = -1.0
    elif s == "e":
        score = 0.0
    else:
        print s,t 
        errorwordsarray.append(line)
        continue
    pne[t]= score
    wordsarray.append(t)
wordsarrayEncode = []
for w in wordsarray:
    wordsarrayEncode.append(w.encode("utf-8"))
#sentvoc = list(set(wordsarrayEncode) & set(freqwords.keys()))
#sentvoc2 = list(set(wordsarrayEncode) & set(freqpair.keys()))
sentvoc = list(set(wordsarray) & set(model.vocab.keys()))
sentvoeEncode = []
for w in sentvoc:
    sentvoeEncode.append(w.encode("utf-8"))

freqpair,freqwords = makekyoukilist(set(sentvoeEncode))
N = 0.0
for k in freqwords.keys():
    N += freqwords[k]
inputword = sentvoc[0]
inputwordlist = sentvoc
def sslvalue(inputword,inputwordlist,N = N):
    ssl = 0.0
    for k in inputwordlist:
        ssl += (((freqpair[k][inputword] - float(freqwords[inputword] * freqwords[k])/float(N)) ** 2 )/(float(freqwords[inputword] * freqwords[k])/float(N)))
    ssl = ssl/len(inputwordlist)
    return ssl

def PredictPosOrNeg(inputword,worddatalist,N = N):
    poswordslist = []
    negwordslist  = []
    neuwordslist = []
    for w in worddatalist:
        if pne[w.decode('utf-8')] == 1.0:
            poswordslist.append(w)
        elif  pne[w.decode('utf-8')] == -1.0:
            negwordslist.append(w)
    PosSSL = sslvalue(inputword,poswordslist,N)
    NegSSL = sslvalue(inputword,negwordslist,N)
    #print PosSSL - NegSSL
    if PosSSL - NegSSL > 200:
        return 1.0
    elif NegSSL - PosSSL > 200:
        return -1.0
    else:
        return 0.0

worddatalist = sentvoeEncode
NewNeuwordlist = []
NewPoswordlist = []
NewNegwordlist = []
for k in model.vocab.keys():
    posnegvalue = PredictPosOrNeg(k.encode("utf-8"),worddatalist,N = N)
    if posnegvalue == 0.0:
        NewNeuwordlist.append(k)
    elif posnegvalue == 1.0:
        NewPoswordlist.append(k)
    else:
        NewNegwordlist.append(k)

point1 = 0
point2 = 0
point3 = 0
for k in NewPoswordlist:
    try:
        print pne[k],k
        if pne[k] == 1.0:
            point1 += 1.0
        elif pne[k] == 0.0:
            point2 += 1.0
        else:
            point3 += 1.0
        point += pne[k]
    except:
        continue
#K-means
from k_means_2 import KMeans
poswordslist = []
negwordslist  = []
neuwordslist = []
for w in worddatalist:
    if pne[w.decode("utf-8")] == 1.0:
        poswordslist.append(w)
    elif  pne[w.decode("utf-8")] == -1.0:
        negwordslist.append(w)
    else:
        neuwordslist.append(w)


posveclist = {}
for k in poswordslist:
    posveclist[k] = np.array(model[k.decode("utf-8")]/np.linalg.norm(model[k.decode("utf-8")]))
M = np.array(posveclist.values())
kmeans = KMeans(n_clusters=30, random_state=10)
kmeans_model = kmeans.fit(M)
labels = kmeans_model.labels_
posd = zip(labels, features)
posword2vecdic = dict(zip(poswordslist, labels))
poscentroids = kmeans_model.cluster_centers_

negveclist = {}
for k in negwordslist:
    negveclist[k] = np.array(model[k.decode("utf-8")]/np.linalg.norm(model[k.decode("utf-8")]))
M = np.array(negveclist.values())
kmeans = KMeans(n_clusters=30, random_state=10)
kmeans_model = kmeans.fit(M)
labels = kmeans_model.labels_
negd = zip(labels, M)
negword2vecdic = dict(zip(negwordslist, labels))
negcentroids = kmeans_model.cluster_centers_

neueclist = {}
for k in negwordslist:
    neueclist[k] = np.array(model[k.decode("utf-8")]/np.linalg.norm(model[k.decode("utf-8")]))
M = np.array(neueclist.values())
kmeans = KMeans(n_clusters=10, random_state=10)
kmeans_model = kmeans.fit(M)
labels = kmeans_model.labels_
neud = zip(labels, M)
neuword2vecdic = dict(zip(neuwordslist, labels))
neucentroids = kmeans_model.cluster_centers_

#knnでpos/neg/neutralに分類
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
neigh = KNeighborsClassifier(n_neighbors=1, metric= cosine_similarity)
X = (list(poscentroids) + list(negcentroids) + list(neucentroids))
y = []
for i in range(30):
    y.append(1)

for i in range(30):
    y.append(-1)

for i in range(30):
    y.append(0)

neigh.fit(X, y) 
Newposvoclist = []
Newneuvoclist = []
Newnegvoclist = []
X_test = [model[model.vocab.keys()[i]] for i in range(len(model.vocab.keys()))]
Y_pred = neigh.predict(X_test) 
X_test = np.array(X_test)
voclist = model.vocab.keys()
voclist = np.array(voclist)
Newposvoclist = voclist[Y_pred == 1.0]
Newneuvoclist = voclist[Y_pred == 0.0]
Newnegvoclist = voclist[Y_pred == -1.0]
Newposveclist = X_test[Y_pred == 1.0]
Newneuveclist = X_test[Y_pred == 0.0]
Newnegveclist = X_test[Y_pred == -1.0]
word2vecdic = {}
n = 0
for eachvoclist, veclist in [(Newposvoclist,Newposveclist), (Newneuvoclist,Newneuveclist), (Newnegvoclist,Newnegveclist)]:
    kmeans = KMeans(n_clusters=500, random_state=10)
    M = veclist
    kmeans_model = kmeans.fit(M)
    labels = (kmeans_model.labels_ + n * 500)
    d = zip(labels, M)
    #word2vecdic1 = zip(eachvoclist, labels)
    word2vecdic.update(dict(zip(eachvoclist, labels)))
    n += 1


"""
def PredictPosOrNegWithWord2vec(inputword,worddatalist,N = N):
    poswordslist = []
    negwordslist  = []
    neuwordslist = []
    for w in worddatalist:
        if pne[w.decode("utf-8")] == 1.0:
            poswordslist.append(w)
        elif  pne[w.decode("utf-8")] == -1.0:
            negwordslist.append(w)
    PosSSL = sslvalue(inputword,poswordslist,N)
    NegSSL = sslvalue(inputword,negwordslist,N)
    #print PosSSL - NegSSL
    if PosSSL - NegSSL > 200:
        for w in (poswordslist + negwordslist):
            try:
                v = model.similarity(inputword.decode("utf-8"),w.decode("utf-8"))
                if v > 0.4:
                    #print v
                    return 1.0
                    break
            except:
                continue
        #print v
        return 0.0
    elif NegSSL - PosSSL > 200:
        for w in (poswordslist + negwordslist):
            try:
                v = model.similarity(inputword.decode("utf-8"),w.decode("utf-8"))
                if v > 0.4:
                    #print v
                    return -1.0
                    break
            except:
                continue
        #print v
        return 0.0
    else:
        return 0.0

worddatalist = sentvoeEncode
NewNeuwordlist2 = []
NewPoswordlist2 = []
NewNegwordlist2 = []
for k in model.vocab.keys():
    posnegvalue = PredictPosOrNegWithWord2vec(k.encode("utf-8"),worddatalist,N = N)
    if posnegvalue == 0.0:
        NewNeuwordlist2.append(k)
    elif posnegvalue == 1.0:
        NewPoswordlist2.append(k)
    else:
        NewNegwordlist2.append(k)

point1 = 0
point2 = 0
point3 = 0
for k in NewPoswordlist2:
    try:
        print pne[k],k
        if pne[k] == 1.0:
            point1 += 1.0
        elif pne[k] == 0.0:
            point2 += 1.0
        else:
            point3 += 1.0
        point += pne[k]
    except:
        continue



def PredictPosOrNegWithWord2veckai(inputwordlist,worddatalist,N = N,thread = 0.4):
    poswordslist = []
    negwordslist  = []
    neuwordslist = []
    for w in worddatalist:
        if pne[w.decode("utf-8")] == 1.0:
            poswordslist.append(w)
        elif  pne[w.decode("utf-8")] == -1.0:
            negwordslist.append(w)
    newwordslist = []
    Newdict = {}
    Newdictkyu = {}
    for w in inputwordlist:
        predictvalue = PredictPosOrNeg(w,worddatalist,N = N)
        Newdict[w] = predictvalue
        Newdictkyu[w] = predictvalue
        if Newdict[w] != 0.0:
            newwordslist.append(w)
    for w2 in newwordslist:
        for w1 in (newwordslist + poswordslist + negwordslist):
            try:
                v = model.similarity(w2.decode("utf-8"),w1.decode("utf-8"))
            except:
                v = 0.0
            if (w2 != w1) & (v > thread):
                break
        if v <= thread:
            print v
            Newdict[w2] = 0.0
    return Newdictkyu,Newdict

from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report
cv = KFold(n=len(sentvoc), n_folds=10, indices=True)
score = []
score2 = []
predictscores = []
predictscores2 = []
accuratescores = []
Newpne = {}
Newpne2 = {}
for train,test in cv:
    cosentvoc_train = np.array(sentvoc)[train]
    cosentvoc_test = np.array(sentvoc)[test]
    PredictRusult = PredictPosOrNegWithWord2veckai(cosentvoc_test,cosentvoc_train,N = N,thread  = 0.6)
    Newpne = PredictRusult[0]
    Newpne2 = PredictRusult[1]
    for m in cosentvoc_test:
        if Newpne2[m] == pne[m.decode("utf-8")]:
            score2.append(1.0)
        else:
            score2.append(0.0)
        if Newpne[m] == pne[m.decode("utf-8")]:
            score.append(1.0)
        else:
            score.append(0.0)
        predictscores.append(Newpne[m])
        predictscores2.append(Newpne2[m])
        accuratescores.append(pne[m.decode("utf-8")])
print(classification_report(accuratescores, predictscores))
print(classification_report(accuratescores, predictscores2))
print np.mean(score), len(score)
print np.mean(score2), len(score2)
"""
for k in preprocessed_docs.keys():
    docvec[k] = np.zeros(1500)
    sentence = preprocessed_docs[k]
    for m in sentence:
        try:
            labelnum = word2vecdic[m]
            docvec[k][labelnum] = (docvec[k][labelnum] + 1)
        except:
            continue

for keys in docvec.keys():
    docvec[k] = np.array(docvec[k]/np.linalg.norm(docvec[k]))

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import neighbors,svm,linear_model
from sklearn.metrics import classification_report

def PredictAndAnalyze2(data,target,clf_cv = svm.SVC(kernel='linear', probability=True),checkauc = False,ifprint = False,balancing = True):
    aucs = []
    y_trueall = []
    y_pridictall = []
    length = min([len(target[target == 0]),len(target[target == 1]),len(target[target == 2])])
    data = np.r_[data[target == 0][0:length],data[target == 1][0:length],data[target == 2][0:length]]
    target = np.r_[target[target == 0][0:length],target[target == 1][0:length],target[target == 2][0:length]]
    """
    length = min([len(target[target == 0]),len(target[target == 1])])
    data = np.r_[data[target == 0][0:length],data[target == 1][0:length]]
    target = np.r_[target[target == 0][0:length],target[target == 1][0:length]]
    """
    kf = KFold(len(target), n_folds=10, shuffle=True)
    vmats = np.array([])
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(target)[train]
        X_test, y_test = np.array(data)[val], np.array(target)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)
        vmat = clf_cv.coef_[0]
        if vmats.shape[0] == 0:
            vmats = vmat
        else:
            vmats = np.c_[vmats,vmat]
        y_true = y_test
        y_trueall = y_trueall + list(y_true)
        y_pridictall = y_pridictall  + list(y_pred)
        if ifprint == True:
            print(classification_report(y_true, y_pred))
        if checkauc == True:
            y_pred_cv = clf_cv.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_cv)
            aucs.append(auc)
    if checkauc == True:
        print np.mean(aucs), np.std(aucs)
    print(classification_report(y_trueall, y_pridictall))
    return y_trueall, y_pridictall,vmats

def PredictAndAnalyze3(data,target,clf_cv = svm.SVC(kernel='linear', probability=True),checkauc = False,ifprint = False,balancing = True):
    aucs = []
    y_trueall = []
    y_pridictall = []
    length = min([len(target[target == 0]),len(target[target == 1]),len(target[target == 2])])
    data = np.r_[data[target == 0][0:length],data[target == 1][0:length],data[target == 2][0:length]]
    target = np.r_[target[target == 0][0:length],target[target == 1][0:length],target[target == 2][0:length]]
    """
    length = min([len(target[target == 0]),len(target[target == 1])])
    data = np.r_[data[target == 0][0:length],data[target == 1][0:length]]
    target = np.r_[target[target == 0][0:length],target[target == 1][0:length]]
    """
    kf = KFold(len(target), n_folds=10, shuffle=True)
    vmats = np.array([])
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(target)[train]
        X_test, y_test = np.array(data)[val], np.array(target)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)
        """
        vmat = clf_cv.coef_[0]
        if vmats.shape[0] == 0:
            vmats = vmat
        else:
            vmats = np.c_[vmats,vmat]
        """
        y_true = y_test
        y_trueall = y_trueall + list(y_true)
        y_pridictall = y_pridictall  + list(y_pred)
        if ifprint == True:
            print(classification_report(y_true, y_pred))
        if checkauc == True:
            y_pred_cv = clf_cv.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_cv)
            aucs.append(auc)
    if checkauc == True:
        print np.mean(aucs), np.std(aucs)
    print(classification_report(y_trueall, y_pridictall))
    return y_trueall, y_pridictall#,vmats

commentarray = {}
for ID in thread.keys():
    commentarray[ID] = {}
    for video_id in thread[ID].keys():
        commentarray[video_id[0:-4]] = thread[ID][str(video_id)]["comment_counter"]

viewarray = {}
for ID in thread.keys():
    viewarray[ID] = {}
    for video_id in thread[ID].keys():
        viewarray[video_id[0:-4]] = thread[ID][str(video_id)]["view_counter"]

def createtargetarray(mincomment,maxcomment,threadviewcount1,threadviewcount2,docvec):
    target2 = []
    for k in docvec.keys():
        if (commentarray[k] > mincomment) & (commentarray[k] < maxcomment):
            if viewarray[k] > threadviewcount2:
                target2.append(0)
            elif viewarray[k] > threadviewcount1:
                target2.append(1)
            else:
                target2.append(2)
    return np.array(target2)

def createtargetarray2(mincomment,maxcomment,threadviewcount1,threadviewcount2,docvec):
    target2 = []
    for k in docvec.keys():
        if (commentarray[k] > mincomment) & (commentarray[k] < maxcomment):
            if commentarray[k] > threadviewcount2:
                target2.append(0)
            elif commentarray[k] > threadviewcount1:
                target2.append(1)
            else:
                target2.append(2)
    return np.array(target2)

def createtvectorMat2(mincomment,maxcomment,docvec):
    vectorMat2 = docvec.values()
    vectorMat2 = np.array(vectorMat2)
    arrays = [] 
    for name in docvec.keys():
            if (commentarray[name] > mincomment) & (commentarray[name] < maxcomment):
                arrays.append(True)
            else:
                arrays.append(False)
    arrays = np.array(arrays)           
    vectorMat2 = vectorMat2[arrays]
    return (vectorMat2)

target2 = createtargetarray(500,10000000,10760.0,34544,docvec)
target2 = createtargetarray2(500,10000000,1000,5000,docvec)
data2 = createtvectorMat2(500,10000000,docvec)
k = PredictAndAnalyze2(data = data2,target = target2,clf_cv = svm.LinearSVC())

M = k[2].T
meanvec = np.zeros(1500)
for j in range(10):
    meanvec = meanvec + M[j]
meanvec = meanvec/10.0
array = {}
for j in range(1500):
    array[j] = abs(meanvec[j])
rankarray = sorted(array.items(),key = lambda x:x[1],reverse = True)
for v in rankarray[0:30]:
    text = u""
    print v[0],v[1],meanvec[v[0]]
    for key in word2vecdic.keys():
        if word2vecdic[key] == v[0]:
            text += key
    print text



#回帰
kf = KFold(len(target), n_folds=10, shuffle=True)
    vmats = np.array([])
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(target)[train]
        X_test, y_test = np.array(data)[val], np.array(target)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)

from sklearn import svm
from sklearn import cross_validation

commentountlist = []
for k in docvec.keys():
    viewcountlist.append(commentarray[k])
#データを6割をトレーニング、4割をテスト用とする
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data2, np.array(viewcountlist), test_size=0.2)
reg = svm.SVR(kernel='rbf', C=10).fit(x_train, y_train)
reg.fit(x_train, y_train)
svrreg = svm.LinearSVR().fit(x_train, y_train)

M = k2.T
meanvec = np.zeros(1500)
for j in range(10):
    meanvec = meanvec + M[j]
meanvec = meanvec/10.0
array = {}
for j in range(1500):
    array[j] = abs(k2[j])
rankarray = sorted(array.items(),key = lambda x:x[1],reverse = True)
for v in rankarray[0:30]:
    text = u""
    print v[0],v[1],k2[v[0]]
    for key in word2vecdic.keys():
        if word2vecdic[key] == v[0]:
            text += key
    print text
array = 0
if 


