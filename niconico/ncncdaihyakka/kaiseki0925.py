#encoding:utf-8
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

#modelwiki = word2vec.Word2Vec.load("wiki1-4.model")
#data = word2vec.Text8Corpus('allcomment.txt')
#data = word2vec.Text8Corpus('allcomment2kaiseiki.txt')
#modelnico = word2vec.Word2Vec(data, size=50)
#modelnico = word2vec.Word2Vec.load("allcomment2.model")
#modelnico = word2vec.Word2Vec.load("allcomment1.model")
#modelnico = word2vec.Word2Vec.load("allcomment2kaiseiki.model") #コメントをスペースでつなぐ改行正規化済み
#modelnico = word2vec.Word2Vec.load("allcomment2.model")　#コメントをスペースでつなぐ改行正規化なし
#modelnico = word2vec.Word2Vec.load("allcomment_kai.model") #コメントをひとつずつ改行正規化済み

#modelnico = word2vec.Word2Vec.load("models/allcomment2kaiseiki.model")
#modelnico = word2vec.Word2Vec.load("allcomment_2kai.model")
#modelnico = word2vec.Word2Vec.load("models/allcomment2.model")
#modelnico = word2vec.Word2Vec.load("models/allcomment2bigram_kai.model")
#modelnico = word2vec.Word2Vec.load("models/allcomment2bigram.model")
#modelnico = word2vec.Word2Vec.load("models/allcomment_kai.model")
#modelnico = word2vec.Word2Vec.load("allcomment_2kai.model")
modelnico = word2vec.Word2Vec.load("allcomment2_kai0000_0006.model")
model = modelnico
"""
word = [u"腹筋"]
out2 = modelnico.most_similar(positive= word)
wordarray = []
for j in out2:
    wordarray.append(j[0])
    print j[0]
"""

#レコメンド絡み
def wordvec(word,model = modelnico):
    try:
        v = model[word]/np.linalg.norm(model[word])
        return v
    except:
        return np.zeros(len(model[model.vocab.keys()[0]]))


def morphological_analysis(text):
    word2freq = defaultdict(int)
    mecab = MeCab.Tagger('-u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/ncnc.dic')
    node = mecab.parseToNode(text)
    while node:
        #if node.feature.split(",")[0] == "名詞":
        word2freq[node.surface] += 1
        node = node.next
    return word2freq

def output(word2freq):
    for word, freq in sorted(word2freq.items(),key = lambda x: x[1], reverse=True):
        print str(freq), word
        
def makevec(word2freq):
    freqcount = 0
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    for word, freq in sorted(word2freq.items(),key = lambda x: x[1], reverse=True):
        if int(freq) > 1:
            v += freq * wordvec(word.decode("utf-8"))
            freqcount += freq
    if (v == np.zeros(len(model[model.vocab.keys()[0]]))).all():
        return np.zeros(len(model[model.vocab.keys()[0]]))
    else:
        return (v/np.linalg.norm(v))

def createvector(video_id,ID="0000",mincount = 0,maxcount = -1):
    if video_id == "sm9":
        return np.zeros(len(model[model.vocab.keys()[0]]))
    else:
        #filename = ("comment2_" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment2_kai" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment2_bigram" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment2bigram_kai" + ID + "/" + str(video_id) + ".txt")
        filename = ("comment_kai" + ID + "/" + str(video_id) + ".txt")
        f = open(filename)
        data = f.read()
        f.close()
        datalines = data.split('\n')
        data = ""
        for n in range(mincount,maxcount):
            try:
                data += (datalines[n] + " ")
            except:
                #print len(datalines),ID,video_id
                break
        v = makevec(morphological_analysis(data))
        return v

def createvector2(video_id,ID="0000",count = 100):
    if video_id == "sm9":
        return np.zeros(len(model[model.vocab.keys()[0]]))
    else:
        #filename = ("comment2_" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment2_kai" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment2_bigram" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment2bigram_kai" + ID + "/" + str(video_id) + ".txt")
        filename = ("comment_kai" + ID + "/" + str(video_id) + ".txt")
        f = open(filename)
        data = f.read()
        f.close()
        datalines = data.split('\n')
        data = ""
        counts = (len(datalines)/count)
        for n in range(0,count):
            try:
                data += (datalines[(n * counts)] + " ")
            except:
                #print len(datalines),ID,video_id
                break
        v = makevec(morphological_analysis(data))
        return v

vectorinfo = {}
"""
filepass = ('../data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/')
textinfo = {}
thread = {}
count = 0
#for file in files[1:2]:
for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    #print file
    filename = ID + ".dat"
    filepass = '../data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/video/' + str(filename)
    f = open(filepass)
    lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
    data1 = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()
    Lines2 = {}
    count = 0
    textinfo[ID] = {}
    thread[ID] = {}
    for line in lines2:
        try:
            Lines2[count] = literal_eval(line)
        except:
            line = line.replace('null', '"null"')
            Lines2[count] = literal_eval(line)
        thread[ID][(Lines2[count]["video_id"] + ".dat")] = Lines2[count]
        #thread["0000"][(Lines2[count]["video_id"] + ".dat")]["title"] = Lines2[count]["title"].decode('unicode_escape')
        textinfo[ID][Lines2[count]["video_id"]] = Lines2[count]["title"].decode('unicode_escape')
        count += 1
f = file("textinfo0000-0006.dump","w")
pickle.dump(textinfo,f)
f.close()
f = file("thread0000-0006.dump","w")
pickle.dump(thread,f)
f.close()
"""
g = file("textinfo0000-0006.dump","r")
textinfo = pickle.load(g)
g.close()
g = file("thread0000-0006.dump","r")
thread = pickle.load(g)
g.close()

for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    vectorinfo[ID] = {}
    for j in textinfo[ID].keys():
        #print j
        try:
            vectorinfo[ID][j] = createvector(video_id = j, ID = ID,mincount = 500,maxcount = 10000000)
            #vectorinfo[ID][j] = createvector(video_id = j, ID = ID,count = 500)
        except:
            #vectorinfo[ID][j] = np.zeros(len(model[model.vocab.keys()[0]]))
            print ID,j
"""
g = file("comment2_Data/vectorinfo.dump","r")
g = file("comment2_bigram_Data/vectorinfo.dump","r")
g = file("comment2_kai_Data/vectorinfo.dump","r")
g = file("comment2bigram_kai_Data/vectorinfo.dump","r")
g = file("comment_kai_Data/vectorinfo.dump","r")
vectorinfo = pickle.load(g)
g.close()
"""
l = {}
for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    l[ID] = vectorinfo[ID].keys()


for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    for j in range(0,len(l[ID])):
        if (ID == "0000") & (j == 0):
            vectorMat = vectorinfo["0000"][l["0000"][0]]
        else:
            vectorMat = np.c_[vectorMat,vectorinfo[ID][l[ID][j]]]

data = vectorMat.T

#機会学習
#34544は平均値で10760.0は中央値（再生回数）
from sklearn import neighbors,svm,linear_model
from sklearn.metrics import classification_report
target = []
for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    for j in range(0,len(l[ID])):
        if thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > 34544:#10760.0:#34544:
            target.append(0)
        elif thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > 10760.0:
            target.append(1)
        else:
            target.append(2)

target = np.array(target)



#コメント数で制限
def createtargetarray(mincomment,maxcomment,threadviewcount1,threadviewcount2):
    target2 = []
    for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        for j in range(0,len(l[ID])):
            if (thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] > mincomment) & (thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] < maxcomment):
                if thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > threadviewcount2:
                    target2.append(0)
                elif thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > threadviewcount1:
                    target2.append(1)
                else:
                    target2.append(2)
    return np.array(target2)

def createtargeMat(target2):
    targetMat = np.zeros((len(target2),3))
    for n in range(len(target2)):
        targetMat[n][target2[n]] = 1.0
    return np.array(targetMat)
targetMat = createtargeMat(target2)
#２値分類
def createtargetarray2(mincomment,maxcomment,threadviewcount):
    target2 = []
    for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        for j in range(0,len(l[ID])):
            if (thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] > mincomment) & (thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] < maxcomment):
                if thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > threadviewcount:
                    target2.append(0)
                else:
                    target2.append(1)
    return np.array(target2)
#コメント数で分類
def createtargetarraycomment(mincomment,threadviewcount):
    target2 = []
    for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        for j in range(0,len(l[ID])):
            if thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] > mincomment:
                if thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] > threadviewcount:
                    target2.append(0) 
                else:
                    target2.append(1)
    return np.array(target2)
def createtvectorMat(mincomment,maxcomment,vectorinfo):
    vectorMat2 = np.array([])
    for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        for j in range(0,len(l[ID])):
            if (thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] > mincomment) & (thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] < maxcomment):
                if vectorMat2.shape[0] == 0:
                    vectorMat2 = vectorinfo[ID][l[ID][j]]
                else:
                    vectorMat2 = np.c_[vectorMat2,vectorinfo[ID][l[ID][j]]]
    return(vectorMat2.T)

#10760.0と34544
target2 = createtargetarraycomment(500,5000)
target2 = createtargetarray(500,10000000,10760.0,34544)
data2 = createtvectorMat(500,100000000,vectorinfo)
y = commentarray.values()
X = 

scores = []
knn = neighbors.KNeighborsClassifier(n_neighbors=10) #metric='manhattan'
classifier = svm.SVC(kernel='linear', probability=True)#,class_weight={0:3,2:1})
logreg = linear_model.LogisticRegression(C=1e1)
"""
for j in range(0,10):
    perm = np.random.permutation(len(target))
    target2 = target[perm]
    data2 = data[perm]
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data2, target2, test_size=0.2)
    knn.fit(x_train,y_train)
    label_predict = knn.predict(x_test)
    y_pred = label_predict
    y_true = y_test
    print(classification_report(y_true, y_pred))#, target_names=target_names))
for j in range(0,10):
    perm = np.random.permutation(len(target))
    target2 = target[perm]
    data2 = data[perm]
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data2, target2, test_size=0.2)
    classifier.fit(x_train,y_train)
    label_predict = knn.predict(x_test)
    y_pred = label_predict
    y_true = y_test
    print(classification_report(y_true, y_pred))
"""
#交差検定
from sklearn import cross_validation
from sklearn.cross_validation import KFold
scores = cross_validation.cross_val_score(knn, data2, target2, cv=5)
print np.mean(scores)
scores = cross_validation.cross_val_score(classifier, data2, target2, cv=5)
print np.mean(scores)
scores = cross_validation.cross_val_score(logreg, data2, target2, cv=5)
print np.mean(scores)

def PredictAndAnalyze(data = data,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True),checkauc = False,ifprint = False,balancing = True):
    kf = KFold(len(target), n_folds=10, shuffle=True)
    aucs = []
    y_trueall = []
    y_pridictall = []
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(target)[train]
        if balancing == True:
            length = min([len(y_train[y_train == 0]),len(y_train[y_train == 1]),len(y_train[y_train == 2])])
            X_train = np.r_[X_train[y_train == 0][0:length],X_train[y_train == 1][0:length],X_train[y_train == 2][0:length]]
            y_train = np.r_[y_train[y_train == 0][0:length],y_train[y_train == 1][0:length],y_train[y_train == 2][0:length]]
        X_test, y_test = np.array(data)[val], np.array(target)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)
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
    return y_trueall, y_pridictall

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
    #vmats = np.array([])
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(target)[train]
        X_test, y_test = np.array(data)[val], np.array(target)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)
        #vmat = clf_cv.coef_[0]
        """
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

def PredictAndAnalyze4(data,targetMat,clf_cv = svm.SVC(kernel='linear', probability=True),checkauc = False,ifprint = False,balancing = True):
    aucs = []
    y_trueall = []
    y_pridictall = []
    #length = min([len(targetMat[targetMat == 0]),len(targetMat[targetMat == 1]),len(targetMat[targetMat == 2])])
    #data = np.r_[data[targetMat == 0][0:length],data[targetMat == 1][0:length],data[targetMat == 2][0:length]]
    #targetMat = np.r_[targetMat[targetMat == [1,0,0]][0:length],targetMat[targetMat == 1][0:length],targetMat[targetMat == 2][0:length]]
    """
    length = min([len(targetMat[targetMat == 0]),len(targetMat[targetMat == 1])])
    data = np.r_[data[targetMat == 0][0:length],data[targetMat == 1][0:length]]
    targetMat = np.r_[targetMat[targetMat == 0][0:length],targetMat[targetMat == 1][0:length]]
    """
    kf = KFold(len(targetMat), n_folds=10, shuffle=True)
    #vmats = np.array([])
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(targetMat)[train]
        X_test, y_test = np.array(data)[val], np.array(targetMat)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)
        #vmat = clf_cv.coef_[0]
        """
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

data2 = createtvectorMat(500,100000000,vectorinfo2)
target = createtargetarray(500,10000000,10760.0,34544)
target2 = createtargetarray2(500,100000000,34544)
target2 = createtargetarraycomment(0,2170)
targetMat = createtargeMat(target2)
k0 = PredictAndAnalyze2(data = data2,target = target2,clf_cv = svm.SVC(kernel='rbf', probability=True))#,class_weight={0:4,1:1}))
#k1 = PredictAndAnalyze2(data2,target2,clf_cv = neighbors.KNeighborsClassifier(n_neighbors=10))
k2 = PredictAndAnalyze2(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1))
k = PredictAndAnalyze4(data = data2,target = targetMat,clf_cv = clf_cv)


#vector作成
vectorinfo2 = {}
for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        vectorinfo2[ID] = {}
        for j in textinfo[ID].keys():
            #print j
            try:
                vectorinfo2[ID][j] = createvector(video_id = j, ID = ID,mincount = -500,maxcount = 0)
            except:
                #vectorinfo[ID][j] = np.zeros(len(model[model.vocab.keys()[0]]))
                print ID,j 
data2 = createtvectorMat(500,100000000,vectorinfo2)
target2 = createtargetarray(500,10000000,10760.0,34544)
target2 = createtargetarray2(500,10000000,34544)
target2 = createtargetarraycomment(100,2170)

k0 = PredictAndAnalyze2(data2,target2,clf_cv = svm.SVC(kernel='linear', probability=True))
k2 = PredictAndAnalyze2(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1))
#一貫性があるかチェック
M = k2[2].T
M1 = k0[2].T
チョー░░
for n1 in range(0,10):
    for n2 in range(n1,10):
        print n1,n2,np.dot((M[n1]/np.linalg.norm(M[n1])),(M[n2]/np.linalg.norm(M[n2])))
meanvec = np.zeros(len(model[model.vocab.keys()[0]]))
for n in range(0,10):
    meanvec = meanvec + (M[n]/np.linalg.norm(M[n]))
meanvec = meanvec/np.linalg.norm(meanvec)

meanvec1 = np.zeros(len(model[model.vocab.keys()[0]]))
for n in range(0,10):
    meanvec1 = meanvec1 + (M1[n]/np.linalg.norm(M1[n]))
meanvec1 = meanvec1/np.linalg.norm(meanvec1)
#
#重みの大きい次元の抽出
featurearray = {}
for n in range(0,20):
    featurearray[n] = abs(meanvec[n])
featurearray1 = []
for n in range(0,10):
    if meanvec1[n] > 0.1:
        featurearray1.append(n)
voclist = model.vocab.keys()

vecinfo = sorted(featurearray.items(), key=lambda x: x[1],reverse = True)
for num in range(0,10):
    n = vecinfo[num]
    wordrank = {}
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    v[n[0]] = 1.0
    for k in range(0,len(voclist)):
        vector = (wordvec(voclist[k],model))
        if (np.dot(v, vector) > 0.1) | (np.dot(v, vector) < -0.1):
            wordrank[(voclist[k])] = np.dot(v, vector)
    k = sorted(wordrank.items(), key=lambda x: x[1],reverse = True)
    print "==========",n[0],meanvec[n[0]],"=========="
    for m in k[0:10]:
        print m[0],m[1]
    k = sorted(wordrank.items(), key=lambda x: x[1],reverse = False)
    print "==========",n[0],meanvec[n[0]],"=========="
    for m in k[0:10]:
        print m[0],m[1]
"""
vecinfo = sorted(featurearray.items(), key=lambda x: x[1],reverse = True)
for num in sorted(range(-10,0),reverse = True):
    n = vecinfo[num]
    wordrank = {}
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    v[n[0]] = 1.0
    for k in range(0,len(voclist)):
        vector = (wordvec(voclist[k],model))
        if (np.dot(v, vector) > 0.1) | (np.dot(v, vector) < -0.1):
            wordrank[(voclist[k])] = np.dot(v, vector)
    k = sorted(wordrank.items(), key=lambda x: x[1],reverse = True)
    print "==========",n,  "=========="
    for m in k[0:20]:
        print m[0],m[1]
    k = sorted(wordrank.items(), key=lambda x: x[1],reverse = False)
    print "==========",n,"=========="
    for m in k[0:20]:
        print m[0],m[1]
"""

vecinfo1 = sorted(featurearray1.items(), key=lambda x: x[1],reverse = True)
for n in vecinfo:
    wordrank = {}
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    v[n[0]] = 1.0
    for k in range(0,len(voclist)):
        vector = (wordvec(voclist[k],model))
        if np.dot(v, vector) > 0.1:
            wordrank[(voclist[k])] = np.dot(v, vector)
    k = sorted(wordrank.items(), key=lambda x: x[1],reverse = True)
    print "==========",n, meanvec[n[0]], "=========="
    for m in k[0:10]:
        print m[0],m[1]

wordrank = {}
v = np.zeros(len(model[model.vocab.keys()[0]]))
v[2] = 1.0
for k in range(0,len(voclist)):
    vector = (wordvec(voclist[k],model))
    if np.dot(v, vector) > 0.2:
        wordrank[(voclist[k])] = np.dot(v, vector)

voclist = modelnico.vocab.keys()
wordrank = {}
for k in range(0,len(voclist)):
    vector = (wordvec(voclist[k],model))
    if (np.dot(meanvec, vector) > 0.1) | (np.dot(meanvec, vector) < -0.1):
        wordrank[(voclist[k])] = np.dot(meanvec, vector)


"""
for k in range(0,len(voclist)):
    for j in range(k,len(voclist)):
        vector = (wordvec(voclist[k],model) + wordvec(voclist[j],model))
        vector = (vector/np.linalg.norm(vector))
        if np.dot(meanvec, vector) > 0.3:
            print (voclist[k] + " " + voclist[j])
            wordrank[(voclist[k] + " " + voclist[j])] = np.dot(meanvec, vector)
"""
k = sorted(wordrank.items(), key=lambda x: x[1],reverse = True)
for m in k[0:20]:
    print m[0],m[1]
wordrank2 = {}
voclist2 = wordrank.keys()
for k in range(0,len(voclist2 = wordrank.keys()):
    for j in range(k,len(voclist2 = wordrank.keys()):
        vector = (wordvec(voclist2[k],model) + wordvec(voclist2[j],model))
        vector = (vector/np.linalg.norm(vector))
        if np.dot(meanvec, vector) > 0.3:
            print (k + " " + j)
            wordrank2[(k + " " + j)] = np.dot(meanvec, vector)
k = sorted(wordrank2.items(), key=lambda x: x[1],reverse = True)
for m in k[0:100]:
    print m[0],m[1]
#tf-idf
def makewordlist(ID,video_id,mincommentlines,maxcommentlines):
    #filename = ("comment2_" + ID + "/" + str(video_id) + ".txt")
    #filename = ("comment2_kai" + ID + "/" + str(video_id) + ".txt")
    filename = ("comment_kai" + ID + "/" + str(video_id) + ".txt")
    f = open(filename)
    text = f.read()
    f.close()
    datalines = text.split('\n')
    text = ""
    for n in range(mincommentlines,maxcommentlines):
        try:
            text += (datalines[n] + " ")
        except:
            #print len(datalines), ID,video_id
            break
    wordlist = ""
    word2freq = defaultdict(int)
    mecab = MeCab.Tagger('-u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/ncnc.dic')
    node = mecab.parseToNode(text)
    while node:
        #if (node.feature.split(",")[0] == "名詞") | (node.feature.split(",")[0] == "形容詞") | (node.feature.split(",")[0] == "形容動詞"):
        wordlist += node.surface
        wordlist += " "
        word2freq[node.surface] += 1
        node = node.next
    return word2freq,wordlist[0:-1]
def makeTfidfTextList(mincoumment,maxcomment,mincommentlines,maxcommentlines):
    word2freqlist = {}
    wordlist = {}
    for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        word2freqlist[ID] = {}
        wordlist[ID] = {}
        for j in textinfo[ID].keys():
            if (thread[ID][(str(j) + ".dat")]["comment_counter"] > mincoumment) & (thread[ID][(str(j) + ".dat")]["comment_counter"] < maxcomment):
                try:
                    word2freqlist[ID][j], wordlist[ID][j] = makewordlist(ID,j,mincommentlines,maxcommentlines)
                except:
                    print ID,j
    tfidfTextList = {}
    voc = model.vocab.keys()
    for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        print ID
        for n in wordlist[ID].keys():
            tfidfTextList[n] = ""
            for w in wordlist[ID][n].split(' '):
                try:
                    k = model[w.decode("utf-8")]
                    tfidfTextList[n] += w
                    tfidfTextList[n] += " "
                except:
                    #print n,w
                    continue
    return (tfidfTextList,word2freqlist)


def tokenize(text):
    wakatilist = text.split(" ")
    return wakatilist

 
tfidf = TfidfVectorizer(tokenizer=tokenize)
(TfidfTextList, word2freqlist) = makeTfidfTextList(1000,10000)
tfs = tfidf.fit_transform(TfidfTextList.values())
feature_names = tfidf.get_feature_names()

n = 0
idlist = TfidfTextList.keys()

def maketfidfvec(number,feature_names,tfs,idlist,word2freqlist):
    d = dict(zip(feature_names, tfs[number].toarray().T))
    videoid = idlist[number]
    for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        try:
            k = word2freqlist[ID][videoid]
            break
        except:
            continue
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    for word, freq in sorted(k.items(),key = lambda x: x[1], reverse=True):
        if int(freq) > 3:
            try:
                v += wordvec(word.decode("utf-8")) * d[word.decode("utf-8")]
            except:
                #print word,ID,videoid
                continue
    if np.linalg.norm(v) > 0:
        return v/np.linalg.norm(v)
    else:
        return v
"""
import pickle
f = open('vectorinfo.dump','w')
pickle.dump(vectorinfo,f)
f.close()
f = open('tfidfTextList.dump','w')
pickle.dump(tfidfTextList,f)
f.close()
f = open('tfs.dump','w')
pickle.dump(tfs,f)
f.close()
f = open('word2freqlist.dump','w')
pickle.dump(word2freqlist,f)
f.close()
f = open('idlist.dump','w')
pickle.dump(idlist,f)
f.close()
"""
tfidfvectorinfo = {}
sample = tfs.toarray().shape[0]
for n in range(0,sample):
    print n
    tfidfvectorinfo[idlist[n]] = maketfidfvec(n)
"""
f = open('tfidfvectorinfo.dump','w')
pickle.dump(tfidfvectorinfo,f)
f.close()
"""
def createtfidfvectorMat(mincomment,maxcomment,tfidfvectorinfo):
    tfidfvectorMat = np.array([])
    for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
        for j in range(0,len(l[ID])):
            if (thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] > mincomment) & (thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] < maxcomment):
                if (tfidfvectorMat.shape[0] == 0):
                    tfidfvectorMat = tfidfvectorinfo[l[ID][j]]
                else:
                    tfidfvectorMat = np.c_[tfidfvectorMat,tfidfvectorinfo[l[ID][j]]]
    return tfidfvectorMat.T
tfidfdata = createtfidfvectorMat(1000,10000)

k2 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv =linear_model.LogisticRegression(C=1e5))
k0 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True,class_weight = {0:1,1:1}))


#サポートベクトル回帰
from sklearn import svm
from sklearn import cross_validation

viewcountlist = []
for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    for j in range(0,len(l[ID])):
        viewcountlist.append(thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"])
#データを6割をトレーニング、4割をテスト用とする
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data, np.array(viewcountlist), test_size=0.2)
# 線でつないでplotする用にx_test・y_testをx_testの昇順に並び替える
index = y_test.argsort(0).reshape(len(y_test))
x_test = x_test[index]
y_test = y_test[index]
# サポートベクトル回帰を学習データ使って作成
reg = svm.SVR(kernel='rbf', C=10).fit(x_train, y_train)
reg = logreg.fit(x_train, y_train)
# テストデータに対する予測結果のPLOT
plt.plot(y_train)
plt.plot(reg.predict(X_train),c = "red")
plt.show()
# 決定係数R^2
print reg.score(x_test, y_test)


#PCAによる図示
dim=3
pca=sklearn.decomposition.PCA(dim)
result=pca.fit_transform(data)

from mpl_toolkits.mplot3d.axes3d import Axes3D
dim=3
pca=sklearn.decomposition.PCA(dim)
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter3D(result.T[0],result.T[1],result.T[2])
plt.show()

dim=2
pca=sklearn.decomposition.PCA(dim)
fig=plt.figure()
plt.scatter(result.T[0],result.T[1])
plt.show()

result2 = result[0]
result3 = result[0]
for ID in for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    for j in range(0,len(l[ID])):
        if (ID == "0000") & (j == 0):
            vectorMat = vectorinfo["0000"][l["0000"][0]]
        else:
            vectorMat = np.c_[vectorMat,vectorinfo[ID][l[ID][j]]]
    for j in range(0,len(l[ID])):
        if thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > 10000:#np.median(counters):
            result2 = np.c_[result2,result[j]]
        else:
            result3 = np.c_[result3,result[j]]

for ID in for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    for j in range(0,len(l[ID])):
        if (ID == "0000") & (j == 0):
            vectorMat = vectorinfo["0000"][l["0000"][0]]
        else:
            vectorMat = np.c_[vectorMat,vectorinfo[ID][l[ID][j]]]
    for j in range(1,len(l[ID])):
        if thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] > 2500:#np.median(counters):
            result2 = np.c_[result2,result[j]]
        else:
            result3 = np.c_[result3,result[j]]

fig=plt.figure()
ax=Axes3D(fig)
ax.scatter3D(result3[0],result3[1],result3[2],c= "blue")
ax.scatter3D(result2[0],result2[1],result2[2],c="red")
plt.show()

plt.scatter(result3[0],result3[1],c= "b")
plt.scatter(result2[0],result2[1],c="r")
plt.show()

#類似する文書を抜く
def selectTitleFromID(video_id,id="0000"):
    titlerank = {}
    for Id in vectorinfo.keys():
        try:
            inputvec = vectorinfo[Id][video_id]
        except:
            continue
        print Id
    for ID in vectorinfo.keys():
        for j in vectorinfo[ID].keys():
            if np.dot(vectorinfo[ID][j], inputvec) > 0.8:
                titlerank[j] = np.dot(vectorinfo[ID][j], inputvec)
                print j
    k = sorted(titlerank.items(), key=lambda x: x[1],reverse = True)
    for m in k[0:20]:
        for ID in textinfo.keys():
            try:
                print textinfo[ID][m[0]], m[0], m[1]
                print "view_counter", thread[ID][(str(m[0]) + ".dat")]["view_counter"]
                print "comment_counter", thread[ID][(str(m[0]) + ".dat")]["comment_counter"]
                tag  = ""
                for j in thread[ID][(str(m[0]) + ".dat")]["tags"]:
                    tag += (j["tag"].decode('unicode_escape') + ",")
                print  ("tags: " + tag[0:-1])
                #print j, textinfo[j], np.dot(vectorinfo[j], vectorinfo[video_id])
            except:
                continue

def selectTitleFromWord(wordarray):
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    for j in wordarray:
        v += wordvec(j)
    v = v/np.linalg.norm(v)
    titlerank = {}
    for ID in vectorinfo.keys():
        for j in vectorinfo[ID].keys():  
            if np.dot(v, vectorinfo[ID][j]) > 0.5:
                titlerank[j] = np.dot(v, vectorinfo[ID][j])
    k = sorted(titlerank.items(), key=lambda x: x[1],reverse = True)
    for m in k[0:20]:
        for ID in textinfo.keys():
            try:
                print textinfo[ID][m[0]], m[0], m[1]
            except:
                continue

#閲覧数とコメント数
viewcounters = []
for ID in for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    for j in thread[ID].keys():
        viewcounters.append(thread[ID][j]["view_counter"])

commentcounter = []
for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    for j in thread[ID].keys():
        commentcounter.append(thread[ID][j]['comment_counter'])

commentviewcounter = []
for ID in for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
    for j in range(0,len(l[ID])):
        if (ID == "0000") & (j == 0):
            vectorMat = vectorinfo["0000"][l["0000"][0]]
        else:
            vectorMat = np.c_[vectorMat,vectorinfo[ID][l[ID][j]]]
    for j in thread[ID].keys():
        print j
        commentviewcounter.append([thread[ID][j]['comment_counter'],thread[ID][j]["view_counter"]])
commentviewcounter2 = sorted(commentviewcounter)
plt.scatter(np.array(commentviewcounter2).T[0][0:6100],np.array(commentviewcounter2).T[1][0:6100])
plt.xlabel("comment_counter")
plt.ylabel("view_counter")
plt.show()


commentcounter2 = list(commentcounter)
commentcounter2.sort(key=int)
plt.title("0000-0003commentcount")
plt.xlabel("rank")
plt.ylabel("commentcount")
plt.plot(commentcounter2)

plt.title("0000-0003commentcount_under6000")
plt.xlabel("rank")
plt.ylabel("commentcount")
plt.plot(commentcounter2[0:6000])

counters2 = []
for j in thread["0003"].keys():
    counters2.append(thread["0003"][j]["view_counter"])

commentcounter3 = []
for j in thread["0003"].keys():
    commentcounter3.append(thread["0003"][j]["comment_counter"])

plt.scatter(counters2,commentcounter3)
plt.show()