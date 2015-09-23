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

#modelwiki = word2vec.Word2Vec.load("wiki1-4.model")
#data = word2vec.Text8Corpus('allcomment.txt')
data = word2vec.Text8Corpus('allcomment2kaiseiki.txt')
#modelnico = word2vec.Word2Vec(data, size=50)
#modelnico = word2vec.Word2Vec.load("allcomment2.model")
#modelnico = word2vec.Word2Vec.load("allcomment1.model")
modelnico = word2vec.Word2Vec.load("allcomment2kaiseiki.model") #コメントをスペースでつなぐ改行正規化済み
#modelnico = word2vec.Word2Vec.load("allcomment2.model")　#コメントをスペースでつなぐ改行正規化なし
#modelnico = word2vec.Word2Vec.load("allcomment_kai.model") #コメントをひとつずつ改行正規化済み
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

def createvector(video_id,ID="0000"):
    if video_id == "sm9":
        return np.zeros(len(model[model.vocab.keys()[0]]))
    else:
        filename = ("comment2_" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment2_kai" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment2_bigram" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment2bigram_kai" + ID + "/" + str(video_id) + ".txt")
        #filename = ("comment_kai" + ID + "/" + str(video_id) + ".txt")
        f = open(filename)
        data = f.read()
        f.close()
        v = makevec(morphological_analysis(data))
        return v
vectorinfo = {}

files = os.listdir('../data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/video')
textinfo = {}
thread = {}
count = 0
#for file in files[1:2]:
for ID in ["0000","0001","0002","0003"]:
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

for ID in ["0000","0001","0002","0003"]:
    vectorinfo[ID] = {}
    for j in textinfo[ID].keys():
        #print j
        try:
            vectorinfo[ID][j] = createvector(video_id = j, ID = ID)
        except:
            #vectorinfo[ID][j] = np.zeros(len(model[model.vocab.keys()[0]]))
            print ID,j


l = {}
for ID in ["0000","0001","0002","0003"]:
    l[ID] = vectorinfo[ID].keys()


for ID in ["0000","0001","0002","0003"]:
    for j in range(0,len(l[ID])):
        if (ID == "0000") & (j == 0):
            vectorMat = vectorinfo["0000"][l["0000"][0]]
        else:
            vectorMat = np.c_[vectorMat,vectorinfo[ID][l[ID][j]]]
data = vectorMat.T



#閲覧数
viewcounters = []
for ID in ["0000","0001","0002","0003"]:
    for j in thread[ID].keys():
        viewcounters.append(thread[ID][j]["view_counter"])

commentcounter = []
for ID in ["0000","0001","0002","0003"]:
    for j in thread[ID].keys():
        commentcounter.append(thread[ID][j]['comment_counter'])

commentcounter2 = list(commentcounter)
commentcounter2.sort(key=int)
plt.plot(commentcounter2)

counters2 = []
for j in thread["0003"].keys():
    counters2.append(thread["0003"][j]["view_counter"])

commentcounter3 = []
for j in thread["0003"].keys():
    commentcounter3.append(thread["0003"][j]["comment_counter"])

plt.scatter(counters2,commentcounter3)
plt.show()


#機会学習
#34544は平均値で10760.0は中央値（再生回数）
from sklearn import neighbors,svm
from sklearn.metrics import classification_report
target = []
for ID in ["0000","0001","0002","0003"]:
    for j in range(0,len(l[ID])):
        if thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > 34544:#10760.0:#34544:
            target.append(0)
        elif thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > 10760.0:
            target.append(1)
        else:
            target.append(2)
target = np.array(target)


#コメント数で制限
target2 = []
for ID in ["0000","0001","0002","0003"]:
    for j in range(0,len(l[ID])):
        if thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] > 100 & thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] < 1500:
            if thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > 34544:#10760.0:#34544:
                target2.append(0)
            elif thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > 10760.0:
                target2.append(1)
            else:
                target2.append(2)

target2 = np.array(target2)


vectorMat2 = np.array([])
for ID in ["0000","0001","0002","0003"]:
    for j in range(0,len(l[ID])):
        if thread[ID][(str(l[ID][j]) + ".dat")]["comment_counter"] > 100:
            if vectorMat2.shape[0] == 0:
                vectorMat2 = vectorinfo[ID][l[ID][j]]
            else:
                vectorMat2 = np.c_[vectorMat2,vectorinfo[ID][l[ID][j]]]

data2 = vectorMat2.T



scores = []
knn = neighbors.KNeighborsClassifier(n_neighbors=10) #metric='manhattan'
classifier = svm.SVC(kernel='linear', probability=True)#,class_weight={0:3,2:1})
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
scores = cross_validation.cross_val_score(classifier, data, target, cv=5)
print np.mean(scores)
scores = cross_validation.cross_val_score(logreg, data, target, cv=5)
print np.mean(scores)

def PredictAndAnalyze(data = data,target = target,clf_cv = svm.SVC(kernel='linear', probability=True),checkauc = False):
    kf = KFold(len(target), n_folds=10, shuffle=True)
    aucs = []
    y_trueall = []
    y_pridictall = []
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(target)[train]
        X_test, y_test = np.array(data)[val], np.array(target)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)
        y_true = y_test
        y_trueall = y_trueall + list(y_true)
        y_pridictall = y_pridictall  + list(y_pred)
        print(classification_report(y_true, y_pred))
        if checkauc == True:
            y_pred_cv = clf_cv.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_cv)
            aucs.append(auc)
    if checkauc == True:
        print np.mean(aucs), np.std(aucs)
    print(classification_report(y_trueall, y_pridictall))
    return y_trueall, y_pridictall

k1 = PredictAndAnalyze(data2,target2,clf_cv = neighbors.KNeighborsClassifier(n_neighbors=10))
k2 = PredictAndAnalyze(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e5))


#tf-idf
def makewordlist(ID,video_id):
    filename = ("comment2_" + ID + "/" + str(video_id) + ".txt")
    #filename = ("comment2_kai" + ID + "/" + str(video_id) + ".txt")
    f = open(filename)
    text = f.read()
    f.close()
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

word2freqlist = {}
wordlist = {}
for ID in ["0000","0001","0002","0003"]:
    word2freqlist[ID] = {}
    wordlist[ID] = {}
    for j in textinfo[ID].keys():
        try:
            word2freqlist[ID][j], wordlist[ID][j] = makewordlist(ID,j)
        except:
            print ID,j

tfidfTextList = {}
voc = model.vocab.keys()
for ID in ["0000","0001","0002","0003"]:
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

def tokenize(text):
    wakatilist = text.split(" ")
    return wakatilist

tfidf = TfidfVectorizer(tokenizer=tokenize)
tfs = tfidf.fit_transform(tfidfTextList.values())
feature_names = tfidf.get_feature_names()

n = 0
idlist = tfidfTextList.keys()

def maketfidfvec(number):
    d = dict(zip(feature_names, tfs[number].toarray().T))
    videoid = idlist[number]
    for ID in ["0000","0001","0002","0003"]:
        try:
            k = word2freqlist[ID][videoid]
            break
        except:
            continue
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    for word, freq in sorted(k.items(),key = lambda x: x[1], reverse=True):
        if int(freq) > 0:
            try:
                v += freq * wordvec(word.decode("utf-8"))* d[word.decode("utf-8")]
            except:
                #print word,ID,videoid
                continue
    if np.linalg.norm(v) > 0:
        return v/np.linalg.norm(v)
    else:
        return v

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

tfidfvectorinfo = {}
sample = tfs.toarray().shape[0]
for n in range(0,sample):
    print n
    tfidfvectorinfo[idlist[n]] = maketfidfvec(n)

f = open('tfidfvectorinfo.dump','w')
pickle.dump(tfidfvectorinfo,f)
f.close()

tfidfvectorMat = np.array([])
for ID in ["0000","0001","0002","0003"]:
    for j in range(0,len(l[ID])):
        if (tfidfvectorMat.shape[0] == 0):
            tfidfvectorMat = tfidfvectorinfo[l[ID][j]]
        else:
            tfidfvectorMat = np.c_[tfidfvectorMat,tfidfvectorinfo[l[ID][j]]]
tfidfdata = tfidfvectorMat.T
#k0 = PredictAndAnalyze(data = tfidfdata,target = target,clf_cv = svm.SVC(kernel='linear', probability=True,class_weight={0:4,1:1}))
k2 = PredictAndAnalyze(data = tfidfdata,target = target,clf_cv =linear_model.LogisticRegression(C=1e5,class_weight={0:4,1:1}))
k0 = PredictAndAnalyze(data = tfidfdata,target = target,clf_cv = svm.SVC(kernel='linear', probability=True))


#サポートベクトル回帰
from sklearn import svm
from sklearn import cross_validation

viewcountlist = []
for ID in ["0000","0001","0002","0003"]:
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
plt.plot(y_test)
plt.plot(reg.predict(x_test),c = "red")
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
for ID in ["0000","0001","0002","0003"]:
    for j in range(0,len(l[ID])):
        if thread[ID][(str(l[ID][j]) + ".dat")]["view_counter"] > 10000:#np.median(counters):
            result2 = np.c_[result2,result[j]]
        else:
            result3 = np.c_[result3,result[j]]

for ID in ["0000","0001","0002","0003"]:
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

