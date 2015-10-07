#encoding:utf-8
import numpy as np
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
from k_means_2 import KMeans

modelnico = word2vec.Word2Vec.load("allcomment0000-0006_200.model")
model = modelnico
def maketext(ID,video_id,mincommentlines,maxcommentlines):
	#filename = ("comment2_" + ID + "/" + str(video_id) + ".txt")
	#filename = ("comment2_kai" + ID + "/" + str(video_id) + ".txt")
	filename = ("comment_kai" + ID + "/" + str(video_id) + ".txt")
	f = open(filename)
	text = f.read()
	f.close()
	datalines = text.split('\n')
	#print len(datalines)
	text = ""
	for n in range(mincommentlines,maxcommentlines):
		try:
			text += (datalines[n] + " ")
		except:
			break
	return text.decode("utf-8").split(" ")

voclist = model.vocab.keys()
veclist = {}
for k in voclist:
	veclist[k] = np.array(model[k]/np.linalg.norm(model[k]))
M = np.array(veclist.values())
voclist = veclist.keys()
voclist2 = voclist[0:1000]
"""
def wordclustering(model,voclist):
	vecnamelist = {}
	for name in voclist:
		vecnamelist[name] = np.array(model[name]/np.linalg.norm(model[name]))
	return vecnamelist

def createwordvecmat(vecnamelist,voclist):
	wordvecmat = np.array([])
	for name in voclist:
		if wordvecmat.shape[0] == 0:
			wordvecmat = vecnamelist[name]
		else:
			vec = vecnamelist[name]
			wordvecmat = np.c_[wordvecmat,vec]
	return (wordvecmat.T)

vecnamelist = wordclustering(model,voclist)
features = createwordvecmat(vecnamelist,voclist)
"""
features = M
kmeans = KMeans(n_clusters=500, random_state=100)
kmeans_model = kmeans.fit(features)
labels = kmeans_model.labels_
d = zip(labels, features)
feature2 = []
for label, feature in d:
	if label == 145:
		feature2.append(feature)

for label, feature in zip(labels, features):
    print(label, feature, feature.sum())
word2vecdic = dict(zip(voclist, labels))
for k in word2vecdic.keys():
	if word2vecdic[k] == 3:
		print k

#ベクトル作成
def maketextdoc(mincoumment,maxcomment,mincommentlines,maxcommentlines):
	doc = {}
	for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
		for j in textinfo[ID].keys():
			if (thread[ID][(str(j) + ".dat")]["comment_counter"] > mincoumment) & (thread[ID][(str(j) + ".dat")]["comment_counter"] < maxcomment):
				try:
					doc[j] = maketext(ID,j,mincommentlines,maxcommentlines)
				except:
					print ID,j
	return doc

g = file("textinfo0000-0006.dump","r")
textinfo = pickle.load(g)
g.close()
g = file("thread0000-0006.dump","r")
thread = pickle.load(g)
g.close()

preprocessed_docs = maketextdoc(100,10000000,0,100)
docvec = {}

for k in preprocessed_docs.keys():
	docvec[k] = np.zeros(1000)
	sentence = preprocessed_docs[k]
	for m in sentence:
		try:
			labelnum = word2vecdic[m]
			docvec[k][labelnum] = (docvec[k][labelnum] + 1)
		except:
			continue

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
    kf = KFold(len(target), n_folds=10, shuffle=True)
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(target)[train]
        X_test, y_test = np.array(data)[val], np.array(target)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)
        y_true = y_test
        y_trueall = y_trueall + list(y_true)
        y_pridictall = y_pridictall  + list(y_pred)
        vmat0 = clf_cv.coef_[0]
        vmat1 = clf_cv.coef_[1]
        vmat2 = clf_cv.coef_[2]        
        if ifprint == True:
            print(classification_report(y_true, y_pred))
        if checkauc == True:
            y_pred_cv = clf_cv.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_cv)
            aucs.append(auc)
    if checkauc == True:
        print np.mean(aucs), np.std(aucs)
    print(classification_report(y_trueall, y_pridictall))
    return y_trueall, y_pridictall,vmat0,vmat1,vmat2

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

def createtvectorMat(mincomment,maxcomment,docvec):
	vectorMat2 = np.array([])
	for name in docvec.keys():
			if (commentarray[name] > mincomment) & (commentarray[name] < maxcomment):
				if (np.array(vectorMat2)).shape[0] == 0:
					vectorMat2 = docvec[name]
				else:
					vectorMat2 = np.c_[vectorMat2,docvec[name]]
	return(vectorMat2.T)

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

for name in docvec.keys():
			if (commentarray[name] > 100) :
				arrays.append(True)
			else:
				arrays.append(False)


target2 = createtargetarray(100,10000000,10760.0,34544,docvec)
data2 = createtvectorMat2(100,10000000,docvec)
length = min([len(target2[target2 == 0]),len(target2[target2 == 1]),len(target2[target2 == 2])])
data = np.r_[data2[target2 == 0][0:length],data2[target2 == 1][0:length],data2[target2 == 2][0:length]]

k = PredictAndAnalyze2(data = data2,target = target2)


clf_cv = svm.LinearSVC()
clf_cv = svm.SVC(kernel='linear', probability=True)
clf_cv.fit(data2[0:5000],target2[0:5000])
y_pred = clf_cv.predict(data2[5000:6000])
y_true = target2[5000:6000]
vmat0 = clf_cv.coef_[0]
vmat1 = clf_cv.coef_[1]
vmat2 = clf_cv.coef_[2]        
print(classification_report(y_true, y_pred))