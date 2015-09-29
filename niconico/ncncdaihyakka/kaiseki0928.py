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

from kaiseki0925 import wordvec,morphological_analysis,output,makevec,createvector,createtargetarray,createtvectorMat,PredictAndAnalyze,makewordlist,makeTfidfTextList
from kaiseki0925 import tokenize,maketfidfvec,createtfidfvectorMat
data = word2vec.Text8Corpus('allcomment2kaiseiki.txt')
#modelnico = word2vec.Word2Vec(data, size=50)
#modelnico = word2vec.Word2Vec.load("allcomment2.model")
#modelnico = word2vec.Word2Vec.load("allcomment1.model")
modelnico = word2vec.Word2Vec.load("allcomment2kaiseiki.model") #コメントをスペースでつなぐ改行正規化済み
#modelnico = word2vec.Word2Vec.load("allcomment2.model")　#コメントをスペースでつなぐ改行正規化なし
#modelnico = word2vec.Word2Vec.load("allcomment_kai.model") #コメントをひとつずつ改行正規化済み
model = modelnico



target2 = createtargetarray(100,100000000,10000,30000)
data2 = createtvectorMat(100,100000000)
k0 = PredictAndAnalyze(data2,target2,clf_cv = svm.SVC(kernel='linear', probability=True,class_weight={0:2,1:1}))
k0 = PredictAndAnalyze(data2,target2,clf_cv = svm.SVC(kernel='linear', probability=True))
k1 = PredictAndAnalyze(data2,target2,clf_cv = neighbors.KNeighborsClassifier(n_neighbors=10))
k2 = PredictAndAnalyze(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1))

l = {}
for ID in ["0000","0001","0002","0003"]:
    l[ID] = vectorinfo[ID].keys()

(TfidfTextList, word2freqlist) = makeTfidfTextList(100,100000000)
tfs = tfidf.fit_transform(TfidfTextList.values())
idlist = TfidfTextList.keys()
tfidfvectorinfo = {}
sample = tfs.toarray().shape[0]
for n in range(0,sample):
	tfidfvectorinfo[idlist[n]] = maketfidfvec(n,100,100000000)
tfidfdata = createtfidfvectorMat(100,100000000)

k2 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv =linear_model.LogisticRegression(C=1e5))
k0 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))

#コメント数毎で比較
tfidf = TfidfVectorizer(tokenizer=tokenize)
for narray in range(2,10):
	target2 = createtargetarray(narray * 100 - 100,narray * 100 + 100,10000,30000)
	data2 = createtvectorMat(narray * 100 - 100,narray * 100 + 100)
	(TfidfTextList, word2freqlist) = makeTfidfTextList(narray * 100 - 100,narray * 100 + 100)
	tfs = tfidf.fit_transform(TfidfTextList.values())
	idlist = TfidfTextList.keys()
	tfidfvectorinfo = {}
	sample = tfs.toarray().shape[0]
	for n in range(0,sample):
		tfidfvectorinfo[idlist[n]] = maketfidfvec(n,narray * 100 - 100,narray * 100)
	tfidfdata = createtfidfvectorMat(narray * 100 - 100,narray * 100 + 100)
	print narray * 100
	k2 = PredictAndAnalyze(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1))
	k2 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv =linear_model.LogisticRegression(C=1e1))
	k0 = PredictAndAnalyze(data = data2,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))
	k0 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))



