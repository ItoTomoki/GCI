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



target2 = createtargetarray(100,100000000,10760.0,34544)
data2 = createtvectorMat(100,100000000,vectorinfo)
k0 = PredictAndAnalyze(data2,target2,clf_cv = svm.SVC(kernel='linear', probability=True,class_weight={0:2,1:1}))
k0 = PredictAndAnalyze(data2,target2,clf_cv = svm.SVC(kernel='linear', probability=True))
k1 = PredictAndAnalyze(data2,target2,clf_cv = neighbors.KNeighborsClassifier(n_neighbors=10))
k2 = PredictAndAnalyze(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1))

l = {}
for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
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
	target2 = createtargetarray(narray * 100 - 100,narray * 100 + 100,10760.0,34544)
	data2 = createtvectorMat(narray * 100 - 100,narray * 100 + 100)
	(TfidfTextList, word2freqlist) = makeTfidfTextList(narray * 100 - 100,narray * 100 + 100,narray * 100 - 100,narray * 100 + 100)
	tfs = tfidf.fit_transform(TfidfTextList.values())
	idlist = TfidfTextList.keys()
	tfidfvectorinfo = {}
	sample = tfs.toarray().shape[0]
	for n in range(0,sample):
		tfidfvectorinfo[idlist[n]] = maketfidfvec(n)
	tfidfdata = createtfidfvectorMat(narray * 100 - 100,narray * 100 + 100)
	print narray * 100
	k2 = PredictAndAnalyze2(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1),balancing = False)
	k2 = PredictAndAnalyze2(data = tfidfdata,target = target2,clf_cv =linear_model.LogisticRegression(C=1e1),balancing = False)
	k0 = PredictAndAnalyze2(data = data2,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True),balancing = False)
	k0 = PredictAndAnalyze2(data = tfidfdata,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True),balancing = False)


#コメント数毎で比較2
tfidf = TfidfVectorizer(tokenizer=tokenize)
def valuate(mincount2,maxcount2,PredictAndAnalyze = PredictAndAnalyze2):
	for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
		vectorinfo[ID] = {}
		for j in textinfo[ID].keys():
			#print j
			try:
				vectorinfo[ID][j] = createvector(video_id = j, ID = ID,mincount = mincount2,maxcount = maxcount2)
			except:
				#vectorinfo[ID][j] = np.zeros(len(model[model.vocab.keys()[0]]))
				print ID,j
	target2 = createtargetarray(maxcount2,100000000,10760.0,34544)
	data2 = createtvectorMat(maxcount2,100000000,vectorinfo)
	(TfidfTextList, word2freqlist) = makeTfidfTextList(maxcount2,100000000,mincount2,maxcount2)
	tfidf = TfidfVectorizer(tokenizer=tokenize)
	tfs = tfidf.fit_transform(TfidfTextList.values())
	idlist = TfidfTextList.keys()
	feature_names = tfidf.get_feature_names()
	tfidfvectorinfo = {}
	sample = tfs.toarray().shape[0]
	print sample, len(feature_names), 
	for n in range(0,sample):
		#print n
		tfidfvectorinfo[idlist[n]] = maketfidfvec(n,feature_names = feature_names,tfs = tfs,idlist= idlist,word2freqlist = word2freqlist)
	l = {}
	for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
		l[ID] = vectorinfo[ID].keys()
	tfidfdata = createtfidfvectorMat(maxcount2,100000000,tfidfvectorinfo)
	print mincount2,maxcount2
	k2 = PredictAndAnalyze(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1))
	k22 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv =linear_model.LogisticRegression(C=1e1))
	k0 = PredictAndAnalyze(data = data2,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))
	k00 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))
	return k2,k0

def valuate1(mincount2,maxcount2,PredictAndAnalyze = PredictAndAnalyze):
	for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
		vectorinfo[ID] = {}
		for j in textinfo[ID].keys():
			#print j
			try:
				vectorinfo[ID][j] = createvector(video_id = j, ID = ID,mincount = mincount2,maxcount = maxcount2)
			except:
				#vectorinfo[ID][j] = np.zeros(len(model[model.vocab.keys()[0]]))
				print ID,j
	target2 = createtargetarray(maxcount2,100000000,10760.0,34544)
	data2 = createtvectorMat(maxcount2,100000000)
	print mincount2,maxcount2
	k2 = PredictAndAnalyze(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1))
	#k2 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv =linear_model.LogisticRegression(C=1e1))
	k0 = PredictAndAnalyze(data = data2,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))
	#k0 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))
	return k2,k0

def valuate3(mincount2,maxcount2,PredictAndAnalyze = PredictAndAnalyze2):
	for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
		vectorinfo[ID] = {}
		for j in textinfo[ID].keys():
			#print j
			try:
				vectorinfo[ID][j] = createvector(video_id = j, ID = ID,mincount = mincount2,maxcount = maxcount2)
			except:
				#vectorinfo[ID][j] = np.zeros(len(model[model.vocab.keys()[0]]))
				print ID,j
	target2 = createtargetarray(abs(mincount2),100000000,10760.0,34544)
	data2 = createtvectorMat(abs(mincount2),100000000,vectorinfo)
	(TfidfTextList, word2freqlist) = makeTfidfTextList(abs(mincount2),100000000,maxcount2,mincount2)
	tfs = tfidf.fit_transform(TfidfTextList.values())
	idlist = TfidfTextList.keys()
	feature_names = tfidf.get_feature_names()
	tfidfvectorinfo = {}
	sample = tfs.toarray().shape[0]
	for n in range(0,sample):
		tfidfvectorinfo[idlist[n]] = maketfidfvec(n,feature_names = feature_names,tfs = tfs,idlist= idlist,word2freqlist = word2freqlist)
	l = {}
	for ID in ["0000","0001","0002","0003","0004","0005","0006"]:
		l[ID] = vectorinfo[ID].keys()
	tfidfdata = createtfidfvectorMat(abs(mincount2),100000000,tfidfvectorinfo)
	print mincount2,maxcount2
	k2 = PredictAndAnalyze(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1))
	k22 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv =linear_model.LogisticRegression(C=1e1))
	k0 = PredictAndAnalyze(data = data2,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))
	k00 = PredictAndAnalyze(data = tfidfdata,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))
	return k2,k0

#最初のコメントを利用
for k in [10,100,500,700,1000]:
	tfidf = TfidfVectorizer(tokenizer=tokenize)
	k2,k0 = valuate(0,k,PredictAndAnalyze = PredictAndAnalyze2)
	f = file(("svm" + str(k) + "valuate1"+ ".dump"),"w")
	pickle.dump(k0[2].T,f)
	f.close()
	f = file(("logreg" + str(k) + "valuate1" + ".dump"),"w")
	pickle.dump(k2[2].T,f)
	f.close()
	if k != 10:
		k2,k0 = valuate(50,k,PredictAndAnalyze = PredictAndAnalyze2)
		f = file(("svm" + str(k) + "valuate2"+ ".dump"),"w")
		pickle.dump(k0[2].T,f)
		f.close()
		f = file(("logreg" + str(k) + "valuate2" + ".dump"),"w")
		pickle.dump(k2[2].T,f)
		f.close()
	k2,k0 = valuate3((-1*k),0,PredictAndAnalyze = PredictAndAnalyze2)
	f = file(("svm" + str(k) + "valuate3"+ ".dump"),"w")
	pickle.dump(k0[2].T,f)
	f.close()
	f = file(("logreg" + str(k) + "valuate3" + ".dump"),"w")
	pickle.dump(k2[2].T,f)
	f.close()
#最初の50コメント無視
for k in [100,500,1000]:
	valuate(50,k)
#最後のコメント利用
for k in [-10,-100,-500,-700,-1000]:
	valuate3(k,0)

