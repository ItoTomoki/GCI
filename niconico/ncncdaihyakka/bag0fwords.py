import numpy as np

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
maketext('0003','sm30883',0,1000000)

import pickle
import gensim
g = file("textinfo.dump","r")
textinfo = pickle.load(g)
g.close()
g = file("thread.dump","r")
thread = pickle.load(g)
g.close()

g = file("textinfo0000-0006.dump","r")
textinfo = pickle.load(g)
g.close()
g = file("thread0000-0006.dump","r")
thread = pickle.load(g)
g.close()

from gensim import corpora, models, similarities

def vec2dense(vec, num_terms):
	'''Convert from sparse gensim format to dense list of numbers'''
	return list(gensim.matutils.corpus2dense([vec], num_terms=num_terms).T[0])

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

def createtargetarray(mincomment,maxcomment,threadviewcount1,threadviewcount2,bow_docs):
	target2 = []
	for k in bow_docs.keys():
		if (commentarray[k] > mincomment) & (commentarray[k] < maxcomment):
			if viewarray[k] > threadviewcount2:
				target2.append(0)
			elif viewarray[k] > threadviewcount1:
				target2.append(1)
			else:
				target2.append(2)
	return np.array(target2)

def createtvectorMat(mincomment,maxcomment,bow_docs,dct):
	vectorMat2 = np.array([])
	for name in bow_docs.keys():
			if (commentarray[name] > mincomment) & (commentarray[name] < maxcomment):
				if (np.array(vectorMat2)).shape[0] == 0:
					sparse = bow_docs[name]
					vectorMat2 = vec2dense(sparse, num_terms=len(dct))
				else:
					sparse = bow_docs[name]
					vectorMat2 = np.c_[vectorMat2,vec2dense(sparse, num_terms=len(dct))]
	return(vectorMat2.T)

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
        vmat = clf_cv.coef_[0]
        if ifprint == True:
            print(classification_report(y_true, y_pred))
        if checkauc == True:
            y_pred_cv = clf_cv.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_cv)
            aucs.append(auc)
    if checkauc == True:
        print np.mean(aucs), np.std(aucs)
    print(classification_report(y_trueall, y_pridictall))
    return y_trueall, y_pridictall,vmat

import gensim
#100後から以上の動画を最初の100後を使って文類
preprocessed_docs = maketextdoc(100,10000000,0,100)
dct = gensim.corpora.Dictionary(preprocessed_docs.values())
unfiltered = dct.token2id.keys()
dct.filter_extremes(no_below=3, no_above=0.6)
filtered = dct.token2id.keys()
filtered_out = set(unfiltered) - set(filtered)
bow_docs = {}
bow_docs_all_zeros = {}
for name in preprocessed_docs.keys():
	sparse = dct.doc2bow(preprocessed_docs[name])
	bow_docs[name] = sparse
	dense = vec2dense(sparse, num_terms=len(dct))
	bow_docs_all_zeros[name] = all(d == 0 for d in dense)
target2 = createtargetarray(100,10000000,10760.0,34544,bow_docs)
data2 = createtvectorMat(100,10000000,bow_docs,dct)
PredictAndAnalyze2(data = data2,target = target2)

#douga数毎で分類
for number in [200,300,400,500,600,700,800,900,1000]:
	preprocessed_docs = maketextdoc(number - 100,number + 100,0,number + 100)
	dct = gensim.corpora.Dictionary(preprocessed_docs.values())
	unfiltered = dct.token2id.keys()
	dct.filter_extremes(no_below=3, no_above=0.6)
	filtered = dct.token2id.keys()
	filtered_out = set(unfiltered) - set(filtered)
	bow_docs = {}
	bow_docs_all_zeros = {}
	for name in preprocessed_docs.keys():
		sparse = dct.doc2bow(preprocessed_docs[name])
		bow_docs[name] = sparse
		dense = vec2dense(sparse, num_terms=len(dct))
		bow_docs_all_zeros[name] = all(d == 0 for d in dense)
	target2 = createtargetarray(number - 100,number + 100,10760.0,34544,bow_docs = bow_docs)
	data2 = createtvectorMat(number - 100,number + 100,bow_docs = bow_docs,dct = dct)
	print number-100,number + 100
	print "LogisticRegression"
	k2 = PredictAndAnalyze2(data2,target2,clf_cv =linear_model.LogisticRegression(C=1e1))
	f = file(("LogReg" + str(number) + ".dump"),"w")
	pickle.dump(k2[2].T,f)
	f.close()
	print "svm.SVC"
	k0 = PredictAndAnalyze2(data = data2,target = target2,clf_cv = svm.SVC(kernel='linear', probability=True))
	f = file(("svm" + str(number) + ".dump"),"w")
	pickle.dump(k0[2].T,f)
	f.close()



#最初のコメントを利用
def bagofwordsvaluate(mincount2,maxcount2,PredictAndAnalyze = PredictAndAnalyze2):
	preprocessed_docs = maketextdoc(maxcount2,100000000,mincount2,maxcount2)
	dct = gensim.corpora.Dictionary(preprocessed_docs.values())
	unfiltered = dct.token2id.keys()
	dct.filter_extremes(no_below=3, no_above=0.6)
	filtered = dct.token2id.keys()
	filtered_out = set(unfiltered) - set(filtered)
	bow_docs = {}
	bow_docs_all_zeros = {}
	for name in preprocessed_docs.keys():
		sparse = dct.doc2bow(preprocessed_docs[name])
		bow_docs[name] = sparse
		dense = vec2dense(sparse, num_terms=len(dct))
		bow_docs_all_zeros[name] = all(d == 0 for d in dense)
	target3 = createtargetarray(maxcount2,100000000,10760.0,34544,bow_docs = bow_docs)
	data3 = createtvectorMat(maxcount2,100000000,bow_docs = bow_docs,dct = dct)
	print mincount2,maxcount2,len(dct)
	print "LogisticRegression"
	k2 = PredictAndAnalyze2(data3,target3,clf_cv =linear_model.LogisticRegression(C=1e1))
	f = file(("LogReg" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k2[2].T,f)
	f.close()
	print "svm.SVC"
	k0 = PredictAndAnalyze2(data = data3,target = target3,clf_cv = svm.SVC(kernel='linear', probability=True))
	f = file(("svm" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k0[2].T,f)
	f.close()


def bagofwordsvaluate3(mincount2,maxcount2,PredictAndAnalyze = PredictAndAnalyze2):
	preprocessed_docs = maketextdoc(abs(mincount2),100000000,mincount2,maxcount2)
	dct = gensim.corpora.Dictionary(preprocessed_docs.values())
	unfiltered = dct.token2id.keys()
	dct.filter_extremes(no_below=3, no_above=0.6)
	filtered = dct.token2id.keys()
	filtered_out = set(unfiltered) - set(filtered)
	bow_docs = {}
	bow_docs_all_zeros = {}
	for name in preprocessed_docs.keys():
		sparse = dct.doc2bow(preprocessed_docs[name])
		bow_docs[name] = sparse
		dense = vec2dense(sparse, num_terms=len(dct))
		bow_docs_all_zeros[name] = all(d == 0 for d in dense)
	target3 = createtargetarray(abs(mincount2),100000000,10760.0,34544,bow_docs = bow_docs)
	data3 = createtvectorMat(abs(mincount2),100000000,bow_docs = bow_docs,dct = dct)
	print mincount2,maxcount2,len(dct)
	print "LogisticRegression"
	k2 = PredictAndAnalyze2(data3,target3,clf_cv =linear_model.LogisticRegression(C=1e1))
	f = file(("LogReg" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k2[2].T,f)
	f.close()
	print "svm.SVC"
	k0 = PredictAndAnalyze2(data = data3,target = target3,clf_cv = svm.SVC(kernel='linear', probability=True))
	f = file(("svm" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k0[2].T,f)
	f.close()

for k in [10,100,500,700,1000]:
	bagofwordsvaluate(0,k,PredictAndAnalyze = PredictAndAnalyze2)
	if k != 10:
		bagofwordsvaluate(50,k,PredictAndAnalyze = PredictAndAnalyze2)
	bagofwordsvaluate3((-1*k),0,PredictAndAnalyze = PredictAndAnalyze2)
