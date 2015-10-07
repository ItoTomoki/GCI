#encoding:utf-8
import numpy as np
import pickle
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
"""
g = file("textinfo0003-0006.dump","r")
textinfo = pickle.load(g)
g.close()
g = file("thread0003-0006.dump","r")
thread = pickle.load(g)
g.close()
"""
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
    """
    length = min([len(target[target == 0]),len(target[target == 1])])
    data = np.r_[data[target == 0][0:length],data[target == 1][0:length]]
    target = np.r_[target[target == 0][0:length],target[target == 1][0:length]]
    """
    kf = KFold(len(target), n_folds=10, shuffle=True)
    vmats0 = np.array([])
    vmats1 = np.array([])
    vmats2 = np.array([])
    for train, val in kf:
        X_train, y_train = np.array(data)[train], np.array(target)[train]
        X_test, y_test = np.array(data)[val], np.array(target)[val]
        clf_cv.fit(X_train, y_train)
        y_pred = clf_cv.predict(X_test)
        vmat0 = clf_cv.coef_[0]
        vmat1 = clf_cv.coef_[1]
        vmat2 = clf_cv.coef_[2]
        if vmats0.shape[0] == 0:
            vmats0 = vmat0
            vmats1 = vmat1
            vmats2 = vmat2
        else:
            vmats0 = np.c_[vmats0,vmat0]
            vmats1 = np.c_[vmats1,vmat1]
            vmats2 = np.c_[vmats2,vmat2]
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
    return y_trueall, y_pridictall,vmats0,vmats1,vmats2

#100後から以上の動画を最初の100語を使って文類
preprocessed_docs = maketextdoc(100,10000000,0,100)
dct = gensim.corpora.Dictionary(preprocessed_docs.values())
unfiltered = dct.token2id.keys()
dct.filter_extremes(no_below=3, no_above=0.6)
filtered = dct.token2id.keys()
filtered_out = set(unfiltered) - set(filtered)
print "Save Dictionary..."
dct_txt = "id2word.txt"
dct.save_as_text(dct_txt)
print "  saved to %s\n" % dct_txt

bow_docs = {}
bow_docs_all_zeros = {}
for name in preprocessed_docs.keys():
	sparse = dct.doc2bow(preprocessed_docs[name])
	bow_docs[name] = sparse
	dense = vec2dense(sparse, num_terms=len(dct))
	bow_docs_all_zeros[name] = all(d == 0 for d in dense)
#LSI
names = preprocessed_docs.keys()
lsi_docs = {}
num_topics = 100
lsi_model = gensim.models.LsiModel(bow_docs.values(),
	id2word=dct.load_from_text('id2word.txt'),
	num_topics=num_topics)

for name in names:
	vec = bow_docs[name]
	sparse = lsi_model[vec]
	dense = vec2dense(sparse, num_topics)
	lsi_docs[name] = sparse
	print name, ":", dense
	print "\nTopics"
	print lsi_model.print_topics()

unit_vecs = {}
for name in names:
	vec = vec2dense(lsi_docs[name], num_topics)
	unit_vec = vec/np.linalg.norm(vec)
	unit_vecs[name] = unit_vec
	print name, ":", unit_vec	
l = unit_vecs.keys()


#lda
from gensim import corpora, models, similarities
dictionary = dct
#model = ldamodel.LdaModel(bow_corpus, id2word=dictionary, num_topics=100)
ldamodel = gensim.models.ldamodel.LdaModel(bow_docs.values(), id2word=dictionary, num_topics=200)
lda_docs = {}
names = preprocessed_docs.keys()
num_topics = 200
for name in names:
	vec = bow_docs[name]
	sparse = ldamodel[vec]
	dense = vec2dense(sparse, num_topics)
	lda_docs[name] = sparse
	print name, ":", dense
print "\nTopics"
print ldamodel.print_topics()
unit_vecs = {}
for name in names:
	vec = vec2dense(lsi_docs[name], num_topics)
	unit_vec = vec/np.linalg.norm(vec)
	unit_vecs[name] = unit_vec
	#print name, ":", unit_vec	
l = unit_vecs.keys()

def createtargetarray(mincomment,maxcomment,threadviewcount1,threadviewcount2,l = l):
	target2 = []
	for k in l:
		if (commentarray[k] > mincomment) & (commentarray[k] < maxcomment):
			if viewarray[k] > threadviewcount2:
				target2.append(0)
			elif viewarray[k] > threadviewcount1:
				target2.append(1)
			else:
				target2.append(2)
	return np.array(target2)

def createtvectorMat(mincomment,maxcomment,unit_vecs,l = l):
    vectorMat2 = np.array([])
    for j in l:
            if (commentarray[j] > mincomment) & (commentarray[j] < maxcomment):
                if vectorMat2.shape[0] == 0:
                    vectorMat2 = unit_vecs[j]
                else:
                    vectorMat2 = np.c_[vectorMat2,unit_vecs[j]]
    return(vectorMat2.T)

target2 = createtargetarray(500,10000000,10760.0,34544)
data2 = createtvectorMat(500,10000000,unit_vecs)
k = PredictAndAnalyze2(data = data2,target = target2)
k = PredictAndAnalyze2(data = data2,target = target2, clf_cv = svm.LinearSVC())

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
	f = file(("LogReg0" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k2[2],f)
	f.close()
	f = file(("LogReg1" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k2[3],f)
	f.close()
	f = file(("LogReg2" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k2[4],f)
	f.close()
	print "svm.SVC"
	k0 = PredictAndAnalyze2(data = data3,target = target3,clf_cv = svm.SVC(kernel='linear', probability=True))
	f = file(("svm0" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k0[2],f)
	f.close()
	f = file(("svm1" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k0[3],f)
	f.close()
	f = file(("svm2" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k0[4],f)
	f.close()
	print "linearSVC"
	k1 = PredictAndAnalyze(data = data2,target = target2,clf_cv = svm.LinearSVC())
	f = file(("limearsvm0" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k1[2],f)
	f.close()
	f = file(("linearsvm1" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k1[3],f)
	f.close()
	f = file(("linearsvm2" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k1[4],f)
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
	f = file(("LogReg0" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k2[2],f)
	f.close()
	f = file(("LogReg1" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k2[3],f)
	f.close()
	f = file(("LogReg2" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k2[4],f)
	f.close()
	print "svm.SVC"
	f = file(("svm0" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k0[2],f)
	f.close()
	f = file(("svm1" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k0[3],f)
	f.close()
	f = file(("svm2" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k0[4],f)
	f.close()
	print "linearSVC"
	k1 = PredictAndAnalyze(data = data2,target = target2,clf_cv = svm.LinearSVC())
	f = file(("limearsvm0" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k1[2],f)
	f.close()
	f = file(("linearsvm1" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k1[3],f)
	f.close()
	f = file(("linearsvm2" + str(mincount2) + str(maxcount2) + ".dump"),"w")
	pickle.dump(k1[4],f)
	f.close()

for k in [10,100,500,700,1000]:
	bagofwordsvaluate(0,k,PredictAndAnalyze = PredictAndAnalyze2)
	if k != 10:
		bagofwordsvaluate(50,k,PredictAndAnalyze = PredictAndAnalyze2)
	bagofwordsvaluate3((-1*k),0,PredictAndAnalyze = PredictAndAnalyze2)

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





    
