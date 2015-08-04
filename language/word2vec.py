#encoding:utf8
from gensim.models import word2vec
import numpy as np
data = word2vec.Text8Corpus('word_for_word2vec_2013_kai.txt')
model = word2vec.Word2Vec(data, size=50)
model.save("2013_kai.model")
model = word2vec.Word2Vec.load("2013_kai.model")
voc = model.vocab.keys()
#vector = model[voc[1]]
#model.similarity(u"銀行", u"みずほ")
print "Please input positive words.If you complete, input 0"
p = np.array([])
posinput = 1
while posinput != "0":
    posinput = raw_input().decode("utf-8")
    p = np.hstack([p,posinput])
print "Please input negative words.If you complete, input 0"
n = []
neginput = 1
while neginput != "0":
    neginput = raw_input().decode("utf-8")
    n = np.hstack([n,neginput])
#out = model.most_similar(positive=[u'銀行'])
#for x in out:print x[0],x[1]
try:
    out = model.most_similar(positive=p,negative=n)
except:
    print "Eroor"    
for x in out:print x[0],x[1]

f = open('word_for_word2vec_2013_kai.txt')
data1 = f.read()
newstexts = data1.split('\n')
lines2 = f.readlines()


def wordvec(word,model = model):
    try:
        v = model[word]/np.linalg.norm(model[word])
        return v
    except:
        return np.zeros(len(model[model.vocab.keys()[0]]))

import sys
import MeCab
from collections import defaultdict
def morphological_analysis(text):
    word2freq = defaultdict(int)
    mecab = MeCab.Tagger('-u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic')
    node = mecab.parseToNode(text)
    while node:
    	#if node.feature.split(",")[0] == "名詞":
        if (node.feature.split(",")[0] == "名詞") | (node.feature.split(",")[0] == "形容動詞") | (node.feature.split(",")[0] == "形容詞"):
            word2freq[node.surface] += 1
        node = node.next
    return word2freq
def output(word2freq):
    for word, freq in sorted(word2freq.items(),key = lambda x: x[1], reverse=True):
        print str(freq), word

def makevec(word2freq):
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    for word, freq in sorted(word2freq.items(),key = lambda x: x[1], reverse=True):
        if int(freq) > 1:
            v += freq * wordvec(word.decode("utf-8"))
    if np.linalg.norm(v) > 0:
    	return v/np.linalg.norm(v)
    else:
    	return v

def selectTitleFromID(video_id):
    titlerank = {}
    for j in textDic.keys():
        if np.dot(textDic[j]["vec"], textDic[video_id]["vec"]) > 0.8:
            titlerank[j] = np.dot(textDic[j]["vec"], textDic[video_id]["vec"])
    k = sorted(titlerank.items(), key=lambda x: x[1],reverse = True)
    for m in k[0:20]:
        print textDic[m[0]]["info"][7], m[0], m[1]

def selectTitleFromWord(wordarray):
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    for j in wordarray:
        v += wordvec(j)
    v = v/np.linalg.norm(v)
    titlerank = {}
    for j in textDic.keys():
        if np.dot(textDic[j]["vec"],v) > 0.5:
            titlerank[j] = np.dot(textDic[j]["vec"],v)
    k = sorted(titlerank.items(), key=lambda x: x[1],reverse = True)
    for m in k[0:20]:
        print textDic[m[0]]["info"][7], textDic[m[0]]["info"][0],textDic[m[0]]["info"][1], m[1]
np.dot(makevec(morphological_analysis(lines2[2])),makevec(morphological_analysis(lines2[10])))
textDic = {}
c = 0
for line in csv.reader(open("2013.csv","rU")):
	textDic[c] = {}
	textDic[c]["info"] = line
	#k[c]["vec"] = makevec(morphological_analysis(newstexts[2*c]))
	textDic[c]["vec"] = makevec(morphological_analysis(textDic[c]["info"][8]))
	textDic[c]["vec"] += makevec(morphological_analysis(textDic[c]["info"][9]))
	textDic[c]["vec"] = textDic[c]["vec"]/np.linalg.norm(textDic[c]["vec"])
	c += 1
	#print c
print textDic[1][7]


