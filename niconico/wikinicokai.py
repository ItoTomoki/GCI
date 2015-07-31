#encoding:utf-8
from gensim.models import word2vec
import numpy as np

#modelwiki = word2vec.Word2Vec.load("wiki1-4.model")
#data = word2vec.Text8Corpus('allcomment.txt')
#modelnico = word2vec.Word2Vec(data, size=50)
#modelnico = word2vec.Word2Vec.load("comennt.model")
modelnico = word2vec.Word2Vec.load("comment2.model")
model = modelnico
"""
word = [u"腹筋"]
out2 = modelnico.most_similar(positive= word)
wordarray = []
for j in out2:
    wordarray.append(j[0])
    print j[0]
"""
#wikivoc  = modelwiki.vocab.keys()
nicovoc = modelnico.vocab.keys()
#nicowikivoc = list(set(nicovoc) & set(wikivoc))

#共通するベクトルの抽出
"""
cowordlist = {}
for j in nicowikivoc:
    for i in nicowikivoc:
        if (modelnico.similarity(i, j) - modelwiki.similarity(i,j)) < 0.1:
            #cowordlist[[i,j]] = 0.50 * (modelnico.similarity(i, j) + modelwiki.similarity(i,j))
            print i,j
cowordlist = {}
for j in nicowikivoc:
    for i in nicowikivoc:
        if ((modelnico.similarity(i, j) > 0.75) & (modelwiki.similarity(i,j) > 0.75) & (i != j)):
            if cowordlist.has_key(j):
                cowordlist[j].append(i)
            else:
                cowordlist[j] = []
                cowordlist[j].append(j)
    print j

cowordvoc = cowordlist.keys()
"""
#方法１
def henkan(word):
    wordarray = []
    wordremainarray = []
    wordremainarray2 = []
    out2 = modelnico.most_similar(positive= word) 
    for j in out2:
        wordarray.append(j[0])
        print j[0]
    print "======================="
    nicoword  = list(set(wordarray) & set(nicowikivoc))
    remainvoc = list(set(wordarray) - set(nicoword))
    for j in remainvoc:
        #wordremainarray2.append(henkan(j))
        out2 = modelnico.most_similar(positive= remainvoc)
        for j in out2:
            wordremainarray.append(j)
        wordremainarray2 = list(set(wordremainarray) & set(wikivoc))[0:len(wordremainarray)]
        for j in wordremainarray2: print j
    henkanword = modelwiki.most_similar(positive=(list(nicoword) + list(wordremainarray2)))
    #return henkanword[0][0]
    return henkanword
#方法2
def henkan2(word,nicowikivoc):
    wordarray2 = []
    for j in nicowikivoc:
        if modelnico.similarity(word, j) > 0.8:
            wordarray2.append(j)
            print j
    if len(wordarray2) < 5:
        print "=============0.75============"
        wordarray2 = []
        for j in nicowikivoc:
            if modelnico.similarity(word, j) > 0.75:
                wordarray2.append(j)
                #print j   
    if len(wordarray2) < 5:
        print "=============0.7============"
        wordarray2 = []
        for j in nicowikivoc:
            if modelnico.similarity(word, j) > 0.7:
                wordarray2.append(j)
                #print j
    if len(wordarray2) < 5:
        print "=============0.6============"
        wordarray2 = []
        for j in nicowikivoc:
            if modelnico.similarity(word, j) > 0.6:
                wordarray2.append(j)
                #print j
    return wordarray2

def henkan3(word):
    wordarray = []
    wordremainarray = []
    wordremainarray2 = []
    out2 = modelnico.most_similar(positive= word)
    out = henkan2(word,cowordvoc)
    for j in out2:
        wordarray.append(j[0])
        print j[0]
    print "======================="
    nicoword  = list(set(wordarray) & set(cowordvoc))
    remainvoc = list(set(wordarray) - set(nicoword))
    for j in remainvoc:
        #wordremainarray2.append(henkan(j))
        out2 = modelnico.most_similar(positive= remainvoc)
        for j in out2:
            wordremainarray.append(j)
        if len(wordremainarray) > 0:
            wordremainarray2 = list(set(wordremainarray) & set(cowordvoc))[0:len(wordremainarray)]
            for j in wordremainarray2: print j
    try:
        henkanword = modelwiki.most_similar(positive=(list(nicoword) + list(wordremainarray2)))
    except:
        henkanword = modelwiki.most_similar(positive= out)
        for x in out:print x
    #return henkanword[0][0]
    return henkanword
#レコメンド絡み
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
        if node.feature.split(",")[0] == "名詞":
            word2freq[node.surface] += 1
        node = node.next
    return word2freq
def output(word2freq):
    for word, freq in sorted(word2freq.items(),key = lambda x: x[1], reverse=True):
        print str(freq), word
def makevec(word2freq):
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    for word, freq in sorted(word2freq.items(),key = lambda x: x[1], reverse=True):
        if int(freq) > 10:
            v += freq * wordvec(word.decode("utf-8"))
    return v/np.linalg.norm(v)

filename = "comment2/sm83.txt" #"sm9922.txt"
def createvector(video_id):
    filename = ("comment2/" + str(video_id) + ".txt")
    f = open(filename)
    data = f.read()
    f.close()
    v = makevec(morphological_analysis(data))
    return v
vectorinfo = {}


import json
import os
from ast import literal_eval
import re
#f = open('data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/0000/sm449.dat')
files = os.listdir('data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/video')
textinfo = {}
for file in files[1:2]:
    #print file
    file = "0000.dat"
    filepass = 'data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/video/' + str(file)
    f = open(filepass)
    lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
    data1 = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()
    #print type(data1) # 文字列データ
    Lines2 = {}
    #lines1 = data1.split('n') # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
    #print type(lines2)
    count = 0
    for line in lines2:
        Lines2[count] = literal_eval(line)
        #print Lines2[count]["video_id"], Lines2[count]["title"].decode('unicode_escape')
        textinfo[Lines2[count]["video_id"]] = Lines2[count]["title"].decode('unicode_escape')
        count += 1

for j in textinfo.keys():
    #print j
    vectorinfo[j] = createvector(j)
#print textinfo["sm3068"]
def selectTitleFromID(video_id):
    titlerank = {}
    for j in vectorinfo.keys():
        if np.dot(vectorinfo[j], vectorinfo[video_id]) > 0.8:
            titlerank[j] = np.dot(vectorinfo[j], vectorinfo[video_id])
    k = sorted(titlerank.items(), key=lambda x: x[1],reverse = True)
    for m in k[0:20]:
        print textinfo[m[0]], m[0], m[1]
        #print j, textinfo[j], np.dot(vectorinfo[j], vectorinfo[video_id])
def selectTitleFromWord(wordarray):
    v = np.zeros(len(model[model.vocab.keys()[0]]))
    for j in wordarray:
        v += wordvec(j)
    v = v/np.linalg.norm(v)
    titlerank = {}
    for j in vectorinfo.keys():  
        if np.dot(v, vectorinfo[j]) > 0.5:
            titlerank[j] = np.dot(v, vectorinfo[j])
    k = sorted(titlerank.items(), key=lambda x: x[1],reverse = True)
    for m in k[0:20]:
        print textinfo[m[0]], m[0], m[1]

def jikkou():
    print "Please input [selectTitleFromID] or [selectTitleFromWord] or [showTitleAndID]:(if you want to quit, please input”やめる” )"
    x = raw_input().decode("utf-8")
    #x = sys.stdin.read()
    if x.encode("utf-8") == "[showTitleAndID]":
        for j in textinfo.keys():
            print j, textinfo[j]
    if x.encode("utf-8") == "[selectTitleFromID]":
        print x
        print "Please input ID"
        y = raw_input().decode("utf-8")
        selectTitleFromID(y.encode("utf-8"))
    if x.encode("utf-8") == "[selectTitleFromWord]":
        inputwordarray = []
        print x
        print "Please Words(if you want to quit, input 'stop')"
        y = ''
        while y.encode("utf-8") != "stop":
            y = raw_input().decode("utf-8")
            inputwordarray.append(y)
        for j in inputwordarray:
            print j
        selectTitleFromWord(inputwordarray)
    if x.encode("utf-8") == "やめる":
        print "Bye"
    else:
        print x
        jikkou()

jikkou()  


#np.dot(k,wordvec(u"おっくせんまん"))
        
