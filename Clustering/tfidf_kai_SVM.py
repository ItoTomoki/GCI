# -*- coding: utf-8 -*-
import pandas.io.data as web
import MeCab                    # 形態素解析器MeCab
import datetime
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
CharEncoding = 'utf-8'

fileName = ("../language/20135.csv")
f = open(fileName, 'r')
reader = csv.reader(f)
#textlist = {}
textlist = []
timestamp = {}
M = 0
for row in reader:
    if len(row[8]) <> 0:
        timestamp[M] = row[5]
        text = row[9]
        #textlist.update({timestamp:text})
        textlist.append(text)
        M = M + 1
    
    if M > 100:
        break
    

f.close()


txt_num = len(textlist)
print 'total texts:', txt_num
  
fv_tf = []                      # ある文書中の単語の出現回数を格納するための配列
fv_df = {}                      # 単語の出現文書数を格納するためのディクショナリ
word_count = []                 # 単語の総出現回数を格納するための配列
  
fv_tf_idf = []                  # ある文書中の単語の特徴量を格納するための配列
tf_idf = {}
count_flag = {}                 # fv_dfを計算する上で必要なフラグを格納するためのディクショナリ
TotalwordsList = {} 
wordsCountList = {}
WordAppearCount = {}
# 各文書の形態素解析と、単語の出現回数の計算
for txt_id, txt in enumerate(textlist):
#for txt_id in textlist.keys():
    txt = textlist[txt_id]
    # MeCabを使うための初期化
    tagger = MeCab.Tagger()
    result = tagger.parse(txt)
    node = tagger.parseToNode(txt)
    keywords = []
    while node:
        surface = node.surface
        meta = node.feature.split(",")
        if meta[0] == ('名詞' or'形容詞' or '動詞' or '形容動詞'):
            keywords.append(node.surface)
        node = node.next
    #print result
    
    fv = {}  # 単語の出現回数を格納するためのディクショナリ
    
    wordList = keywords
    #wordList = result.split()[:-1:2]
    #for j in wordList:
    #    print j
    #print "========================="
                      
    words = len(wordList)                   # ある文書の単語の総出現回数
    
    for word in wordList:
        fv.setdefault(word,0)
        fv[word]+=1
    for word,count in fv.items():
        #print '%i %-16s %i' % (txt_id, word, count)
        wordsCountList.update({(txt_id,word): count})
    #print words
    TotalwordsList.update({txt_id:words})
    for i in fv.keys():
            WordAppearCount[i] = (WordAppearCount.get(i,0) + 1)
"""
for i in wordsCountList.keys():
    print i[0], i[1], wordsCountList[i]
"""

for key in wordsCountList.keys():
    for i in range(0,len(textlist)):
        if key[0] == i:
            idf = math.log(float(len(textlist)) / float(WordAppearCount.get(key[1])))
            tf = float(wordsCountList[key])/ float(TotalwordsList[key[0]])
            tf_idf.update({(i,key[1]):(idf*tf)})
tf_idf_List =  sorted(tf_idf.items(), key=lambda x: x[1],reverse = True)
WordElements = {} #単語リスト
for j in tf_idf.keys():
    #print j[0],j[1], tf_idf[j]
    WordElements[j[1]] = WordElements.get(j[1],[]) + [[j[0],tf_idf[j]]]
#列にワード，行に文章のID行列生成
tfidfArray = np.zeros([len(textlist), len(WordElements)])
#print tfidfArray.shape
m  =0
for j in WordElements.keys():
    T = WordElements[j]
    for t in T:
        #print t[0], t[1]
        tfidfArray[t[0],m]  = t[1]
    m  = m + 1
#print m
X = [[-4,-4,-1],[-2,-2,0],[1,1,2],[3,3,5]] #入力事例
y = [1,1,0,0] #クラスラベル
clf = svm.SVC(kernel='rbf') #Support Vector Classification(分類)、RBFカーネルを使用
clf.fit(X, y) #学習
clf.predict([2,2,4]) #予測
