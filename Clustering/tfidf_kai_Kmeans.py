# -*- coding: utf-8 -*-
import pandas.io.data as web
import MeCab                    # 形態素解析器MeCab
import datetime
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
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
    
    if M > 500:
        break
    

f.close()

#print textlist
# 文書集合のサンプル 
#textlist = ["ミニアルバム☆ 新谷良子withPBB「BANDScore」 絶賛発売chu♪ いつもと違い、「新谷良子withPBB」名義でのリリース！！ 全５曲で全曲新録！とてもとても濃い１枚になりましたっ。 PBBメンバーと作り上げた、新たなバンビポップ。 今回も、こだわり抜いて", "2012年11月24日 – 2012年11月24日(土)／12:30に行われる、新谷良子が出演するイベント詳細情報です。", "単語記事: 新谷良子. 編集 Tweet. 概要; 人物像; 主な ... その『ミルフィーユ・桜葉』という役は新谷良子の名前を広く認知させ、本人にも大切なものとなっている。 このころは演技も歌も素人丸出し（ ... え、普通のことしか書いてないって？ 「普通って言うなぁ！」", "2009年10月20日 – 普通におっぱいが大きい新谷良子さん』 ... 新谷良子オフィシャルblog 「はぴすま☆だいありー♪」 Powered by Ameba ... 結婚 356 名前： ノイズh(神奈川県)[sage] 投稿日：2009/10/19(月) 22:04:20.17 ID:7/ms/OLl できたっちゃ結婚か", "2010年5月30日 – この用法の「壁ドン（壁にドン）」は声優の新谷良子の発言から広まったものであり、一般的には「壁際」＋「追い詰め」「押し付け」などと表現される場合が多い。 ドンッ. 「……黙れよ」. このように、命令口調で強引に迫られるのが女性のロマンの"] 
  
#txt_num = len(textlist)
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
#列にワード，行に文章の行列生成
tfidfArray = np.zeros([len(textlist), len(WordElements)])
print tfidfArray.shape
m  =0
for j in WordElements.keys():
    T = WordElements[j]
    for t in T:
        #print t[0], t[1]
        tfidfArray[t[0],m]  = t[1]
    m  = m + 1
#print m

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
"""
p= pdist(tfidfArray, metric="euclidean") #ユークリッド距離を採用する
Z= linkage(p, method="single") #最小最短距離法をmethodで指定する

dendrogram(Z)

plt.show()
"""
"""
k_means= KMeans(init='random', n_clusters=10) #init：初期化手法、n_clusters=クラスタ数を指定
k_means.fit(tfidfArray)
Y=k_means.labels_  #得られた各要素のラベル
"""
import random
#print textlist[1]
def minkowskiDist(v1, v2, p):
    if len(v1) == 0 & len(v1) == 0:
        return 0.0
    else:
        dist = 0.0
        for i in range(len(v1)):
            dist += abs(np.array(v1[i]) - np.array(v2[i]))**p
        return dist**(1.0/p)

def getCentroid(M):
    M = np.array(M)
    m = M.sum(axis = 0)
    m = m/len(M)
    return m

def kmeans(tfidfArray, k, verbose = False):

    initialCentroids = random.sample(tfidfArray, k)

    clusters  = []
    Centroid = []
    for e in initialCentroids:
        clusters.append([e])
    converged = False
    numIterations = 0
    while not converged:
        numIterations += 1
        for i in range(k):
            Centroid.append([])
        for i in range(0,k):
            Centroid[i] = getCentroid(clusters[i])
            print len(Centroid[i])
        ClusterIteration = []
        for i in range(k):
            ClusterIteration.append([])
        newClusters = []
        for i in range(k):
            newClusters.append([])

        number = 0
        for e in tfidfArray:
            try:
                smallestDistance = minkowskiDist(e,Centroid[0],2)
            except:
                distance = 1000000
            index = 0
            for i in range(1,k):
                try:
                    distance = minkowskiDist(e,Centroid[i],2)
                except:
                    distance = 1000000
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            newClusters[index].append(e)
            ClusterIteration[index].append(number)
            number += 1
        converged = True
        for i in range(len(clusters)):
            print sum(abs((getCentroid(newClusters[i]) - getCentroid(clusters[i]))))
            if (len(newClusters[i]) > 0) | (len(clusters[i]) > 0):
                if sum(abs((getCentroid(newClusters[i]) - getCentroid(clusters[i])))) > 0.001:
                    converged = False
        print "======================="
        clusters = newClusters
        if verbose:
            print "Iteration #" + str(numIterations)
            for c in clusters:
                print c
            print ''
    return ClusterIteration

clusters = kmeans(tfidfArray, 8)
print clusters[0]
print clusters[1]
print clusters[2]
print clusters[3]
for y in clusters[0]:
    m = 0
    for i in tf_idf_List:
        if i[0][0] == y:
            print i[0][0],timestamp[y], i[0][1], i[1]
            m = m + 1
            if m > 5:
                print "======================="
                break
#for i in tf_idf_List:


"""
print len(Y), len(textlist)
print len(Y[Y == 1])
for y in range(0,len(Y)):
    if Y[y] == 1:
        #print textlist[y]
        #print "==================="

        m = 0
        for i in tf_idf_List:
            if i[0][0] == y:
                print i[0][0],timestamp[y], i[0][1], i[1]
                m = m + 1
            if m > 10:
                print "======================="
                break
"""




