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

class bicluster:
  def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
    self.left = left
    self.right = right
    self.vec = np.array(vec)
    self.id = id
    self.distance = distance

def EuclidDistance(p,q):
    return math.sqrt(sum((p-q)**2))

def pearson(p1,p2):
    # calculate p's variance
    sum1 = sum(p1)
    sum2 = sum(p2)
 
    sum1Sq = sum([pow(p,2) for p in p1])
    sum2Sq = sum([pow(p,2) for p in p2])
 
    pSum = sum([p1[i]*p2[i] for i in range(len(p1))])
 
    num = pSum-(sum1*sum2/len(p1))
    den=math.sqrt((sum1Sq-pow(sum1,2)/len(p1))*(sum2Sq-pow(sum2,2)/len(p1)))
 
    if den == 0: return 0
    return 1.0 - num/den

def hcluster(rows, distance=EuclidDistance):
    distances = {}
    currentclustid = -1
 
    clust = [bicluster(tfidfArray[i], id=i) for i in range(len(rows))]
 
    while len(clust) > 1:
        lowestpair = (0,1)
        closest = distance(clust[0].vec, clust[1].vec)
 
        # Get lowest cluster
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
 
                d = distances[(clust[i].id, clust[j].id)]
 
                if d < closest:
                    closest = d
                    lowestpair = (i,j)
        # Calculate average cluster
        #mergevec = [(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]
        mergevec = (clust[lowestpair[0]].vec + clust[lowestpair[0]].vec)/2
        # Make new clusterふたつまとめたclusterを作る
        #print mergevec
        #print clust[lowestpair[0]]
        #print clust[lowestpair[1]]
        #print closest
        newcluster = bicluster(vec = mergevec, left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance=closest, id=currentclustid)
 
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
 
    return clust[0]

def printclust(clust, labels=None, n=0):
    for i in range(n): print ' ',
    if clust.id < 0:
        print '-'
    else:
        if labels == None:
            print clust.id
        else:
            print labels[clust.id]
 
    if clust.left != None: 
        printclust(clust.left, labels=labels, n=n+1)
    if clust.right != None: 
        printclust(clust.right, labels=labels, n=n+1)
"""
if __name__ == '__main__':
    clust=hcluster(tfidfArray)
    printclust(clust, labels=None)
"""

from PIL import Image,ImageDraw
#import clusters
 
#
# Calculate heights of the dendrogram
#
def getheight(clust):
    if clust.left == None and clust.right == None: 
        return 1
    return getheight(clust.left) + getheight(clust.right)
 
#
# Calculate distance with depth
#
def getdepth(clust):
    if clust.left == None and clust.right == None: 
        return 0
    return max(getdepth(clust.left),getdepth(clust.right)) + clust.distance
 
#
# Draw whole dendrogram
#
def drawdendrogram(clust, labels, png='clusters.png'):
    h = getheight(clust) * 20
    w = 1200
    depth = getdepth(clust)
 
    scaling=float(w-150)/depth
 
    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)
    draw.line((0, h/2, 10, h/2), fill=(255,0,0))
 
    drawnode(draw, clust, 10, (h/2), scaling, labels)
    img.save(png, 'PNG')
 
#
# Draw a node
#
def drawnode(draw, clust, x, y, scaling, labels):
    if clust.id < 0:
        h1 = getheight(clust.left) * 20
        h2 = getheight(clust.right) * 20
        top = y-(h1+h2)/2
        bottom = y+(h1+h2)/2
 
        ll = clust.distance * scaling
 
        draw.line((x, top+h1/2, x, bottom-h2/2), fill=(255,0,0))
 
        draw.line((x, top+h1/2, x+ll, top+h1/2), fill=(255,0,0))
 
        draw.line((x, bottom-h2/2, x+ll, bottom-h2/2), fill=(255,0,0))
 
        drawnode(draw, clust.left, x+ll, top+h1/2, scaling, labels)
        drawnode(draw, clust.right, x+ll, bottom-h2/2, scaling, labels)
    else:
        draw.text((x+5,y-7), str(labels[clust.id]), (0,0,0))

def kcluster(rows, distance=EuclidDistance, k=4):
    # Make k clusters with random
    ranges=[(min([row[i] for row in rows]), max([row[i] for row in rows])) for i in range(len(rows[0]))]
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]
 
    lastmatches = None
    for t in range(100): #最大100回繰り返す
        print 'Iteration %d' % t
        bestmatches=[[] for i in range(k)]
 
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)
 
        if bestmatches == lastmatches: #変化しなくなったらおわり
            break
        lastmatches = bestmatches 
 
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs
    return bestmatches
 
if __name__ == '__main__':
    kclust = kcluster(tfidfArray, k=5)
    labels = range(0,len(textlist))
    print kclust
    #print [labels[r] for r in kclust[0]]
    clust=hcluster(tfidfArray)
    labels = range(0,len(textlist))
    drawdendrogram(clust,labels,png='newsclust.png')
    im = Image.open("newsclust.png")
    im.show()



"""
if __name__ == '__main__':
        #blognames,words,data=clusters.readfile('blogdata.txt')
    clust=hcluster(tfidfArray)
    labels = range(0,len(textlist))
    drawdendrogram(clust,labels,png='blogclust.png')
    im = Image.open("blogclust.png")
    im.show()
"""

"""
k_means= KMeans(init='random', n_clusters=10) #init：初期化手法、n_clusters=クラスタ数を指定
k_means.fit(tfidfArray)
Y=k_means.labels_  #得られた各要素のラベル
#print textlist[1]
"""
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




