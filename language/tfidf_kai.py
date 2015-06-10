# -*- coding: utf-8 -*-
import pandas.io.data as web
import MeCab                    # 形態素解析器MeCab
import datetime
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
CharEncoding = 'utf-8'
fileName = ("20135.csv")
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
    """
    if M > 100:
        break
    """

f.close()
#print textlist
# 文書集合のサンプル 
#text = ["ミニアルバム☆ 新谷良子withPBB「BANDScore」 絶賛発売chu♪ いつもと違い、「新谷良子withPBB」名義でのリリース！！ 全５曲で全曲新録！とてもとても濃い１枚になりましたっ。 PBBメンバーと作り上げた、新たなバンビポップ。 今回も、こだわり抜いて", "2012年11月24日 – 2012年11月24日(土)／12:30に行われる、新谷良子が出演するイベント詳細情報です。", "単語記事: 新谷良子. 編集 Tweet. 概要; 人物像; 主な ... その『ミルフィーユ・桜葉』という役は新谷良子の名前を広く認知させ、本人にも大切なものとなっている。 このころは演技も歌も素人丸出し（ ... え、普通のことしか書いてないって？ 「普通って言うなぁ！」", "2009年10月20日 – 普通におっぱいが大きい新谷良子さん』 ... 新谷良子オフィシャルblog 「はぴすま☆だいありー♪」 Powered by Ameba ... 結婚 356 名前： ノイズh(神奈川県)[sage] 投稿日：2009/10/19(月) 22:04:20.17 ID:7/ms/OLl できたっちゃ結婚か", "2010年5月30日 – この用法の「壁ドン（壁にドン）」は声優の新谷良子の発言から広まったものであり、一般的には「壁際」＋「追い詰め」「押し付け」などと表現される場合が多い。 ドンッ. 「……黙れよ」. このように、命令口調で強引に迫られるのが女性のロマンの"] 
  
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
    for i in range(0,len(text)):
        if key[0] == i:
            idf = math.log(float(len(textlist)) / float(WordAppearCount.get(key[1])))
            tf = float(wordsCountList[key])/ float(TotalwordsList[key[0]])
            tf_idf.update({(i,key[1]):(idf*tf)})
tf_idf_List =  sorted(tf_idf.items(), key=lambda x: x[1],reverse = True)
#print tf_idf_List
fDate = []
fP = []
f = {}
for j in range(0,len(text)):
    m = 0
    for i in tf_idf_List:
        """
        if i[0][0] == j:
            print timestamp[j], i[0][1], i[1]
            m = m + 1
            if m > 10:
                print "======================="
                break
        """
        if i[0][0] == j:
            if i[0][1] == "インフレ":#"国債":
                tstr = timestamp[i[0][0]]
                tdatetime = datetime.datetime.strptime(tstr, '%Y-%m-%d %H:%M:%S')
                f.update({tdatetime: i[1]})
fDate = np.sort(f.keys())
for k in fDate:
    fP.append(f[k])
    print k
plt.title('inflation of tfidfs from 2013.5.1 to 2013.5.31')
plt.plot(fDate, fP)
plt.show()
#出力