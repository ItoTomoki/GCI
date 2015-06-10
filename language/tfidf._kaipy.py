# -*- coding: utf-8 -*-
  
import MeCab                    # 形態素解析器MeCab
  
import math
 
# 文書集合のサンプル 
text = ["ミニアルバム☆ 新谷良子withPBB「BANDScore」 絶賛発売chu♪ いつもと違い、「新谷良子withPBB」名義でのリリース！！ 全５曲で全曲新録！とてもとても濃い１枚になりましたっ。 PBBメンバーと作り上げた、新たなバンビポップ。 今回も、こだわり抜いて", "2012年11月24日 – 2012年11月24日(土)／12:30に行われる、新谷良子が出演するイベント詳細情報です。", "単語記事: 新谷良子. 編集 Tweet. 概要; 人物像; 主な ... その『ミルフィーユ・桜葉』という役は新谷良子の名前を広く認知させ、本人にも大切なものとなっている。 このころは演技も歌も素人丸出し（ ... え、普通のことしか書いてないって？ 「普通って言うなぁ！」", "2009年10月20日 – 普通におっぱいが大きい新谷良子さん』 ... 新谷良子オフィシャルblog 「はぴすま☆だいありー♪」 Powered by Ameba ... 結婚 356 名前： ノイズh(神奈川県)[sage] 投稿日：2009/10/19(月) 22:04:20.17 ID:7/ms/OLl できたっちゃ結婚か", "2010年5月30日 – この用法の「壁ドン（壁にドン）」は声優の新谷良子の発言から広まったものであり、一般的には「壁際」＋「追い詰め」「押し付け」などと表現される場合が多い。 ドンッ. 「……黙れよ」. このように、命令口調で強引に迫られるのが女性のロマンの"] 
  
txt_num = len(text)
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
for txt_id, txt in enumerate(text):
    # MeCabを使うための初期化
    tagger = MeCab.Tagger()
    result = tagger.parse(txt)
    #print result
     
    fv = {}  # 単語の出現回数を格納するためのディクショナリ
    wordList = result.split()[:-1:2]
    #print wordList                  
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
            idf = math.log(float(len(text)) / float(WordAppearCount.get(key[1])))
            tf = float(wordsCountList[key])/ float(TotalwordsList[key[0]])
            tf_idf.update({(i,key[1]):(idf*tf)})
tf_idf_List =  sorted(tf_idf.items(), key=lambda x: x[1],reverse = True)
print tf_idf_List

for j in range(0,len(text)):
    for i in tf_idf_List:
        if i[0][0] == j:
            print i[0][0], i[0][1], i[1]

#出力