# -*- coding: utf-8 -*-
import csv
import MeCab
import re
#tagger = MeCab.Tagger('-Owakati')
tagger = MeCab.Tagger('-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic,/usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/hatena-keyword.dic') 
fo = file('word_for_word2vec_2013.txt','w')
#fo = file('fb_word_for_word2vec.txt','w')
for line in csv.reader(open("2013.csv","rU")):
      text = (line[8] + line[9])
      line = re.sub('http?://.*','', text)
      fo.write(tagger.parse(line))
fo.close()
