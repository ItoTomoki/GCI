# -*- coding: utf-8 -*-
import csv
import MeCab
import re
import unicodedata
import sys
#tagger = MeCab.Tagger('-Owakati')
tagger = MeCab.Tagger('-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic,/usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/hatena-keyword.dic') 
fo = file('word_for_word2vec_2013_kai.txt','w')
#fo = file('fb_word_for_word2vec.txt','w')
for line in csv.reader(open("2013.csv","rU")):
      text = (line[8] + line[9])
      #text = line[8]
      try:
      	text = unicodedata.normalize('NFKC', text.decode("utf-8"))
      except:
      	print "normalize Eroor"
      #line = re.sub('http?://.*','', text)
      text = re.sub(re.compile("[!-/:-@[-`{-~]"), '', text.encode("utf-8"))
      text = text.replace("\n", "")
      text = text.replace("\t", "")
      text = text.replace("n", "")
      text = text.replace(" ", "")
      text = text.replace("　", "")
      text = re.sub(re.compile("[!-/:-@[-`{-~]"), '', text)
      if text != '':
      	fo.write(tagger.parse(text))
      	fo.write('\n')
fo.close()
