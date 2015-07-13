# -*- coding: utf-8 -*-
import csv
import MeCab
import re
tagger = MeCab.Tagger('-Owakati')
fo = file('fb_word_for_word2vec.txt','w')
for line in csv.reader(open("20135.csv","rU")):
      text = (line[8] + line[9])
      line = re.sub('http?://.*','', text)
      fo.write(tagger.parse(line))
fo.close()