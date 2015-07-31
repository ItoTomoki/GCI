#encoding:utf8
from gensim.models import word2vec
import numpy as np
data = word2vec.Text8Corpus('fb_word_for_word2vec.txt')
#model = word2vec.Word2Vec(data, size=50)
#model.save("2013.model")
model = word2vec.Word2Vec.load("2013.model")
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
