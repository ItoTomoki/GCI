from gensim.models import word2vec
data = word2vec.Text8Corpus('boj/all_1006_wakati.txt')
data = word2vec.Text8Corpus('fb_word_for_word2vec.txt')
model = word2vec.Word2Vec(data, size=50)
p = raw_input().decode("utf-8")
n = raw_input().decode("utf-8")

out = model.most_similar(positive=[u'金融'])
for x in out:print x[0],x[1]
out = model.most_similar(positive=[x],negative=[y])
for x in out:print x[0],x[1]
