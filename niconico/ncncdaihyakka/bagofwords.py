#encoding-utf-8
def maketexts(ID,video_id,mincommentlines,maxcommentlines):
    #filename = ("comment2_" + ID + "/" + str(video_id) + ".txt")
    #filename = ("comment2_kai" + ID + "/" + str(video_id) + ".txt")
    filename = ("comment_kai" + ID + "/" + str(video_id) + ".txt")
    f = open(filename)
    text = f.read()
    f.close()
    datalines = text.split('\n')
    text = ""
    for n in range(mincommentlines,maxcommentlines):
        try:
            text += (datalines[n] + " ")
        except:
            #print len(datalines), ID,video_id
            break
    return text.decode("utf-8")

docs = {}
def maketextdic(mincoumment,maxcomment,mincommentlines,maxcommentlines):
	docs = {}
	for ID in ["0000","0001","0002","0003"]:
		for video_id in textinfo[ID].keys():
			if (thread[ID][(str(video_id) + ".dat")]["comment_counter"] > mincoumment) & (thread[ID][(str(video_id) + ".dat")]["comment_counter"] < maxcomment):
				try:
					docs[video_id] = maketexts(ID,video_id,mincommentlines,maxcommentlines)
				except:
					print ID,j
	return docs
import gensim

docs = 	maketextdic(0,10000000,0,100)
preprocessed_docs = {}
for name in docs.keys():
	preprocessed = gensim.parsing.preprocess_string(docs[name])
	preprocessed_docs[name] = preprocessed
	print name, ":", preprocessed






dct = gensim.corpora.Dictionary(docs.values())
unfiltered = dct.token2id.keys()
dct.filter_extremes(no_below=3, no_above=0.6)
filtered = dct.token2id.keys()
filtered_out = set(unfiltered) - set(filtered)