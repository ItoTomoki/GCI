#encoding:utf-8

def minkowskiDist(v1, v2, p):
	dist = 0.0
	for i in range(len(v1)):
		dist += abs(v1[i] - v2[i])**p
	return dist**(1.0/p)

def kmeans(examples, exampleType, k, verbose):
	initialCentroids = random.sample(examples, k)

	clusters  = []
	for e in initialCentroids:
		clusters.append(Cluster([e], exampleType))


