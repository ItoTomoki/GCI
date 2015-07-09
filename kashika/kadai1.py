#encoding:utf8
import mysql.connector
import MySQLdb
import numpy as np
import codecs
import sys
import networkx as nx
import matplotlib.pyplot as plt
#sys.stdout = codecs.getwriter("utf8")(sys.stdout)
#sys.stdin = codecs.getreader("sutf8")(sys.stdin)
LISt = {}
connector = MySQLdb.connect(host="localhost", db="Restaourant", user="root",passwd="", charset="utf8")
cursor = connector.cursor()
sql = "SELECT * from chefmozaccepts;"
cursor.execute(sql)
PayList = cursor.fetchall()
for j in PayList:
	LISt[int(j[0])] = {}
	LISt[int(j[0])]["pay"] = j[1]

sql = "SELECT * from chefmozcuisine;"
cursor.execute(sql)
cuizineList = cursor.fetchall()
for j in cuizineList:
	try:
		LISt[int(j[0])]["cuizine"] = j[1]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["cuizine"] = j[1]

sql = "SELECT * from chefmozparking;"
cursor.execute(sql)
ParkingList = cursor.fetchall()
for j in ParkingList:
	try:
		LISt[int(j[0])]["parking"] = j[1]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["parking"] = j[1]

sql = "SELECT * from geoplaces2;"
cursor.execute(sql)
InfoList = cursor.fetchall()
for j in InfoList:
	try:
		LISt[int(j[0])]["name"] = j[4]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["name"] = j[4]

	try:
		LISt[int(j[0])]["city"] = j[6]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["city"] = j[6]
	try:
		LISt[int(j[0])]["country"] = j[8]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["country"] = j[8]
	try:
		LISt[int(j[0])]["alcohol"] = j[11]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["alcohol"] = j[11]
	try:
		LISt[int(j[0])]["smoking"] = j[12]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["smoking"] = j[12]
	try:
		LISt[int(j[0])]["dress_code"] = j[13]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["dress_code"] = j[13]
	try:
		LISt[int(j[0])]["accessibility"] = j[14]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["accessibility"] = j[14]
	try:
		LISt[int(j[0])]["price"] = j[15]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["price"] = j[15]
	try:
		LISt[int(j[0])]["Rambience"] = j[17]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["Rambience"] = j[17]
	try:
		LISt[int(j[0])]["franchise"] = j[18]
	except:
		LISt[int(j[0])] = {}
		LISt[int(j[0])]["franchise"] = j[18]

def Listdistance(i,j):
	d = 0
	for key in LISt[i].keys():
		try:
			if LISt[i][key] == LISt[j][key]:
				d += 1
		except:
			d  = d
	return d

G = nx.MultiDiGraph()
Rate = {}
labels = {}
for key1 in LISt.keys():
	sql = "select rating from rating_final where placeID = "
	sql += str(key1)
	sql += ";"
	try:
		cursor.execute(sql)
		Rate= cursor.fetchall()
	except:
		#print sql
		Rate = 0
	m = 0
	for i in Rate:
		m += int(i[0])
	if len(Rate) > 0:
		Rate = int(m/len(Rate))
		print Rate
		labels[key1] = Rate
		G.add_node(key1,rate=Rate)
	else:
		Rate = "None"
	#labels[key1] = Rate
	#G.add_node(key1,rate=Rate)
keyall = LISt.keys()
for i in range(0,len(keyall)):
	for j in range(i,len(keyall)):
#for i in range(0,50):
	#for j in range(i,50):
		key1 = int(keyall[i])
		key2 = int(keyall[j])
		d = Listdistance(key1,key2)
		#print i,j
		if d > 0:
			#print int(key1),int(key2),d
			try:
				G.add_edge(int(key1),int(key2),weight=d)
			except:
				continue
print G.nodes()
pos = nx.spring_layout(G)
#print labels.keys() # グラフ形式を選択。ここではスプリングモデルでやってみる
nx.draw_networkx_nodes(G, pos, with_labels=False)
nx.draw_networkx_labels(G,pos,labels,font_size=16) # グラフ描画。 オプションでノードのラベル付きにしている
plt.savefig("labels_node.png")
nx.draw(G, pos, with_labels=False)
nx.draw_networkx_labels(G,pos,labels,font_size=16) # グラフ描画。 オプションでノードのラベル付きにしている
plt.savefig("labels_node_edge.png")
plt.show() # matplotlibでグラフ表示
#nx.write_gexf(G,"sample3.gexf")






