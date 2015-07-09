#coding: utf8
import networkx as nx

sample = {}
sample["A"] = {}
sample["B"] = {}
sample["C"] = {}
sample["A"]["sex"] = "Male"
sample["B"]["sex"] = "Female"
sample["C"]["sex"] = "Male"
sample["A"]["scores"] = {}
sample["B"]["scores"] = {}
sample["C"]["scores"] = {}
sample["A"]["scores"]["B"] = 1.0
sample["B"]["scores"]["A"] = 1.0
sample["B"]["scores"]["C"] = 1.0
sample["C"]["scores"]["A"] = 1.0

G = nx.MultiDiGraph()
for person in sample:
        G.add_node(person,sex=sample[person]["sex"])
        for person1 in sample:
                for person2 in sample[person1]["scores"]:
                            G.add_edge(person1,person2,weight=sample[person1]["scores"][person2])
                            nx.write_gexf(G,"sample.gexf")
