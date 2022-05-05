import json
import sys
import pickle
import os
import random

import networkx as nx 

class KGDatasetBuilder(object):
    def __init__(self, src_gexf_file):
        self.graph = nx.read_gexf(src_gexf_file)
        self.entity2id = {}
        self.relation2id = {}
        self.edges = []
    
    def build_entity2id(self):
        for node in self.graph.nodes:
            if node not in self.entity2id:
                self.entity2id[node] = len(self.entity2id)
    
    def build_relation2id(self):
        """
        BelongToUser
        HasDependence:
        HasDevDependence:
        RelateToNE0:
        RelateToNE1:
        RelateToNE2:
        RelateToNE3:
        RelateToNE4:
        RelateToNE5:
        HasDomain
        HasTopic
        HasKeyword
        SameToNE
        """
        self.relation2id["BelongToUser"] = 0
        self.relation2id["HasDependence"] = 1
        self.relation2id["HasDevDependence"] = 2
        self.relation2id["RelateToNE0"] = 3
        self.relation2id["RelateToNE1"] = 4
        self.relation2id["RelateToNE2"] = 5
        self.relation2id["RelateToNE3"] = 6
        self.relation2id["RelateToNE4"] = 7
        self.relation2id["RelateToNE5"] = 8

        self.relation2id["HasTopic"] = 9
        self.relation2id["HasKeyword"] = 10
        self.relation2id["SameToNE"] = 11
        pass

    def build_dataset(self, out_dir):
        self.build_entity2id()
        self.build_relation2id()
        for (u, v) in self.graph.edges:
            relation_kind = self.graph[u][v]["kind"]
            self.edges.append((self.entity2id[u], self.entity2id[v], self.relation2id[relation_kind]))
        
        with open(out_dir+"/relation2id.txt", "w", encoding="utf-8") as outf:
            outf.write(str(len(self.relation2id))+"\n")
            for key in self.relation2id:
                outf.write("{}\t{}\n".format(key, self.relation2id[key]))
        with open(out_dir+"/entity2id.txt", "w", encoding="utf-8") as outf:
            outf.write(str(len(self.entity2id))+"\n")
            for key in self.entity2id:
                outf.write("{}\t{}\n".format(key, self.entity2id[key]))
        
        total_length = len(self.edges)
        train_length = total_length // 10 * 8
        test_length = (total_length - train_length) // 2
        train_idxs = set(random.sample(range(total_length), train_length))

        with open(out_dir+"/train2id.txt", "w", encoding="utf-8") as trainf, open(out_dir+"/test2id.txt", "w", encoding="utf-8") as testf, open(out_dir+"/valid2id.txt", "w", encoding="utf-8") as validf:
            trainf.write(str(train_length)+"\n")
            testf.write(str(test_length)+"\n")
            validf.write(str(total_length - train_length - test_length)+"\n")
            for idx, (u, v, rel) in enumerate(self.edges):
                if idx in train_idxs:
                    trainf.write("{}\t{}\t{}\n".format(u, v, rel))
                else:
                    if test_length > 0:
                        testf.write("{}\t{}\t{}\n".format(u, v, rel))
                        test_length -= 1
                    else:
                        validf.write("{}\t{}\t{}\n".format(u, v, rel))
        pass

if __name__ == "__main__":
    kgd = KGDatasetBuilder("./dls.gexf")
    kgd.build_dataset(out_dir="./KG")