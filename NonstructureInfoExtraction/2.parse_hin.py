import os
import sys
import json
import re

import networkx as nx

class HINParser(object):
    def __init__(self, hin_output, repoidx2reponame, out_file):
        repo2vec = dict()
        name2vec = dict()
        keyword2vec = dict()
        topic2vec = dict()
        word2vec = dict()
        user2vec = dict()
        domain2vec = dict()
        with open(repoidx2reponame, "r", encoding="utf-8") as inf:
            idx2repo = json.load(inf)
        with open(hin_output, "r", encoding="utf-8") as inf:
            next(inf)
            for line in inf:
                splits = line.strip().split(" ")
                if splits[1] == "r":
                    repo = idx2repo[splits[0][5:]]
                    repo2vec[repo] = self.get_vec(splits)
                elif splits[1] == "n":
                    name = self.recover_blankspace(splits[0][5:])
                    name2vec[name] = self.get_vec(splits)
                elif splits[1] == "k":
                    keyword = self.recover_blankspace(splits[0][8:])
                    keyword2vec[keyword] = self.get_vec(splits)
                elif splits[1] == "t":
                    topic = self.recover_blankspace(splits[0][6:])
                    topic2vec[topic] = self.get_vec(splits)
                elif splits[1] == "w":
                    word = self.recover_blankspace(splits[0][5:])
                    word2vec[word] = self.get_vec(splits)
                elif splits[1] == "u":
                    user = self.recover_blankspace(splits[0][5:])
                    user2vec[user] = self.get_vec(splits)
                elif splits[1] == "d":
                    domain = self.recover_blankspace(splits[0][7:])
                    domain2vec[domain] = self.get_vec(splits)
        with open(out_file, "w", encoding="utf-8") as outf:
            outf.write(json.dumps(user2vec)+"\n")
            outf.write(json.dumps(repo2vec)+"\n")
            outf.write(json.dumps(name2vec)+"\n")
            outf.write(json.dumps(topic2vec)+"\n")
            outf.write(json.dumps(keyword2vec)+"\n")
            outf.write(json.dumps(domain2vec)+"\n")
            outf.write(json.dumps(word2vec)+"\n")
    
    def get_vec(self, splits):
        return [float(item) for item in splits[2:]]
    
    def recover_blankspace(self, text):
        return re.sub(r"%", " ", text)

class HINRecover(object):
    def __init__(self, link_file, repoidx2reponame, out_file):
        self.link_file = link_file
        with open(repoidx2reponame, "r", encoding="utf-8") as inf:
            self.idx2repo = json.load(inf)
        self.graph = nx.Graph()
        self.build()
        nx.write_gexf(self.graph, out_file)
    
    def recover_blankspace(self, text):
        return re.sub(r"%", " ", text)
    
    def __add_node(self, name, **kwargs):
        if not self.graph.has_node(name):
            self.graph.add_node(name, **kwargs)
    
    def __add_edge(self, src, dst, **kwargs):
        if not self.graph.has_edge(src, dst):
            self.graph.add_edge(src, dst, **kwargs)
    
    def get_node_type(self, text:str):
        if text.startswith("$REPO"):
            return "r", self.recover_blankspace("$REPO{}".format(self.idx2repo[text[5:]]))
        if text.startswith("$NAME"):
            return "n", self.recover_blankspace(text)
        if text.startswith("$USER"):
            return "u", self.recover_blankspace(text)
        if text.startswith("$TOPIC"):
            return "t", self.recover_blankspace(text)
        if text.startswith("$KEYWORD"):
            return "k", self.recover_blankspace(text)
        if text.startswith("$DOMAIN"):
            return "d", self.recover_blankspace(text)
        if text.startswith("$WORD"):
            return "w", self.recover_blankspace(text)
        else:
            print(text)
    
    def build(self):
        with open(self.link_file, "r", encoding="utf-8") as inf:
            for line in inf:
                src, dst = line.strip().split(" ")
                src_type, src = self.get_node_type(src)
                dst_type, dst = self.get_node_type(dst)
                self.__add_node(src, type=src_type)
                self.__add_node(dst, type=dst_type)
                self.__add_edge(src, dst)


if __name__ == "__main__":
    hin_output = "./hin/vec.dat"
    repoidx2reponame = "./hin/repoidx2reponame.json"
    out_file = "./exports/parsed_vec.txt"
    HINParser(hin_output, repoidx2reponame, out_file)
    link_file = "./hin/link.dat"
    out_file = "./exports/hin.gexf"
    HINRecover(link_file, repoidx2reponame, out_file)
