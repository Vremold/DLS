import sys
import os
import json
import re
import pickle

import numpy as np
import networkx as nx 

class HINSearcher(object):
    def __init__(self, gexf_file, word_embed):
        self.graph = nx.read_gexf(gexf_file)
        self.repovec = {}
        self.wordvec = {}
        with open(word_embed, "r", encoding="utf-8") as inf:
            next(inf)
            self.repovec = json.loads(inf.readline())
            for repo in self.repovec:
                self.repovec[repo] = np.array(self.repovec[repo])
            next(inf)
            next(inf)
            next(inf)
            next(inf)
            self.wordvec = json.loads(inf.readline())
            for word in self.wordvec:
                self.wordvec[word] = np.array(self.wordvec[word])

        pass

    def get_cos_similarity(self, v1, v2):
        num = np.dot(v1, v2)  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

    def search_repo(self, word, prob):
        repos = {}
        word_vec = self.wordvec.get(word, np.zeros(shape=(100)))
        # print(word_vec)
        word = "$WORD"+word
        if word not in self.graph.nodes:
            return repos
        for nbr in self.graph[word]:
            if self.graph.nodes[nbr]["type"] == "r":
                repo = nbr[5:]
                repo_vec = self.repovec.get(repo, np.zeros(shape=(100)))
                # print(repo, repo_vec)
                if np.sum(repo_vec) == 0:
                    print(repo)
                if repo not in repos:
                    repos[repo] = 0
                repos[repo] += prob * self.get_cos_similarity(repo_vec, word_vec)
        return repos
    
class NESearcher(object):
    def __init__(self, gexf_file, ent_embed, rel_embed):
        self.graph = nx.read_gexf(gexf_file)
        self.entvec = {}
        self.relvec = {}
        with open(ent_embed, "r", encoding="utf-8") as inf:
            for line in inf:
                ent, vec = line.strip().split("\t")
                self.entvec[ent] = np.array(json.loads(vec))
        with open(rel_embed, "r", encoding="utf-8") as inf:
            for line in inf:
                rel, vec = line.strip().split("\t")
                self.relvec[rel] = np.array(json.loads(vec))
        pass
    
    def get_cos_similarity(self, v1, v2):
        num = np.dot(v1, v2)  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

    def search_repo(self, ne, cate, prob):
        repos = {}
        ne = "$NE{}{}".format(cate, ne)
        if ne not in self.graph.nodes:
            return repos
        
        # 获取相似的仓库
        nes = [ne]
        for nbr in self.graph[ne]:
            if self.graph[ne][nbr]["kind"] == "SameToNE":
                nes.append(nbr)
        
        for ent in nes:
            entity_vec = self.entvec.get(ent, np.zeros(shape=(200)))
            if np.sum(entity_vec) == 0:
                print(ne)
            for pnbr in self.graph.predecessors(ent):
                if self.graph[pnbr][ent]["kind"].startswith("RelateToNE"):
                    repovec = self.entvec.get(pnbr, np.zeros(shape=(200)))
                    relation_vec = self.relvec.get(self.graph[pnbr][ent]["kind"], np.zeros(shape=(200)))
                    if np.sum(repovec) == 0 or np.sum(relation_vec) == 0:
                        print(pnbr)
                    repo = pnbr[4:]
                    if repo not in repos:
                        repos[repo] = 0
                    
                    repos[repo] += prob * self.get_cos_similarity(entity_vec-relation_vec, repovec)
        return repos

class Seacher(object):
    def __init__(self, hin, hin_embedding, kg, kg_node_embedding, kg_relation_embedding):
        self.hs = HINSearcher(hin, hin_embedding)
        self.ns = NESearcher(kg, kg_node_embedding, kg_relation_embedding)
    
    def search(self, nes, realwords, alpha=0.7):

        repos = dict()
        for cate in nes:
            for ne in nes[cate]:
                tmp = self.ns.search_repo(ne, cate, nes[cate][ne])
                for repo in tmp:
                    if repo not in repos:
                        repos[repo] = 0
                    repos[repo] += tmp[repo] * alpha
        
        for kw in realwords:
            tmp = self.hs.search_repo(kw, realwords[kw])
            for repo in tmp:
                if repo not in repos:
                    repos[repo] = 0
                repos[repo] += tmp[repo] * (1 - alpha)
        
        repos = sorted(repos.items(), key=lambda x: x[1], reverse=True)
        return repos