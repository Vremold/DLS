import json
import sys
sys.path.append("..")
import pickle
import os
import random

import numpy as np

from pathutil import KG_NODE_EMBEDDING, KG_RELATION_EMBEDDING, HIN_EMBEDDING

class NEEnricher(object):
    def __init__(self, ent_vec_file):
        ent2idx = dict()
        self.ent2vec = dict()
        with open(ent_vec_file, "r", encoding="utf-8") as inf:
            for line in inf:
                ent, vec = line.split("\t")
                if not ent.startswith("$NE"):
                    continue
                ent = ent[4:]
                vec = np.array(json.loads(vec), dtype=float)
                if ent not in ent2idx:
                    ent2idx[ent] = len(ent2idx)
                else:
                    print(ent)
                if ent not in self.ent2vec:
                    self.ent2vec[ent] = vec
        
        self.idx2ent = {value: key for key, value in ent2idx.items()}
        print("NE Length:", len(self.idx2ent))
    
    def get_cos_similarity(self, v1, v2):
        num = np.dot(v1, v2)  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    
    def get_euclidean_dist(self, v1, v2):
        return np.linalg.norm(v1-v2)

    def get_sims(self, cache_file):
        if os.path.exists(cache_file):
            return pickle.load(open(cache_file, "rb"))
        ent_cnt = len(self.idx2ent)
        sims = np.zeros(shape=(ent_cnt, ent_cnt), dtype=float)
        for i in range(ent_cnt):
            vec1 = self.ent2vec[self.idx2ent[i]]
            for j in range(i, ent_cnt):
                vec2 = self.ent2vec[self.idx2ent[j]]
                sims[i][j] = self.get_cos_similarity(vec1, vec2)
                sims[j][i] = sims[i][j]
        
        with open(cache_file, "wb") as outf:
            pickle.dump(sims, outf)
        print("Get Sim Finished!")
        return sims
    
    def calculate_siment(self, threshold, out_file, cache_file):
        sims = self.get_sims(cache_file)
        # return
        ent_cnt = len(self.idx2ent)
        ent2siment = {}
        for i in range(ent_cnt):
            # print("Now processing entity[{}]".format(i))
            sim = []
            for j in range(ent_cnt):
                if j == i:
                    continue
                sim.append((j, sims[i][j]))

            sim.sort(key=lambda x: x[1], reverse=True)
            cluster = [i]
            for idx, s in sim:
                if s < threshold:
                    break
                should_cluster = True
                for clustered_idx in cluster:
                    if sims[clustered_idx][idx] < threshold:
                        should_cluster = False
                        break
                if should_cluster:
                    cluster.append(idx)

            self_ent = self.idx2ent[i]
            ent2siment[self_ent] = []
            for clustered_idx in cluster[1:]:
                ent2siment[self_ent].append((self.idx2ent[clustered_idx], sims[i][clustered_idx]))
        json.dump(ent2siment, open(out_file, "w", encoding="utf-8"), ensure_ascii=False)

class HINEnricher(object):
    def __init__(self, hin_vec_file):
        word2idx = dict()
        self.word2vec = dict()
        with open(hin_vec_file, "r", encoding="utf-8") as inf:
            next(inf)
            next(inf)
            next(inf)
            next(inf)
            next(inf)
            next(inf)
            self.word2vec = json.loads(inf.readline())
        for word in self.word2vec:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        self.idx2word = {value: key for key, value in word2idx.items()}
    
    def get_cos_similarity(self, v1, v2):
        num = np.dot(v1, v2)  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    
    def get_euclidean_dist(self, v1, v2):
        return np.linalg.norm(v1-v2)

    def get_sims(self, cache_file):
        if os.path.exists(cache_file):
            return pickle.load(open(cache_file, "rb"))
        word_cnt = len(self.idx2word)
        print(word_cnt)
        sims = np.zeros(shape=(word_cnt, word_cnt), dtype=float)
        for i in range(word_cnt):
            vec1 = np.array(self.word2vec[self.idx2word[i]], dtype=float)
            for j in range(i, word_cnt):
                vec2 = np.array(self.word2vec[self.idx2word[j]], dtype=float)
                sims[i][j] = self.get_cos_similarity(vec1, vec2)
                sims[j][i] = sims[i][j]
        
        with open(cache_file, "wb") as outf:
            pickle.dump(sims, outf)
        print("Get Sim Finished!")
        return sims
    
    def calculate_simword(self, threshold, out_file, cache_file):
        sims = self.get_sims(cache_file)
        # return
        word_cnt = len(self.idx2word)
        word2simword = {}
        for i in range(word_cnt):
            sim = []
            for j in range(word_cnt):
                if j == i:
                    continue
                sim.append((j, sims[i][j]))

            sim.sort(key=lambda x: x[1], reverse=True)
            cluster = [i]
            for idx, s in sim:
                if s < threshold:
                    break
                should_cluster = True
                for clustered_idx in cluster:
                    if sims[clustered_idx][idx] < threshold:
                        should_cluster = False
                        break
                if should_cluster:
                    cluster.append(idx)

            self_word = self.idx2word[i]
            word2simword[self_word] = []
            for clustered_idx in cluster[1:]:
                word2simword[self_word].append((self.idx2word[clustered_idx], sims[i][clustered_idx]))
        json.dump(word2simword, open(out_file, "w", encoding="utf-8"), ensure_ascii=False)

class ENRicher(object):
    def __init__(self):
        self.ne2simne = json.load(open("./cache/ne2simne.json"))
        self.word2simword = json.load(open("./cache/word2simword.json"))
    
    def enrich_word(self, word):
        return self.word2simword.get(word, [])[:2]
    
    def enrich_ne(self, ne): 
        return self.ne2simne.get(ne, [])[:2]

if __name__ == "__main__":
    print("Computing for HIN")
    nee = HINEnricher(HIN_EMBEDDING)
    nee.calculate_simword(0.835, out_file="./cache/word2simword.json", cache_file="./cache/wordsim.pkl")
    print("Computing for ENT")
    nee = NEEnricher(KG_NODE_EMBEDDING)
    nee.calculate_siment(0.84, out_file="./cache/ne2simne.json", cache_file="./cache/nesim.pkl")
