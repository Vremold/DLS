import sys
import os
import json

sys.path.append("../..")

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from pathutil import VERIFICATION_DATA_DIR

class Searcher(object):
    def __init__(self, d2vmodel="./d2v.mdl"):
        self.d2v = Doc2Vec.load(d2vmodel)
    
    def search(self, text, idx2reponame):
        test_data = word_tokenize(text.lower())
        v1 = self.d2v.infer_vector(test_data)
        sims = self.d2v.docvecs.most_similar([v1], topn=20)
        return [idx2reponame.get(tag, "") for tag, sim in sims]

if __name__ == "__main__":
    idx2reponame = json.load(open("./idx2reponame.json", "r", encoding="utf-8"))
    S = Searcher()
    mrr = 0
    success = 0
    with open(VERIFICATION_DATA_DIR, "r", encoding="utf-8") as inf:
        for line in inf:
            text, ans = line.strip().split("\t")
            repos = S.search(text, idx2reponame)
            if ans in repos:
                success += 1
                mrr += 1 / (repos.index(ans) + 1)
    
    print(mrr, success)
    # 0.576 3