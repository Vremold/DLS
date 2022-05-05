import os
import sys
import json

sys.path.append("../..")

from gensim import corpora,similarities,models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from pathutil import VERIFICATION_DATA_DIR

stop_words = set(stopwords.words('english'))

class Searcher(object):
    def __init__(self, save_mdl_dir):
        self.dictionary = corpora.Dictionary.load(save_mdl_dir+"/train_dictionary.dict")
        self.tfidf = models.TfidfModel.load(save_mdl_dir+"/train_tfidf.model")
        self.index = similarities.SparseMatrixSimilarity.load(save_mdl_dir+'/train_index.index')
        self.corpus = corpora.MmCorpus(save_mdl_dir+'/train_corpuse.mm')
        self.idx2reponame = json.load(open(save_mdl_dir+"/idx2reponame.json", "r", encoding="utf-8"))
    
    def search(self, text):
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        vec = self.dictionary.doc2bow(tokens)

        ifidf_vec = self.tfidf[vec]
        # 计算相似度
        sim = self.index.get_similarities(ifidf_vec)
        related_doc_indices = sim.argsort()[:-21:-1]

        return [self.idx2reponame[str(idx)] for idx in related_doc_indices]

if __name__ == "__main__":
    S = Searcher("./model")
    mrr = 0
    success = 0
    with open(VERIFICATION_DATA_DIR, "r", encoding="utf-8") as inf:
        for line in inf:
            text, ans = line.strip().split("\t")
            repos = S.search(text)
            if ans in repos:
                success += 1
                mrr += 1 / (repos.index(ans) + 1)
    print(success, mrr)
    # 9 4.448

