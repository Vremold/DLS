import os
import sys
import pickle
import json

sys.path.append("../..")

from gensim import corpora,similarities,models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from pathutil import CLEANED_GH_REPOSITORY_FILE, CLEANED_GH_REPOSITORY_README_DIR

stop_words = set(stopwords.words('english'))

class Dataset(object):
    def __init__(self, src_lib_stat_file, src_lib_readme_dir):
        self.src_lib_readme_dir = src_lib_readme_dir
        self.src_lib_stat_file = src_lib_stat_file
        self.readme_filenames = []
        for filename in os.listdir(src_lib_readme_dir):
            self.readme_filenames.append(filename)

    def extract_repo_info_from_github_url(self, github_url):
        first_idx = github_url.rfind("/")
        repo_name = github_url[first_idx+1:]
        second_idx = github_url[:first_idx].rfind("/")
        repo_owner = github_url[second_idx+1:first_idx]
        return repo_owner, repo_name
    
    def extract_text_from_readme(self, readme_filename):
        if readme_filename not in self.readme_filenames:
            return ""
        ret = []
        with open(os.path.join(self.src_lib_readme_dir, readme_filename), "r", encoding="utf-8") as inf:
            for line in inf:
                splits = line.split("\t")
                ret.append(splits[1])
                try:
                    contents = json.loads(splits[-1])
                except Exception:
                    continue
                for sent in contents:
                    if isinstance(sent, str):
                        ret.append(sent)
                    elif isinstance(sent, list):
                        ret.extend(sent)
        return " ".join(ret)
    
    def build_dataset(self):
        vocabulary = dict()
        dataset = []
        idx2reponame = dict()
        with open(self.src_lib_stat_file, "r", encoding="utf-8") as inf:
            for idx, line in enumerate(inf):
                obj = json.loads(line)
                github_url = obj.get("git_url")
                repo_owner, repo_name = self.extract_repo_info_from_github_url(github_url)

                idx2reponame[idx] = "{}#{}".format(repo_owner, repo_name)

                github_description = obj.get("git_description", "")
                npm_description = obj.get("npm_description", "")
                readme_filename = "{}#{}.txt".format(repo_owner, repo_name)
                readme = self.extract_text_from_readme(readme_filename)
                text = github_description + " " + npm_description + " " +readme
                tokens = word_tokenize(text)
                dataset.append([token for token in tokens if token not in stop_words])
        return dataset, idx2reponame

def train_tf_idf(dataset):
    dictionary = corpora.Dictionary(dataset)
    corpus = [dictionary.doc2bow(text) for text in dataset]

    tfidf_model = models.TfidfModel(corpus, dictionary=dictionary)
    corpus_tfidf = tfidf_model[corpus]
    dictionary.save('./model/train_dictionary.dict')  # 保存生成的词典
    tfidf_model.save('./model/train_tfidf.mdl')
    corpora.MmCorpus.serialize('./model/train_corpuse.mm', corpus)
    featurenum = len(dictionary.token2id.keys())  # 通过token2id得到特征数
    # 稀疏矩阵相似度，从而建立索引,我们用待检索的文档向量初始化一个相似度计算的对象
    index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum)
    index.save('./model/train_index.index')

    """
    参考
    https://blog.csdn.net/qq_34333481/article/details/85327090
    """

if __name__ == "__main__":
    dataset, idx2reponame = Dataset(CLEANED_GH_REPOSITORY_FILE, CLEANED_GH_REPOSITORY_README_DIR).build_dataset()
    with open("./model/idx2reponame.json", "w", encoding="utf-8") as outf:
        json.dump(idx2reponame, outf, ensure_ascii=False)
    train_tf_idf(dataset)