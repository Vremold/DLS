import sys
import os
import json

sys.path.append("../..")

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from pathutil  import CLEANED_GH_REPOSITORY_FILE, CLEANED_GH_REPOSITORY_README_DIR

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

                dataset.append((github_description+npm_description, idx))
        return dataset, idx2reponame

def train_doc2vec(dataset, out_file):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(idx)]) for _d, idx in dataset]

    model = Doc2Vec(tagged_data, dm=1, vector_size=100, window=8, min_count=4, workers=4)
    
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    
    model.save(out_file)

if __name__ == "__main__":
    dataset, idx2reponame = Dataset(CLEANED_GH_REPOSITORY_FILE, CLEANED_GH_REPOSITORY_README_DIR).build_dataset()
    with open("./idx2reponame.json", "w", encoding="utf-8") as outf:
        json.dump(idx2reponame, outf)
    train_doc2vec(dataset, out_file="./d2v.mdl")
