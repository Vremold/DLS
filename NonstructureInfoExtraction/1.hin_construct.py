import json
import sys
import os
import pickle
import re

import spacy

nlp = spacy.load("en_core_web_sm")

from ..pathutil import CLEANED_GH_REPOSITORY_FILE, CLEANED_GH_REPOSITORY_README_DIR

class Preprocessor(object):
    def __init__(self, src_lib_stat_file, src_lib_readme_dir, filtered_keywords_file, filtered_topics_file):
        self.src_lib_readme_dir = src_lib_readme_dir
        self.src_lib_stat_file = src_lib_stat_file
        self.readme_filenames = []
        self.error_cnt = 0
        self.repoidx2reponame = dict()
        for filename in os.listdir(src_lib_readme_dir):
            self.readme_filenames.append(filename)
        with open(filtered_keywords_file, "r", encoding="utf-8") as inf:
            self.filtered_keywords = json.load(inf)
        with open(filtered_topics_file, "r", encoding="utf-8") as inf:
            self.filtered_topics = json.load(inf)

    def extract_repo_info_from_github_url(self, github_url):
        first_idx = github_url.rfind("/")
        repo_name = github_url[first_idx+1:]
        second_idx = github_url[:first_idx].rfind("/")
        repo_owner = github_url[second_idx+1:first_idx]
        return repo_owner, repo_name
    
    def cutword_for_repo_name(self, repo_name):
        return re.split(r"-|_", repo_name)
    
    def filterWords(self, text):
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ) ", text)
        allowed_word_types = ["ADJ", "NOUN"]
        doc = nlp(text)
        ret = []
        for token in doc:
            if token.pos_ in allowed_word_types:
                ret.append(token.lemma_)
        # for nc in doc.noun_chunks:
        #     ret.append(nc.lemma_)
        return ret

    def extract_text_from_readme(self, readme_filename):
        if readme_filename not in self.readme_filenames:
            self.error_cnt += 1
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
                
    
    def build_dataset(self, cache_file):
        # if os.path.exists(cache_file):
        #     return
        cnt = 0
        with open(self.src_lib_stat_file, "r", encoding="utf-8") as inf, open(cache_file, "w", encoding="utf-8") as outf:
            for idx, line in enumerate(inf):
                cnt += 1
                obj = json.loads(line)
                github_url = obj.get("git_url")
                repo_owner, repo_name = self.extract_repo_info_from_github_url(github_url)

                github_description = obj.get("git_description", "")
                npm_description = obj.get("npm_description", "")
                readme_filename = "{}#{}.txt".format(repo_owner, repo_name)
                readme = self.extract_text_from_readme(readme_filename)

                github_description_words = self.filterWords(github_description)
                npm_description_words = self.filterWords(npm_description)
                readme_words = self.filterWords(readme)

                keywords = [kw.lower() for kw in obj.get("npm_keywords", []) if kw.lower() in self.filtered_keywords]
                topics = [topic.lower() for topic in obj.get("npm_domain", []) if topic in self.filtered_topics]
                
                self.repoidx2reponame[idx] = "{}#{}".format(repo_owner, repo_name)
                outf.write(json.dumps(
                    {
                        "repo_owner": repo_owner,
                        "repo_name": self.cutword_for_repo_name(repo_name),
                        "topics": topics,
                        "keywords": keywords,
                        "domains": obj.get("npm_domain", []),
                        "readme": readme_words,
                        "description": github_description_words + npm_description_words
                    }
                , ensure_ascii=False)+"\n")
        with open("./hin/repoidx2reponame.json", "w", encoding="utf-8") as inf:
            json.dump(self.repoidx2reponame, inf, ensure_ascii=False)
        print(cnt)
        print(self.error_cnt)

class HINConstructor(object):
    def __init__(self, src_file, stop_words=set()):
        self.src_file = src_file
        self.words = dict()
        self.names = set()
        self.topics = set()
        self.keywords = set()
        self.repos = set()
        self.domains = set()
        self.users = set()
        with open(src_file, "r", encoding="utf-8") as inf:
            for idx, line in enumerate(inf):
                obj = json.loads(line)

                self.repos.add("$REPO"+str(idx))
                self.users.add("$USER"+self.__remove_blankspace(obj["repo_owner"]))
                
                for item in obj["domains"]:
                    domain = self.__remove_blankspace(item.lower())
                    self.domains.add("$DOMAIN"+domain)
                    domain_word = "$WORD"+domain
                    if domain_word not in self.words:
                        self.words[domain_word] = 0
                    self.words[domain_word] += 1
                for item in obj["topics"]:
                    topic = self.__remove_blankspace(item.lower())
                    self.topics.add("$TOPIC"+topic)
                    topic_word = "$WORD" + topic
                    if topic_word not in self.words:
                        self.words[topic_word] = 0
                    self.words[topic_word] += 1
                for item in obj["keywords"]:
                    keyword = self.__remove_blankspace(item.lower())
                    self.keywords.add("$KEYWORD"+keyword)
                    keyword_word = "$WORD" + keyword
                    if keyword_word not in self.words:
                        self.words[keyword_word] = 0
                    self.words[keyword_word] += 1
                for item in obj["repo_name"]:
                    name = self.__remove_blankspace(item.lower())
                    self.names.add("$NAME"+name)
                    name_word = "$WORD" + name
                    if name_word not in self.words:
                        self.words[name_word] = 0
                    self.words[name_word] += 1
                for word in obj["readme"]:
                    word = "$WORD" + self.__remove_blankspace(word.lower())
                    if word not in self.words:
                        self.words[word] = 0
                    self.words[word] += 1
                for word in obj["description"]:
                    word = "$WORD" + self.__remove_blankspace(word.lower())
                    if word not in self.words:
                        self.words[word] = 0
                    self.words[word] += 1
        for word in stop_words:
            if word in self.words:
                del self.words[word]
        pass

    def __remove_blankspace(self, text):
        return re.sub(r" ", "%", text)

    def build_nodes(self, out_file, link_out_file):
        with open(out_file, "w") as outf, open(link_out_file, "w") as linkoutf:
            for R in self.repos:
                outf.write(R + " r\n")
            for N in self.names:
                outf.write(N + " n\n")
                linkoutf.write("{} {}\n".format(N, "$WORD"+N[5:]))
                linkoutf.write("{} {}\n".format("$WORD"+N[5:], N))
            for T in self.topics:
                outf.write(T + " t\n")
                linkoutf.write("{} {}\n".format(T, "$WORD"+T[6:]))
                linkoutf.write("{} {}\n".format("$WORD"+T[6:], T))
            for K in self.keywords:
                outf.write(K + " k\n")
                linkoutf.write("{} {}\n".format(K, "$WORD"+K[8:]))
                linkoutf.write("{} {}\n".format("$WORD"+K[8:], K))
            for D in self.domains:
                outf.write(D + " d\n")
                linkoutf.write("{} {}\n".format(D, "$WORD"+D[7:]))
                linkoutf.write("{} {}\n".format("$WORD"+D[7:], D))
            for U in self.users:
                outf.write(U + " u\n")
            for word in self.words:
                if self.words.get(word, 0) >= 4:
                    outf.write(word + " w\n")
    
    def build_links(self, out_file):
        with open(out_file, "a+") as outf, open(self.src_file, "r", encoding="utf-8") as inf:
            for idx, line in enumerate(inf):
                obj = json.loads(line)
                R = "$REPO" + str(idx)
                U = "$USER" + self.__remove_blankspace(obj["repo_owner"])
                Ts = ["$TOPIC"+self.__remove_blankspace(item.lower()) for item in obj["topics"]]
                # Twords = ["$WORD" + item[6:] for item in Ts]
                Ks = ["$KEYWORD"+self.__remove_blankspace(item.lower()) for item in obj["keywords"]]
                # Kwords = ["$WORD" + item[8:] for item in Ks]
                Ds = ["$DOMAIN"+self.__remove_blankspace(item.lower()) for item in obj["domains"]]
                # Dwords = ["$WORD" + item[7:] for item in Ds]
                Ns = ["$NAME"+self.__remove_blankspace(item.lower()) for item in obj["repo_name"]]
                # Nwords = ["$WORD" + item[5:] for item in Ns]

                sent_tokens = []
                for token in obj["readme"] + obj["description"]:
                    token = "$WORD" + self.__remove_blankspace(token.lower())
                    if self.words.get(token, 0) >= 4:
                        sent_tokens.append(token)
                
                for token in sent_tokens:
                    outf.write("{} {}\n".format(token, R))
                    outf.write("{} {}\n".format(R, token))
                    outf.write("{} {}\n".format(token, U))
                    outf.write("{} {}\n".format(U, token))
                    
                    for T in Ts:
                        outf.write("{} {}\n".format(token, T))
                        outf.write("{} {}\n".format(T, token))
                    
                    for K in Ks:
                        outf.write("{} {}\n".format(token, K))
                        outf.write("{} {}\n".format(K, token))
                    
                    for D in Ds:
                        outf.write("{} {}\n".format(token, D))
                        outf.write("{} {}\n".format(D, token))
                    
                    for N in Ns:
                        outf.write("{} {}\n".format(token, N))
                        outf.write("{} {}\n".format(N, token))
    
    def build_paths(self, out_file):
        with open(out_file, "w") as outf:
            outf.write("wrw 0.16\n")
            outf.write("wuw 0.16\n")
            outf.write("wtw 0.20\n")
            outf.write("wdw 0.16\n")
            outf.write("wnw 0.16\n")
            outf.write("wkw 0.16\n")
    
    def main(self):
        self.build_nodes(out_file="./hin/node.dat", link_out_file="./hin/link.dat")
        self.build_links(out_file="./hin/link.dat")
        self.build_paths(out_file="./hin/path.dat")

if __name__ == "__main__":
    filtered_keywords_file = PROJECT_DIR + "../StructureInfoExtraction/BeforeKG/filtered_keywords.json"
    filtered_topics_file = PROJECT_DIR + "../StructureInfoExtraction/BeforeKG/filtered_topics.json"
    ppor = Preprocessor(
        src_lib_readme_dir=CLEANED_GH_REPOSITORY_README_DIR, 
        src_lib_stat_file=CLEANED_GH_REPOSITORY_FILE,
        filtered_keywords_file=filtered_keywords_file,
        filtered_topics_file=filtered_topics_file
    )
    ppor.build_dataset(cache_file="./cache/raw_dataset.txt")
    
    stop_words = set()
    with open("./stop.txt", "r", encoding="utf-8") as inf:
        for line in inf:
            stop_words.add(line.strip())
    hinc = HINConstructor(src_file="./cache/raw_dataset.txt", stop_words=stop_words)
    hinc.main()
