import json
import sys
import os

from ..config import KEYWORD_FREQUENCY_THRESHOLD, TOPIC_FREQUENCY_THRESHOLD
from ..pathutil import CLEANED_GH_REPOSITORY_FILE

class KeywordAndTopicFilter(object):
    def __init__(self, repo_stat_file):
        self.repo_stat_file = repo_stat_file
    
    def filter_keyowrd(self, out_file, threshold):
        keyword2cnt = {}
        with open(self.repo_stat_file, "r", encoding="utf-8") as inf:
            for line in inf:
                obj = json.loads(line)
                keywords = obj.get("npm_keywords", [])
                for kw in keywords:
                    kw = kw.lower()
                    if kw == "view more":
                        continue
                    if kw not in keyword2cnt:
                        keyword2cnt[kw] = 0
                    keyword2cnt[kw] += 1
        filter_kws = [kw for kw in keyword2cnt if keyword2cnt[kw] > threshold]
        with open(out_file, "w", encoding="utf-8") as outf:
            json.dump(filter_kws, outf, ensure_ascii=False)
    
    def filter_topic(self, out_file, threshold):
        topic2cnt = {}
        with open(self.repo_stat_file, "r", encoding="utf-8") as inf:
            for line in inf:
                obj = json.loads(line)
                topics = obj.get("git_topics", [])
                for t in topics:
                    t = t.lower()
                    if t not in topic2cnt:
                        topic2cnt[t] = 0
                    topic2cnt[t] += 1
        filter_topics = [t for t in topic2cnt if topic2cnt[t] > threshold]
        with open(out_file, "w", encoding="utf-8") as outf:
            json.dump(filter_topics, outf, ensure_ascii=False)

if __name__ == "__main__":
    katf = KeywordAndTopicFilter(CLEANED_GH_REPOSITORY_FILE)
    katf.filter_keyowrd(out_file="./BeforeKG/filtered_keywords.json", threshold=KEYWORD_FREQUENCY_THRESHOLD)
    katf.filter_topic(out_file="./BeforeKG/filtered_topics.json", threshold=TOPIC_FREQUENCY_THRESHOLD)