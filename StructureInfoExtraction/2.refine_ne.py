"""
由于抽取出来的命名实体存在大量的形似项，例如chrome和｜chrome。
这部分采取的方法是将所有实体转换为lemma原型，判断原型的字面重合度，如果重合度过高，那么就将这些形似实体归位出现率最高的那一个，由此构建ne2ne的映射
同时这些形似实体有可能本来的类别不一样，这就还有必要构建一个ne2category的映射
"""

import json
import sys
import os
import re

from fuzzywuzzy import fuzz
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")

class NEPreprocessor(object):
    def __init__(self, ne_file):
        self.ne_file = ne_file
        self.ne2ne = dict()
        self.ne2cate = dict()
    
    def build_ne_category_map(self, out_file):
        if os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8") as inf:
                self.ne2cate = json.load(inf)
            return
        ne2categories = dict()
        with open(self.ne_file, "r", encoding="utf-8") as inf:
            for line in inf:
                splits = line.strip().split("\t")
                for i in range(1, 7):
                    obj = json.loads(splits[i])
                    for ne in obj:
                        ne = ne.lower() # 转换为小写
                        ne = self.ne2ne.get(ne, ne)
                        if ne not in ne2categories:
                            ne2categories[ne] = [0] * 6
                        ne2categories[ne][i - 1] += 1
        for ne in ne2categories:
            self.ne2cate[ne] = int(np.argmax(ne2categories[ne]))
        
        with open(out_file, "w", encoding="utf-8") as outf:
            json.dump(self.ne2cate, outf, ensure_ascii=False)

    def _convert2lemma(self, ne):
        # PUNCTs = ["-", "_", "|"]
        doc = nlp(ne)
        text = " ".join([token.lemma_ for token in doc])
        text = re.sub(r"( - )|( _ )", " ", text)
        return text
    
    def _match_lemma(self, lemma, lemma2nes):
        if lemma in lemma2nes:
            return True, lemma
        ret = {}
        for key in lemma2nes:
            sim = fuzz.ratio(key, lemma)
            if sim >= 90:
                ret[key] = sim
        if len(ret) == 0:
            return False, lemma
        simLemmas = sorted(ret.items(), key=lambda x : x[1], reverse=True)
        return True, simLemmas[0][0]

    def build_ne_ne_map(self, out_file):
        if os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8") as inf:
                self.ne2ne = json.load(inf)
            return 
        nes = dict()
        lemma2nes = dict()
        with open(self.ne_file, "r", encoding="utf-8") as inf:
            for line in inf:
                splits = line.split("\t")
                for i in range(1, 7):
                    obj = json.loads(splits[i])
                    for ne in obj:
                        ne = ne.lower() # 转换为小写
                        lemma = self._convert2lemma(ne)
                        if ne not in nes:
                            nes[ne] = 0
                        nes[ne] += 1
                        lemma_exists, lemma = self._match_lemma(lemma, lemma2nes)
                        if not lemma_exists:
                            lemma2nes[lemma] = set()
                        lemma2nes[lemma].add(ne) 
        
        for lemma in lemma2nes:
            matched_ne = ""
            matched_ne_cnt = 0
            for ne in lemma2nes[lemma]:
                if nes[ne] > matched_ne_cnt:
                    matched_ne_cnt = nes[ne]
                    matched_ne = ne
            for ne in lemma2nes[lemma]:
                self.ne2ne[ne] = matched_ne
        
        with open(out_file, "w", encoding="utf-8") as outf:
            json.dump(self.ne2ne, outf, ensure_ascii=False)
        pass

    def refine_nes(self, line):
        splits = line.strip().split("\t")
        filename = splits[0]
        ret = {0: dict(), 1: dict(), 2: dict(), 3: dict(), 4: dict(), 5: dict()}
        ne_cnt = 0
        for i in range(1, 7):
            obj = json.loads(splits[i])
            for ne in obj:
                ne = self.ne2ne.get(ne, ne)
                cate = self.ne2cate.get(ne)
                print(cate, ne)
                if ne not in ret[cate]:
                    ret[cate][ne] = 0
                ret[cate][ne] += 1
                
                ne_cnt += 1
        return filename, ret

    def refine_input_file(self):
        with open(self.ne_file, "r", encoding="utf-8") as inf, open(self.ne_file+".bak", "w", encoding="utf-8") as outf:
            for line in inf:
                filename, nes = self.refine_nes(line)
                outf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    filename,
                    json.dumps(nes[0], ensure_ascii=False),
                    json.dumps(nes[1], ensure_ascii=False),
                    json.dumps(nes[2], ensure_ascii=False),
                    json.dumps(nes[3], ensure_ascii=False),
                    json.dumps(nes[4], ensure_ascii=False),
                    json.dumps(nes[5], ensure_ascii=False)
                ))
        os.remove(self.ne_file)
        os.rename(self.ne_file+".bak", self.ne_file)

if __name__ == "__main__":
    nepp = NEPreprocessor(ne_file="./BeforeKG")
    nepp.build_ne_ne_map(out_file="./BeforeKG/ne2ne.json")
    nepp.build_ne_category_map(out_file="./BeforeKG/ne2category.json")
    nepp.refine_input_file()
