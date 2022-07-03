import json
import sys
sys.path.append("..")

from fuzzywuzzy import fuzz
import numpy as np

from Query import QueryPredNE, QueryWordExtractor
from Enricher import ENRicher
from Searcher import Seacher
from pathutil import HIN, HIN_EMBEDDING, KG, KG_NODE_EMBEDDING, KG_RELATION_EMBEDDING, PROJECT_ABS_DIR
from config import TEXT_PREPROCESS_CHOICE, BERT_KIND

class NERefiner(object):
    def __init__(self, ne2ne_file, ne2cate_file):
        # self.neabbr = NEAbbr()
        with open(ne2ne_file, "r", encoding="utf-8") as inf:
            self.ne2ne = json.load(inf)
        with open(ne2cate_file, "r", encoding="utf-8") as inf:
            self.ne2cate = json.load(inf)
    
    def process(self, nes:dict):
        ret = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
        for cate in nes:
            for ne in nes[cate]:
                ne = ne.lower()
                ne = self.ne2ne.get(ne, ne)
                new_cate = self.ne2cate.get(ne, cate)
                if ne not in ret[new_cate]:
                    ret[new_cate][ne] = 0
                ret[new_cate][ne] += 1
        
        return ret

class UI(object):
    def __init__(self):
        token_dict_file = PROJECT_ABS_DIR + "/NER/data/token_dict.pkl"
        pretrained_model = PROJECT_ABS_DIR + "/NER/pretrained/bert_{}_{}".format(BERT_KIND, TEXT_PREPROCESS_CHOICE)
        trained_model = PROJECT_ABS_DIR + "/NER/builds/bert_{}_{}.mdl".format(BERT_KIND, TEXT_PREPROCESS_CHOICE)
        self.qpne = QueryPredNE(
            token_dict_file=token_dict_file, 
            pretrained_model=pretrained_model, 
            trained_model=trained_model
        )
        self.ne_refiner = NERefiner(PROJECT_ABS_DIR+"/StructureInfoExtraction/BeforeKG/ne2ne.json", PROJECT_ABS_DIR+"/StructureInfoExtraction/BeforeKG/ne2category.json")
        self.qwe = QueryWordExtractor()
        self.enricher = ENRicher()
        self.s = Seacher(HIN, HIN_EMBEDDING, KG, KG_NODE_EMBEDDING, KG_RELATION_EMBEDDING)
    
    def enrich_ne_version2(self, nes:dict()):
        enriched_nes = dict()
        for cate in nes:
            if cate not in enriched_nes:
                enriched_nes[cate] = dict()
            for ne in nes[cate]:
                if ne not in enriched_nes:
                    enriched_nes[cate][ne] = nes[cate][ne]
                enriches = self.enricher.enrich_ne(ne)
                for ene, prob in enriches:
                    if ene not in enriched_nes[cate]:
                        enriched_nes[cate][ene] = 0
                    enriched_nes[cate][ene] += 0.01 * prob * nes[cate][ne] / len(enriches)
        return enriched_nes
    
    def enrich_ne(self, nes:dict):
        enriched_nes = dict()
        total_cnt = 0
        for cate in nes:
            if cate not in enriched_nes:
                enriched_nes[cate] = dict()
            for ne in nes[cate]:
                if ne not in enriched_nes[cate]:
                    enriched_nes[cate][ne] = {"cnt": nes[cate][ne], "prob": 1}
                    total_cnt += nes[cate][ne]
                enrichs = self.enricher.enrich_ne(ne)
                for ene, prob in enrichs:
                    if ene in enriched_nes[cate] or ene in nes[cate]:
                        continue
                    total_cnt += 1
                    enriched_nes[cate][ene] = {"cnt": 1, "prob": prob}
        
        ret = dict()
        for cate in enriched_nes:
            if cate not in ret:
                ret[cate] = dict()
            for ne in enriched_nes[cate]:
                ret[cate][ne] = enriched_nes[cate][ne]["cnt"] / total_cnt * enriched_nes[cate][ne]["prob"]
        
        return ret
    
    def enrich_word_version2(self, realwords:dict):
        enriched_realwords = dict()

        for rw in realwords:
            if rw not in enriched_realwords:
                enriched_realwords[rw] = realwords[rw]
            enriches = self.enricher.enrich_word(rw)
            for erw, prob in enriches:
                if erw not in enriched_realwords:
                    enriched_realwords[erw] = 0
                enriched_realwords[erw] += 0.01 * prob * realwords[rw] / len(enriches)
        return enriched_realwords

    def enrich_word(self, realwords:dict):
        total_cnt = 0
        enriched_realwords = dict()
        for rw in realwords:
            total_cnt += realwords[rw]
            enriched_realwords[rw] = {"cnt": realwords[rw], "prob": 1}
            enrichs = self.enricher.enrich_word(rw)
            for erw, prob in enrichs:
                if erw in realwords or erw in enriched_realwords:
                    continue
                total_cnt += 1
                enriched_realwords[erw] = {"cnt": 1, "prob": prob}
        
        ret = dict()
        for rw in enriched_realwords:
            ret[rw] = enriched_realwords[rw]["cnt"] / total_cnt * enriched_realwords[rw]["prob"]
        
        return ret
    
    def query(self, text, alpha=0.35):
        nes = self.qpne.predict(text)
        nes = self.ne_refiner.process(nes)
        real_words = self.qwe.extract_useful_words(text)
        repos = self.s.search(nes, real_words, alpha=alpha)
        return nes, real_words, repos
    
    def query_for_ui(self, text, alpha=0.35):
        nes = self.qpne.predict(text)
        nes = self.ne_refiner.process(nes)
        real_words = self.qwe.extract_useful_words(text)
        search_result = self.s.search(nes, real_words, alpha=alpha)[:20]
        return [item[0] for item in search_result]

if __name__ == "__main__":
    system = UI()
    text = "a tool which is able to finds errors and problems for javacript"
    text = "I want a library that help develop Salesforce API applications with JavaScript"
    repos = system.query_for_ui(text)
    print(repos)