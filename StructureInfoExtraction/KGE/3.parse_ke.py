import json
import sys
import os
import pickle

class KEParser(object):
    def __init__(self, ke_file, ke_dataset_dir):
        self.ke_file = ke_file
        self.entity2id = {}
        self.relation2id = {}
        with open(ke_dataset_dir+"/entity2id.txt", "r", encoding="utf-8") as inf:
            next(inf)
            for line in inf:
                splits = line.strip().split("\t")
                self.entity2id[splits[0]] = int(splits[1])
        with open(ke_dataset_dir+"/relation2id.txt", "r", encoding="utf-8") as inf:
            next(inf)
            for line in inf:
                splits = line.strip().split("\t")
                self.relation2id[splits[0]] = int(splits[1])
        self.id2entity = {value: key for key, value in self.entity2id.items()}
        self.id2realtion = {value: key for key, value in self.relation2id.items()}
    
    def parse(self, ent_embedding_file, rel_embedding_file):
        with open(self.ke_file, "r", encoding="utf-8") as keinf:
            # print(keinf.readline())
            obj = json.load(keinf)
            print("Entity Length: ", len(obj["ent_embeddings.weight"]))
            with open(ent_embedding_file, "w", encoding="utf-8") as outf:
                for idx, ent_embedding in enumerate(obj["ent_embeddings.weight"]):
                    outf.write("{}\t{}\n".format(self.id2entity[idx], json.dumps(ent_embedding)))
            with open(rel_embedding_file, "w", encoding="utf-8") as outf:
                for idx, rel_embedding in enumerate(obj["rel_embeddings.weight"]):
                    outf.write("{}\t{}\n".format(self.id2realtion[idx], json.dumps(rel_embedding)))

if __name__ == "__main__":
    kep = KEParser("./KE/embed.vec", "./KG")
    kep.parse("./KE/ent_embedding_vec.txt", "./KE/rel_embedding_vec.txt")
