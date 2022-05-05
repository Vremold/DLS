"""
准备训练过程：
第1步：划分数据集
第2步：建立词典
"""

import os
import sys
import time
import json
import shutil
import random
import pickle

from config import TEXT_PREPROCESS_CHOICE
from pathutil import NER_TRAINING_DATASET_DIR, LABELED_NER_DATA, UNLABELED_NER_DATA

"""
第1步：划分数据集
"""
def divide_dataset(labeled_ner_data, unlabeled_ner_data, dst_dir):
    if TEXT_PREPROCESS_CHOICE == "lemma":
        lower, lemma = True, True
    elif TEXT_PREPROCESS_CHOICE == "uncased":
        lower, lemma = True, False
    else:
        lower, lemma = False, False
    
    total_dataset = []
    train_file = open(os.path.join(dst_dir, "train.txt"), "w", encoding="utf-8")
    valid_file = open(os.path.join(dst_dir, "valid.txt"), "w", encoding="utf-8")
    with open(labeled_ner_data, "r", encoding="utf-8") as inf:
        for line in inf:
            splits = line.strip().split("\t")
            if len(splits) < 3:
                continue
            filename, sent, nes = splits[0], splits[1], splits[2]
            total_dataset.append((filename, sent, nes))
    total_dataset_length = len(total_dataset)
    train_dataset_length = total_dataset_length // 10 * 9
    train_dataset_indexes = random.sample(list(range(total_dataset_length)), train_dataset_length)
    for i in range(total_dataset_length):
        line = total_dataset[i]
        if i in train_dataset_indexes:
            train_file.write("{}\t{}\t{}\n".format(line[0], line[1], line[2]))
        else:
            valid_file.write("{}\t{}\t{}\n".format(line[0], line[1], line[2]))
    train_file.close()
    valid_file.close()

    shutil.copyfile(labeled_ner_data, os.path.join(dst_dir, "test.txt"))

"""
第2步：建立词典
"""
def build_token_dict(src_dir):
    token_dict = {}
    src_files = ["train.txt", "valid.txt", "test.txt"]
    for a_file in src_files:
        srcpath = os.path.join(src_dir, a_file)
        with open(srcpath, "r", encoding="utf-8") as inf:
            for line in inf:
                splits = line.strip().split("\t")
                if a_file != "test.txt" and len(splits) < 3:
                    continue
                if a_file == "test.txt" and len(splits) < 2:
                    continue
                sent = json.loads(splits[1])
                for token in sent:
                    if token not in token_dict:
                        token_dict[token] = len(token_dict)
    
    with open(os.path.join(src_dir, "token_dict.pkl"), "wb") as outf:
        pickle.dump(token_dict, outf)

if __name__ == "__main__":
    divide_dataset(LABELED_NER_DATA, UNLABELED_NER_DATA, NER_TRAINING_DATASET_DIR)

