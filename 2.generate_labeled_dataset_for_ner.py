"""
数据标注规则：
B：ne的开始 0
I：ne的中间 1
O：不是ne的任何一部分 18
S：单个的ne 2
"""

import os
import json
import pickle
import re
import sys

import spacy

from pathutil import LABELING_DATA_DIR, LABELED_NER_DATA, UNLABELED_NER_DATA
from config import TEXT_PREPROCESS_CHOICE, SENTENCE_MIN_LENGTH

nlp = spacy.load("en_core_web_sm")

class NEDatasetBuilder(object):
    def __init__(self, src_labeled_readme, src_labeled_gitdes, src_unlabeled_readme, src_unlabeled_gitdes, n_categories=6, n_labels=3, lower=False, lemma=False):
        self.src_labeled_readme = src_labeled_readme
        self.src_labeled_gitdes = src_labeled_gitdes
        self.src_unlabeled_readme = src_unlabeled_readme
        self.src_unlabeled_gitdes = src_unlabeled_gitdes
        self.lower = lower
        self.lemma = lemma
        self.n_categories = 6
        self.n_labels = 3

    def sent_has_min_length(self, sent:list, min_length=3):
        label_symbols = []
        for i in range(self.n_categories):
            label_symbols.append("#{}#".format(i))
        length = 0
        for token in sent:
            if token in label_symbols:
                continue
            length += 1
        return length >= min_length
    
    def line_has_chinese(self, line:str):
        for ch in line:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False
    
    def really_cut_word(self, text):
        doc = nlp(text)
        if self.lemma:
            return [token.lemma_ for token in doc if "//" not in token.text and "---" not in token.text]
        return [token.text if not self.lower else token.text.lower() for token in doc if "//" not in token.text and "---" not in token.text]
    
    def divide_by_special_tokens(self, pattern, text):
        match_iter = re.findall(pattern, text)
        if not match_iter:
            return text
        ret = []
        st_idx = 0
        for string in match_iter:
            ed_idx = text.find(string, st_idx)
            if ed_idx > st_idx:
                ret.append(text[st_idx:ed_idx])
                ret.append(string)
                st_idx = ed_idx + len(string)
            else:
                ret.append(string)
                st_idx += len(string)
        
        if st_idx < len(text):
            ret.append(text[st_idx:])
        return ret
    
    def cut_word_version2(self, text):
        ne_symbols = ["#0#", "#1#", "#2#", "#3#", "#4#", "#5#"]

        text = self.divide_by_special_tokens(r"#\d#", text)
        final_text = []
        if isinstance(text, str):
            final_text = self.divide_by_special_tokens(r"@|\+|/|\||:|\"", text)
        elif isinstance(text, list):
            for item in text:
                words = self.divide_by_special_tokens(r"@|\+|/|\||:|\"", item)
                if isinstance(words, str):
                    final_text.append(words)
                elif isinstance(words, list):
                    final_text.extend(words)
        
        if isinstance(final_text, str):
            return self.really_cut_word(final_text)
        else:
            ret = []
            for item in final_text:
                if item in ne_symbols:
                    ret.append(item)
                    continue
                ret.extend(self.really_cut_word(item))
            return ret
    
    def refine_token(self, token):
        ret = []
        match_iter = re.findall(r"-|\.|/|:|\(|\)|,|&|!|=|'|\?|\[|\]|\\|>|（|）|#\d#", token)
        if not match_iter:
            return [token]
        st_idx = 0
        # print(match_iter.count())
        for string in match_iter:
            ed_idx = token.find(string, st_idx)
            if ed_idx > st_idx:
                ret.append(token[st_idx:ed_idx])
                ret.append(string)
                st_idx = ed_idx + len(string)
            else:
                ret.append(string)
                st_idx += len(string)
        if st_idx < len(token):
            ret.append(token[st_idx:])
        return ret
    
    def cut_word(self, text):
        """
        这里必须要通过check_dataset_valid的检查
        此处控制是否取小、以及是否取原型
        """

        tmp = re.split(r"\s+", text)
        ret = []
        for token in tmp:
            ret.extend(self.refine_token(token if not self.lower else token.lower()))
        return ret
     
    def refine_filename_or_title_for_readme(self, text):
        text = re.sub(r"#\[\d\]#", "", text)
        text = re.sub(r"#\d#", "", text)
        text = re.sub(r"\[\d\]", "", text)
        return text
    
    def extract_ne(self, sent):
        length = len(sent)
        idx = 0
        tokens = []
        labels = []
        label_symbols = []
        for i in range(self.n_categories):
            label_symbols.append("#{}#".format(i))
        while idx < length:
            if sent[idx] in label_symbols:
                st_idx = idx
                ed_idx = st_idx + 1
                while ed_idx < length and sent[ed_idx] != sent[st_idx]:
                    ed_idx += 1

                category = int(sent[st_idx][1]) * self.n_labels
                # print(st_idx, ed_idx, sent[st_idx + 1: ed_idx])
                tokens.extend(sent[st_idx + 1: ed_idx])
                if ed_idx - st_idx == 2:
                    labels.append(category + 2)
                else:
                    labels.append(category + 0)
                    for _ in range(st_idx + 2, ed_idx):
                        labels.append(category + 1)
                
                idx = ed_idx
            else:
                tokens.append(sent[idx])
                labels.append(self.n_categories * self.n_labels)
            idx += 1
        return tokens, labels

    def build_labeled_data(self, dst_file):
        def do_it(src_file_path, outf, split_cnt_limit):
            with open(src_file_path, "r", encoding="utf-8") as inf:
                for _, line in enumerate(inf):
                    if self.line_has_chinese(line): # 句子中出现中文
                        continue
                    splits = line.strip().split("\t")
                    if len(splits) < split_cnt_limit:
                        continue
                    filename, content = splits[0], splits[-1]
                    sent = self.cut_word_version2(content)
                    if not self.sent_has_min_length(sent):
                        continue
                    tokens, labels = self.extract_ne(sent)
                    if len(tokens) != len(labels):
                        print(line)
                        sys.exit(0)
                    outf.write("{}\t{}\t{}\n".format(filename, json.dumps(tokens, ensure_ascii=False), json.dumps(labels, ensure_ascii=False)))
        
        outf = open(dst_file, "w", encoding="utf-8")
        
        # readme
        do_it(self.src_labeled_readme, outf, split_cnt_limit=3)

        # gitdes
        do_it(self.src_labeled_gitdes, outf, split_cnt_limit=2)
        outf.close()
    
    def build_unlabeled_data(self, dst_file="unlabeled"):
        def do_it(src_file_path, outf, split_cnt_limit):
            with open(src_file_path, "r", encoding="utf-8") as inf:
                for line in inf:
                    splits = line.strip().split("\t")
                    if len(splits) < split_cnt_limit:
                        continue
                    filename, content = splits[0], splits[-1]
                    sent = self.cut_word_version2(content)
                    if not self.sent_has_min_length(sent):
                        continue
                    outf.write("{}\t{}\n".format(filename, json.dumps(sent, ensure_ascii=False)))
        
        outf = open( dst_file, "w", encoding="utf-8")
        
        # readme
        do_it(self.src_unlabeled_readme, outf, split_cnt_limit=3)

        # gitdes
        do_it(self.src_unlabeled_gitdes, outf, split_cnt_limit=2)
        outf.close()
    
    def test(self, text):
        sent = self.cut_word_version2(text)
        if not self.sent_has_min_length(sent):
            return
        tokens, labels = self.extract_ne(sent)
        print(tokens, labels, "\n", len(tokens), len(labels))
        
if __name__ == "__main__":
    src_labeled_readme = os.path.join(LABELING_DATA_DIR, "labeled_readme")
    src_labeled_gitdes = os.path.join(LABELING_DATA_DIR, "labeled_gitdes")
    src_unlabeled_readme = os.path.join(LABELING_DATA_DIR, "unlabeled_readme")
    src_unlabeled_gitdes = os.path.join(LABELING_DATA_DIR, "unlabeled_gitdes")

    n_categories = 6
    n_labels = 3
    if TEXT_PREPROCESS_CHOICE == "lemma":
        lower, lemma = True, True
    elif TEXT_PREPROCESS_CHOICE == "uncased":
        lower, lemma = False, False
    else:
        lower, lemma = True, False
    nedb = NEDatasetBuilder(src_labeled_readme, src_labeled_gitdes, src_unlabeled_readme, src_unlabeled_gitdes, n_categories=n_categories, n_labels=n_labels, lower=lower, lemma=lemma)
    nedb.build_labeled_data(LABELED_NER_DATA)
    nedb.build_unlabeled_data(UNLABELED_NER_DATA)