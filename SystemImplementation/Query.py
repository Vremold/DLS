import sys
import json
import os
import pickle
import re

from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch
from torch.optim import Adam
import spacy
from torchcrf import CRF

TARGET_PAD = 19
CONTEXT_PAD = 0

class NERModel(nn.Module):
    def __init__(self, n_classes, model_name):
        super().__init__()
        self.total_kinds = n_classes
        self.bert = BertModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_size=1024 + 128*3, hidden_size=128, bidirectional=True, batch_first=True)
        self.predict = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128*2, out_features=n_classes)
        )
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=3, padding=2, stride=1, dilation=1)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=3, padding=6, stride=1, dilation=3)
        self.conv5 = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=3, padding=10, stride=1, dilation=5)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.loss = nn.NLLLoss(ignore_index=TARGET_PAD)
        self.crf = CRF(num_tags=self.total_kinds, batch_first=True)
    
    def forward(self, tokens, labels=None, is_train=False):
        mask = tokens != CONTEXT_PAD
        bert_output = self.bert(input_ids=tokens, attention_mask=mask)[0] # bsz * length * embedding
        conv_input = bert_output.permute(0, 2, 1)    #bsz * hidden * length
        conv_res1 = self.conv1(conv_input)[:,:,:-2]      #bsz * out_channel * length
        conv_res3 = self.conv3(conv_input)[:,:,:-6] 
        conv_res5 = self.conv5(conv_input)[:,:,:-10]
        conv_res = torch.cat((conv_res1, conv_res3, conv_res5), dim=1).permute(0, 2, 1)     #bsz * length * 3out_channel
        rnn_input = torch.cat((conv_res, bert_output), dim=2)
        lstm_output = self.lstm(rnn_input)[0]  # bsz * length * 2hidden
        logits = self.predict(lstm_output)
        
        label_mask = tokens != CONTEXT_PAD
        # masked_labels = labels.long() * label_mask.long()

        if not is_train:
            prediction = self.crf.decode(logits[:, 1:, :], mask=label_mask[:, 1:])  # Val/Test: Path Generation: Mask CLS/SEP
            return prediction


class DataPreparer(object):
    def __init__(self, pretrained_model):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        pass

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
    
    def really_cut_word(self, sent):
        doc = self.nlp(sent)
        return [token.text for token in doc if "//" not in token.text and "---" not in token.text]
    
    def cut_word(self, text):
        splits = self.divide_by_special_tokens(r"@|\+|/", text)
        if isinstance(splits, str):
            return self.really_cut_word(splits)
        elif isinstance(splits, list):
            ret = []
            for item in splits:
                ret.extend(self.really_cut_word(item))
            return ret
        else:
            print("你在干什么！")
            sys.exit(0)
        
    def prepare_for_pred(self, sent:list):
        sent_tokens = []
        sent_piece2word = []
        word_idx = 0
        for word in sent:
            tokens = self.tokenizer.tokenize(word)
            sent_piece2word.extend([word_idx] * len(tokens))
            word_idx += 1
            sent_tokens.extend(tokens)
        
        sent_tokens = self.tokenizer.convert_tokens_to_ids(sent_tokens)
        sent_tokens = self.tokenizer.build_inputs_with_special_tokens(sent_tokens)
        sent_piece2word.insert(0, -1)
        sent_piece2word.append(-1)

        tokens = torch.LongTensor([sent_tokens[:512]])

        return tokens, sent_piece2word
    
    def prepare(self, text):
        sent = self.cut_word(text)
        tokens, sent_piece2word = self.prepare_for_pred(sent)
        return sent, tokens, sent_piece2word


class PredictionParser(object):
    def __init__(self):
        pass

    def parse_labels(self, sent, tokens, prediction, piece2word):
        # 除去后缀0
        idx = 0
        while idx < len(tokens) and tokens[idx] != 0:
            idx += 1
        tokens = tokens[1:idx-1]
        prediction = prediction[:-1]
        piece2word = piece2word[1:-1]
        
        labels = [18] * len(sent)
        idx = 0
        while idx < len(prediction):
            if prediction[idx] % 3 == 0 and prediction[idx] != 18:
                category = prediction[idx] // 3
                ed_idx = idx + 1
                while (ed_idx < len(prediction) and prediction[ed_idx] == prediction[idx] + 1):
                    ed_idx += 1
                ed_idx -= 1
                st_word_idx = piece2word[idx]
                ed_word_idx = piece2word[ed_idx]
                # print(idx, ed_idx, st_word_idx, ed_word_idx, category)

                if st_word_idx == ed_word_idx:
                    labels[st_word_idx] = category * 3 + 2
                else:
                    labels[st_word_idx] = category * 3
                    for i in range(st_word_idx + 1, ed_word_idx + 1):
                        labels[i] = category * 3 + 1
                idx = ed_idx + 1
            elif prediction[idx] % 3 == 2:
                labels[piece2word[idx]] = prediction[idx]
                idx += 1
            else:
                idx += 1
        return labels

    def splice_ne(self, tokens):
        ne = " ".join(tokens)
        ne = re.sub(r" -", "-", ne)
        ne = re.sub(r"- ", "-", ne)
        ne = re.sub(r" /", "/", ne)
        ne = re.sub(r"/ ", "/", ne)
        ne = re.sub(r" \.", ".", ne)
        ne = re.sub(r"\. ", ".", ne)
        ne = re.sub(r" \(", "(", ne)
        ne = re.sub(r"\( ", "(", ne)
        ne = re.sub(r" \)", ")", ne)
        ne = re.sub(r"\) ", ")", ne)
        ne = re.sub(r"@ ", "@", ne)
        ne = re.sub(r" @", "@", ne)
        return ne

    def parse(self, sent, tokens, prediction, piece2word):
        def has_hyperlink(text):
            return "//" in text
        
        labels = self.parse_labels(sent, tokens, prediction, piece2word)
        
        ret = {0: set(), 1: set(), 2: set(), 3: set(), 4: set(), 5: set()}
        idx = 0
        while idx < len(labels):
            if labels[idx] % 3 == 0 and labels[idx] != 18:
                category = labels[idx] // 3
                ed_idx = idx + 1
                while ed_idx < len(labels) and labels[ed_idx] == labels[idx] + 1:
                    ed_idx += 1
                ne = self.splice_ne(sent[idx: ed_idx])
                if not has_hyperlink(ne):
                    ret[category].add(ne)
                idx = ed_idx
            elif labels[idx] % 3 == 2:
                category = labels[idx] // 3
                if not has_hyperlink(sent[idx]):
                    ret[category].add(sent[idx])
                idx += 1
            else:
                idx += 1
        return ret


class QueryPredNE(object):
    def __init__(
        self, 
        token_dict_file, 
        pretrained_model="../NER/pretrained/bert_large_cased",
        trained_model="../NER/mdl/bert_large_cased.mdl"
        ):
        self.model = NERModel(n_classes=19, model_name=pretrained_model)
        self.model.load_state_dict(torch.load(trained_model, map_location="cpu"))
        self.dp = DataPreparer(pretrained_model=pretrained_model)
        self.pp = PredictionParser()
    
    def predict(self, text):
        sent, tokens, piece2word = self.dp.prepare(text)
        
        with torch.no_grad():
            prediction = self.model(tokens=tokens, is_train=False)
        
        tokens = tokens.numpy()[0]
        nes = self.pp.parse(sent, tokens, prediction[0], piece2word)
        return nes

class QueryWordExtractor(object):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract_useful_words(self, text):
        doc = self.nlp(text)
        allowed_pos = ["ADJ", "NOUN"]
        ret = dict()
        for token in doc:
            if token.pos_ in allowed_pos:
                if token.lemma_ not in ret:
                    ret[token.lemma_] = 0
                ret[token.lemma_] += 1
        return ret