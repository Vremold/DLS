import json
import argparse
import os
import pickle
import sys
from tqdm import tqdm

from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchcrf import CRF

from ..config import TEXT_PREPROCESS_CHOICE, BERT_KIND

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--n_classes", default=19, type=int)
parser.add_argument("--n_categories", default=6, type=int)
parser.add_argument("--learning_rate", default=1e-5, type=float, help="Batch size" )

TARGET_PAD = 19
CONTEXT_PAD = 0

class MyTrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, token_dict_file="./data/token_dict.pkl"):
        self.data = []
        token_dict = {}
        with open(token_dict_file, "rb") as inf:
            token_dict = pickle.load(inf)
        with open(data_path, "r", encoding="utf-8") as inf:
            for line in inf:
                splits = line.strip().split("\t")
                filename = splits[0]
                sent = json.loads(splits[1])
                nes = json.loads(splits[2])
                
                sent_tokens = []
                sent_labels = []
                sent_piece2word = []

                raw_tokens = [token_dict[word] for word in sent]

                word_idx = 0
                for word, label in zip(sent, nes):
                    tokens = tokenizer.tokenize(word)
                    sent_piece2word.extend([word_idx] * len(tokens))
                    word_idx += 1
                    sent_tokens.extend(tokens)
                    if label % 3 == 0 and label != 18: # 此处需要注意！！
                        sent_labels.append(label)
                        for _ in range(len(tokens) - 1):
                            sent_labels.append(label + 1)
                    elif label % 3 == 1:
                        for _ in range(len(tokens)):
                            sent_labels.append(label)
                    elif label % 3 == 2:
                        if len(tokens) == 1:
                            sent_labels.append(label)
                        else:
                            sent_labels.append(label - 2)
                            for _ in range(len(tokens) - 1):
                                sent_labels.append(label - 1)
                    else:
                        sent_labels.extend([18] * len(tokens))
                
                sent_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)
                sent_tokens = tokenizer.build_inputs_with_special_tokens(sent_tokens)
                sent_labels = [TARGET_PAD] + sent_labels + [TARGET_PAD]
                sent_piece2word.insert(0, -1)
                sent_piece2word.append(-1)

                sent_tokens = torch.LongTensor(sent_tokens)
                sent_labels = torch.LongTensor(sent_labels)
                self.data.append({
                    "filename": filename,
                    "raw_tokens": raw_tokens,
                    "tokens": sent_tokens,
                    "labels": sent_labels,
                    "piece2word": sent_piece2word
                })
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    tokens = [torch.LongTensor(sample["tokens"][:512]) for sample in batch]
    labels = [torch.LongTensor(sample["labels"][:512]) for sample in batch]
    raw_tokens = [sample["raw_tokens"] for sample in batch]
    filenames = [sample["filename"] for sample in batch]
    piece2word = [sample["piece2word"] for sample in batch]

    tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=CONTEXT_PAD)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=TARGET_PAD)

    return {
        "raw_tokens": raw_tokens,
        "tokens": tokens,
        "labels": labels,
        "filenames": filenames,
        "piece2word": piece2word
    }

class NERModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.total_kinds = args.n_classes
        self.bert = BertModel.from_pretrained(args.pretrained_model_name)
        if args.bert_kind == "large":
            self.lstm = nn.LSTM(input_size=1024 + 128 * 3, hidden_size=128, bidirectional=True, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=768 + 128 * 3, hidden_size=128, bidirectional=True, batch_first=True)
        self.predict = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128*2, out_features=args.n_classes)
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
        # print(conv_input.shape, conv_res1.shape, conv_res3.shape, conv_res5.shape)
        conv_res = torch.cat((conv_res1, conv_res3, conv_res5), dim=1).permute(0, 2, 1)     #bsz * length * 3out_channel
        rnn_input = torch.cat((conv_res, bert_output), dim=2)
        lstm_output = self.lstm(rnn_input)[0]  # bsz * length * 2hidden
        # lstm_output = self.lstm(bert_output)[0] 
        logits = self.predict(lstm_output)
        
        label_mask = labels != TARGET_PAD
        masked_labels = labels.long() * label_mask.long()

        if not is_train:
            prediction = self.crf.decode(logits[:, 1:, :], mask=label_mask[:, 1:])  # Val/Test: Path Generation: Mask CLS/SEP
            return prediction

        loss = self.crf(logits[:, 1:, :], masked_labels[:, 1:], mask=label_mask[:, 1:], reduction='mean') # Train: Calculating Loss: Mask CLS/SEP
        return torch.neg(loss)

def accuracy(gold, pred, n_categories):
    def do_it(vec, dic):
        idx = 0
        while idx < len(vec):
            if vec[idx] % 3 == 0 and vec[idx] != 18:
                cate = vec[idx] // 3
                ed_idx = idx + 1
                while ed_idx < len(vec) and vec[ed_idx] == vec[idx] + 1:
                    ed_idx += 1
                ed_idx -= 1
                dic[cate].add(json.dumps([idx, ed_idx]))
                idx = ed_idx + 1
            elif vec[idx] % 3 == 2:
                cate = vec[idx] // 3
                dic[cate].add(json.dumps([idx, idx]))
                idx += 1
            else:
                idx += 1
    gold = gold[1:]
    gold_nes = {}
    pred_nes = {}
    for cate in range(n_categories):
        gold_nes[cate] = set()
        pred_nes[cate] = set()
    
    do_it(gold, gold_nes)
    do_it(pred, pred_nes)
    ne_cnt = 0
    right_cnt = 0
    pred_cnt = 0
    for cate in gold_nes:
        ne_cnt += len(gold_nes[cate])
        pred_cnt += len(pred_nes[cate])
        right_cnt += len(pred_nes[cate].intersection(gold_nes[cate]))
    return right_cnt, ne_cnt, pred_cnt
    

def validate(model, validate_dataloader, args):
    total_nes = 0
    true_nes = 0
    pred_nes = 0

    with torch.no_grad():
        for batch in tqdm(validate_dataloader):
            prediction = model(tokens=batch["tokens"].cuda(), labels=batch["labels"].cuda(), is_train=False)
            gold = batch["labels"].cpu().numpy()
            
            bsz = len(batch["tokens"])

            for i in range(bsz):
                right_cnt, ne_cnt, pred_cnt = accuracy(gold[i], prediction[i], 6)
                total_nes += ne_cnt
                true_nes += right_cnt
                pred_nes += pred_cnt
    precision = true_nes / (pred_nes + 0.0000001)
    recall = true_nes / (total_nes + 0.0000001)
    return precision, recall, 2 * precision * recall / (precision + recall + 0.0000001)
    # print("[Validate] Accu: {}".format(true_nes/total_nes))       
    pass

def train(train_dataset, validate_dataset, args, model=None):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    best_f1 = 0
    
    if not model:
        model = NERModel(args).cuda()

    para_dict = dict(model.named_parameters())
    paras_new = []
    for k, v in para_dict.items():
        if 'predict' in k or "crf" in k:
            paras_new += [{'params': [v], 'lr': 100*args.learning_rate}]
        else:
            paras_new += [{'params': [v], 'lr': args.learning_rate}]
    optimizer = Adam(paras_new)

    # print("[Train] starting training")
    for epoch in range(args.epochs):
        step = 1
        loss = None
        for batch in tqdm(train_dataloader):
            # print(len(batch["tokens"]))
            loss = model(tokens=batch['tokens'].cuda(), labels=batch['labels'].cuda(), is_train=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 10 == 0:
                print("[Train]:Epoch: {}, Batch: {}, Loss: {}\n".format(epoch, step, loss.cpu().detach().numpy()))
        precision, recall, f1 = validate(model, validate_dataloader, args)
        print("[Validate]:Epoch: {}, Precision: {}, Recall: {}, F1: {}\n".format(epoch, precision, recall, f1))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.mdl_save_path)
    return model
            

if __name__ == "__main__":
    args = parser.parse_args()
    args.bert_kind = BERT_KIND
    args.pretrained_model_name = "./pretranined/bert_{}_{}".format(args.bert_kind, TEXT_PREPROCESS_CHOICE)
    args.mdl_save_path = "./builds/bert_{}_{}.mdl".format(args.bert_kind, TEXT_PREPROCESS_CHOICE)
    
    print("Train executed with parameters", args)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
    train_dataset = MyTrainDataset("./data/train.txt", tokenizer)
    validate_dataset = MyTrainDataset("./data/valid.txt", tokenizer)
    model = train(train_dataset, validate_dataset, args)