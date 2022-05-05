import json
import argparse
import os
import pickle

from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchcrf import CRF

from ..config import TEXT_PREPROCESS_CHOICE, BERT_KIND

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
parser = argparse.ArgumentParser()

parser.add_argument("--epochs", default=2, type=int)
parser.add_argument("--n_classes", default=19, type=int)
parser.add_argument("--batch_size", default=32, type=int)

TARGET_PAD = 19
CONTEXT_PAD = 0

class NERModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.total_kinds = args.n_classes
        self.bert = BertModel.from_pretrained(args.pretrained_model_name)
        self.lstm = nn.LSTM(input_size=1024 + 128 * 3, hidden_size=128, bidirectional=True, batch_first=True)
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
        
        label_mask = tokens != CONTEXT_PAD
        # masked_labels = labels.long() * label_mask.long()

        if not is_train:
            prediction = self.crf.decode(logits[:, 1:, :], mask=label_mask[:, 1:])  # Val/Test: Path Generation: Mask CLS/SEP
            return prediction

class MyTestDataset(Dataset):
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
                
                sent_tokens = []
                sent_piece2word = []

                raw_tokens = [token_dict[word] for word in sent]

                word_idx = 0
                for word in sent:
                    tokens = tokenizer.tokenize(word)
                    sent_piece2word.extend([word_idx] * len(tokens))
                    word_idx += 1
                    sent_tokens.extend(tokens)
                
                sent_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)
                sent_tokens = tokenizer.build_inputs_with_special_tokens(sent_tokens)
                sent_piece2word.insert(0, -1)
                sent_piece2word.append(-1)

                sent_tokens = torch.LongTensor(sent_tokens)
                self.data.append({
                    "filename": filename,
                    "raw_tokens": raw_tokens,
                    "tokens": sent_tokens,
                    "piece2word": sent_piece2word
                })
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    tokens = [torch.LongTensor(sample["tokens"][:512]) for sample in batch]
    raw_tokens = [sample["raw_tokens"] for sample in batch]
    filenames = [sample["filename"] for sample in batch]
    piece2word = [sample["piece2word"] for sample in batch]

    tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=CONTEXT_PAD)

    return {
        "raw_tokens": raw_tokens,
        "tokens": tokens,
        "filenames": filenames,
        "piece2word": piece2word
    }

def test(model, test_dataset, args, pred_save_file):
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    # test_result = []
    outf = open(pred_save_file, "w", encoding="utf-8")

    with torch.no_grad():
        cnt = 0
        for batch in tqdm(test_dataloader):
            prediction = model(tokens=batch["tokens"].cuda(), is_train=False)
            # prediction = torch.argmax(prediction, dim=-1).cpu().numpy()
            bsz = len(batch["tokens"])
            for i in range(bsz):
                cnt += 1
                outf.write("{}\t{}\t{}\t{}\t{}\n".format(
                    batch["filenames"][i], 
                    batch["raw_tokens"][i],
                    batch["tokens"][i].numpy().tolist(),
                    prediction[i], 
                    batch["piece2word"][i]
                ))
                outf.flush()
        outf.close()
    # return test_result

class TestResultParser(object):
    def __init__(self, test_result, token_dict_file):
        with open(token_dict_file, "rb") as inf:
            token_dict = pickle.load(inf)
        if isinstance(test_result, str):
            self.test_result = []
            with open(test_result, "r", encoding="utf-8") as inf:
                for line in inf:
                    splits = line.strip().split("\t")
                    filename = splits[0]
                    raw_tokens = json.loads(splits[1])
                    tokens = json.loads(splits[2])
                    prediction = json.loads(splits[3])
                    piece2word = json.loads(splits[4])
                    self.test_result.append((
                        filename, 
                        raw_tokens,
                        tokens,
                        prediction,
                        piece2word
                    ))
        else:
            self.test_result = test_result
        self.token_dict = {value: key for (key, value) in token_dict.items()}
        
    
    def do_it(self, raw_tokens, tokens, prediction, piece2word):
        raw_tokens = [self.token_dict[word] for word in raw_tokens]
        # 除去后缀0
        idx = 0
        while idx < len(tokens) and tokens[idx] != 0:
            idx += 1
        tokens = tokens[1:idx-1]
        piece2word = piece2word[1:-1]
        prediction = prediction[:-1]
        
        labels = [18] * len(raw_tokens)
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
        return raw_tokens, labels

    def parse(self):
        parse_result = []
        for filename, raw_tokens, tokens, prediction, piece2word in self.test_result:
            raw_tokens, labels = self.do_it(raw_tokens, tokens, prediction, piece2word)
            parse_result.append((filename, raw_tokens, labels))
        return parse_result

if __name__ == "__main__":
    args = parser.parse_args()
    args.bert_kind = BERT_KIND

    args.pretrained_model_name = "./pretrained/bert_{}_{}".format(args.bert_kind, TEXT_PREPROCESS_CHOICE)
    args.mdl_save_path = "./builds/bert_{}_{}.mdl".format(args.bert_kind, TEXT_PREPROCESS_CHOICE)
    
    print("test.py executed with parameters: ", args)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
    test_dataset = MyTestDataset("./data/test.txt", tokenizer)
    model = NERModel(args).cuda()
    model.load_state_dict(torch.load(args.mdl_load_path))
    test(model, test_dataset, args, "./data/pred.txt")
    trp = TestResultParser("./data/pred.txt", "./data/token_dict.pkl")
    
    parse_result = trp.parse()
    with open("./data/parsed_pred.txt", "w", encoding="utf-8") as inf:
        for filename, raw_tokens, labels in parse_result:
            inf.write("{}\t{}\t{}\n".format(filename, json.dumps(raw_tokens, ensure_ascii=False), json.dumps(labels, ensure_ascii=False)))