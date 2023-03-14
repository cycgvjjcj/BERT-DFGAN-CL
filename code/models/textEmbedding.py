from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn


class myBertModel(nn.Module):
    def __init__(self,embed,words_num):
        super().__init__()
        self.embedder = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)  # 返回隐层bert-base-multilingual-cased  bert-base-chinese
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # distilbert-base-uncased
        self.words_num = words_num
        self.linear = nn.Linear(in_features=768,out_features=embed)
        print("bert init")

    def embd(self, tokens_id):
        tokens_id_tensor = tokens_id.unsqueeze(0)  # torch.tensor(tokens_id).unsqueeze(0)
        outputs = self.embedder(tokens_id_tensor)
        sentence = outputs[1]
        sentence = sentence.squeeze(0)

        words = outputs[0]
        words = words.squeeze(0)
        wordsm = words.t()
        return sentence, wordsm

    def encode(self, input):
        input = input.unsqueeze(0)
        output = self.embedder(input)
        return output
    def tensor_tokens_id(self,inputs):
        tokens = self.tokenizer.tokenize(inputs)  #分词
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens_id
    def id_tensor_tokens(self,tokens_id):
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_id)
        return tokens

    def forward(self, captions):
        # print("bert forward")
        # print(captions.size())
        i = 0
        sent_emb = np.zeros((32, 256), dtype='float32')
        sent_emb = torch.as_tensor(torch.from_numpy(sent_emb), dtype=torch.float32)
        words_emb = np.zeros((32, 256, 18), dtype='float32')
        words_emb = torch.as_tensor(torch.from_numpy(words_emb), dtype=torch.float32)
        sent_emb = sent_emb.cuda()
        words_emb = words_emb.cuda()
        while i < 32:
            real_word_emb = np.zeros((18, 256), dtype='float32')
            real_word_emb = torch.as_tensor(torch.from_numpy(real_word_emb), dtype=torch.float32)
            captions1 = captions[i].cpu()
            captions1 = captions1.numpy().tolist()
            captions1 = torch.LongTensor(captions1)
            captions1 = captions1.cuda()
            sent_emb_tmp,words_emb_tmp = self.embd(captions1)#[768],[768,18]
            sent_emb[i] = self.linear(sent_emb_tmp)#[256]
            j = 0
            words_emb1 = words_emb_tmp.permute(1, 0)#[18,768]
            while j < self.words_num:
                # words_emb1 =words_emb[i].permute(1,0)
                words_emb2 = words_emb1[j]#[768]
                # print(words_emb2.size())
                words_emb2 = self.linear(words_emb2)#[256]
                real_word_emb[j]=words_emb2
                # words_emb[i] = words_emb1
                j = j + 1
            words_emb[i] = real_word_emb.permute(1,0)
            i = i + 1
        # print("emb_size:")
        # print(sent_emb.size())
        # print(words_emb.size())
        # pass
        return words_emb,sent_emb