# -*- coding: utf-8 -*-
import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
import torch
import math
import textattack
import random


#torch.cuda.is_available = lambda : False
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

textattack.shared.utils.device = 'cuda:1'

class ClassificationModel(nn.Module):
    def __init__(self, model, pos_prompt, neg_prompt):
        super(ClassificationModel, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.model.eval()
        self.pos_prompt = pos_prompt
        self.neg_prompt = neg_prompt

    def score(self, prompt, sentence, model):
        tokenized_prompt = self.tokenizer.encode(prompt , max_length=1024, truncation=True, return_tensors='pt').to('cuda:1')
        tokenized_all = self.tokenizer.encode(prompt + ' ' + sentence, max_length=1024, truncation=True, return_tensors='pt').to('cuda:1')

        loss=model(tokenized_all, labels=tokenized_all).loss - model(tokenized_prompt, labels=tokenized_prompt).loss*len(tokenized_prompt[0])/len(tokenized_all[0])

        return math.exp(loss)
    

    def forward(self, sentence):
        pos = self.score(self.pos_prompt, sentence, self.model)

        neg = self.score(self.neg_prompt, sentence, self.model)

        result = torch.FloatTensor([1-neg, 1-pos])
        result = torch.softmax(result, 0)


        if abs(result[0].item()+result[1].item()-1) >= 1e-6:
            print('detected something')
            result = torch.FloatTensor([1,0])
        return torch.softmax(result, 0)


class TorchTokenizer(GPT2Tokenizer):
    def __init__(self):
        super(TorchTokenizer, self).__init__()


class CustomWrapper(textattack.models.wrappers.ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, list_of_texts):
        results = []
        for text in list_of_texts:
          results.append(self.model(text))

        return torch.stack(results)



model = ClassificationModel('model train_without_labels.txt', 'I loved this movie!', 'I hated this movie!').to('cuda:1')

class_model = CustomWrapper(model)



from textattack.datasets import Dataset
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack import Attacker, AttackArgs


attack = TextFoolerJin2019.build(class_model)
attack.cuda_()

dataset = []
with open('data/test.txt', 'r') as f:
    for line in f:
        dataset.append((' '.join(line.split(' ')[1:]).replace('\n', ''), int(line.split(' ')[0])))

#df = pd.read_csv('data/amz_test.csv')
#for index, row in df.iterrows():
#    dataset.append((row['text'], row['label']))
"""
with open('data/yelp_positive_test.txt', 'r') as f:
    for line in f:
      dataset.append((line.replace('\n', ' '), 1))

with open('data/yelp_negative_test.txt', 'r') as f:
    for line in f:
      dataset.append((line.replace('\n', ' '), 0))
"""
random.shuffle(dataset)

attacker = Attacker(attack, textattack.datasets.Dataset(dataset[:100]), AttackArgs(num_examples=100))
attacker.attack_dataset()
