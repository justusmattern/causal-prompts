import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch import nn
import math

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to('cuda:3')

model.eval()
ce_loss = nn.CrossEntropyLoss(reduction='none')

def score(prompt, sentence, model):
    tokenized_prompt = tokenizer.encode(prompt , max_length=1024, truncation=True, return_tensors='pt').to('cuda:3')
    tokenized_all = tokenizer.encode(prompt + ' ' + sentence, max_length=1024, truncation=True, return_tensors='pt').to('cuda:3')

    loss=model(tokenized_all, labels=tokenized_all).loss - model(tokenized_prompt, labels=tokenized_prompt).loss*len(tokenized_prompt[0])/len(tokenized_all[0])

    return math.exp(loss)


def classify(pos_prompts, neg_prompts, sentence, model):
    pos_loss = 0
    neg_loss = 0

    for prompt in pos_prompts:
        pos_loss += score(prompt, sentence, model)
    for prompt in neg_prompts:
        neg_loss += score(prompt, sentence, model)

    if pos_loss > neg_loss:
      return 0
    else:
      return 1

dataset = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        dataset.append((' '.join(line.split(' ')[1:]).replace('\n', ''), int(line.split(' ')[0])))


#positive_prompts = ["This was really good.", "This was awesome!", "I really liked this!", "Wow!", "What a great movie!", "I liked this movie."]
#negative_prompts = ["This was really bad.", "This was terrible!", "I really hated this!", "Ugh!", "What a horrible movie!", "I did not like this movie."]

positive_prompts = ["This film was fantastic!"]
negative_prompts = ["This film was boring."]

rights = 0
for i, (text, label) in enumerate(dataset[:300]):
    print(i)
    pred = classify(positive_prompts, negative_prompts, text, model)

    if pred == label:
      rights += 1

print('accuracy', rights/300)

