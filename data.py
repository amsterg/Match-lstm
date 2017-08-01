#import tensorflow
import os
import json
import csv
import nltk
import collections
train_file = "../../data/train-v1.1.json"
dev_file = "../../data/dev-v1.1.json"
glove_vecs_50d = "../../data/glove.6B/glove.6B.50d.txt"
glove_vecs_sample = "../../data/glove.6B/sample_glove_50d.txt"
glove_vecs = glove_vecs_50d
glove_dict = {}
with open(glove_vecs) as glove_file:
    glove = glove_file.readlines()

glove_dict = collections.OrderedDict()
glove_dict_rev = collections.OrderedDict()
#for word in range(len(glove)):
#    glove_dict[glove[word].split()[0]] = glove[word].split()[1:]
for word in range(len(glove)):
    glove_dict[glove[word].split()[0]] = (word+1)

for key in list(glove_dict.keys()):
    glove_dict_rev[glove_dict[key]] = key

with open(train_file,'r') as f:
    train_data = json.load(f)

def word2id(str):
    str_tokens = nltk.word_tokenize(str)
    str_tokens_id = []
    for token in range(len(str_tokens)):
        try:
            str_tokens_id += [glove_dict[str_tokens[token]]]
        except KeyError:
            str_tokens_id += [0]
    return str_tokens_id

def id2word(str):
    for token in range(len(str)):
        try:
            str_tokens_id += [glove_dict_rev[str_tokens[token]]]
        except KeyError:
            str_tokens_id += [0]
    return str_tokens_id
#with open(glove_vecs,'r') as f:
#    glove_data = f.read()
with open("train.csv",'w') as csvfile:
    fieldnames = ['Title','Id','context','question','answers']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(train_data['data'])):
        #title = word2id((train_data['data'][i]['title']))
        for j in range(len(train_data['data'][i]['paragraphs'])):
            #for k in range(len(train_data['data'][i]['paragraphs'][j])):
            context = word2id(train_data['data'][i]['paragraphs'][j]['context'])
            """

            for k in range(len(train_data['data'][i]['paragraphs'][j]['qas'])):
                question = train_data['data'][i]['paragraphs'][j]['qas'][k]['question']
                answers = train_data['data'][i]['paragraphs'][j]['qas'][k]['answers']
                id = train_data['data'][i]['paragraphs'][j]['qas'][k]['id']
                writer.writerow({'Title': title, 'Id': id,'context': context,'question': question, 'answers': answers})
                """
            writer.writerow({'context': context})
