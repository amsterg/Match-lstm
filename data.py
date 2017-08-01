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

with open(train_file,'r') as f:
    train_data = json.load(f)

with open("train.csv",'w') as csvfile:
    fieldnames = ['Title','Id','context','question','answers']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(train_data['data'])):
        title = train_data['data'][i]['title']
        for j in range(len(train_data['data'][i]['paragraphs'])):
            #for k in range(len(train_data['data'][i]['paragraphs'][j])):
            context = train_data['data'][i]['paragraphs'][j]['context']
            for k in range(len(train_data['data'][i]['paragraphs'][j]['qas'])):
                question = train_data['data'][i]['paragraphs'][j]['qas'][k]['question']
                answers = train_data['data'][i]['paragraphs'][j]['qas'][k]['answers']
                id = train_data['data'][i]['paragraphs'][j]['qas'][k]['id']
                writer.writerow({'Title': title, 'Id': id,'context': context,'question': question, 'answers': answers})
