#import tensorflow
import os
import json
import csv
import nltk
import collections
import sys
import argparse
parser = argparse.ArgumentParser(description='Process data,create wordvecs from glove',prog = "data.py")
parser.add_argument('--session', type = str, choices = ['train','dev'],help='choose to process over train/dev.')
parser.add_argument('--mode',type = str,choices=['id','vec',''],
                help="to prepare direct wordvecs or through intermediate ids.")
args = parser.parse_args()

session = args.session
print (session+" session")
mode = args.mode

train_file = "../../data/train-v1.1.json"
dev_file = "../../data/dev-v1.1.json"
json_file = ("../../data/"+session+"-v1.1.json")
glove_vecs_50d = "../../data/glove.6B/glove.6B.50d.txt"
glove_vecs_sample = "../../data/glove.6B/sample_glove_50d.txt"
glove_vecs = glove_vecs_50d

glove_dict = collections.OrderedDict()
glove_dict_rev = collections.OrderedDict()

with open(glove_vecs) as glove_file:
    glove = glove_file.readlines()


def data_process(session,mode):
    with open(json_file,'r') as f:
        train_data = json.load(f)

    with open(session+"_words.csv",'w') as csvfile:
        fieldnames = ['Title','Id','context','question','answers']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(train_data['data'])):
            print ("Processing "+str(i)+"Title")
            title = train_data['data'][i]['title']
            for j in range(len(train_data['data'][i]['paragraphs'])):
                #for k in range(len(train_data['data'][i]['paragraphs'][j])):
                context = create_vec((train_data['data'][i]['paragraphs'][j]['context']),mode)
                #print (context[0:2])
                for k in range(len(train_data['data'][i]['paragraphs'][j]['qas'])):
                    question = create_vec(train_data['data'][i]['paragraphs'][j]['qas'][k]['question'],mode)
                    answers = train_data['data'][i]['paragraphs'][j]['qas'][k]['answers']
                    id = train_data['data'][i]['paragraphs'][j]['qas'][k]['id']
                    answers_dict = {}
                    for l in range(len(answers)):
                        #print (answers[l]['answer_start'],answers[l]['answer_start']+len(answers[l]['text']))
                        answers_dict[answers[l]['answer_start']] = create_vec(answers[l]['text'],mode)

                        #print (context[answers[l]['answer_start']:answers[l]['answer_start']+len(answers[l]['text'])])
                        writer.writerow({'Title': title, 'Id': id,'context': context,'question': question, 'answers': answers_dict})


def word2id(word):
    return glove_dict[word]
def id2word(id):
    return glove_dict_rev[id]

def word2vec(word):
    try:
        return glove_dict[word]
    except Exception:
        return [0.0 for _ in range(50)]

def vec2word(vec):
    return glove_dict_rev[vec]
def create_vec(sent,mode):
    if mode == 'id':
        return 1
    elif mode == 'vec':
        print ([word2vec(token) for token in nltk.word_tokenize(sent.lower())])
    else:
        return sent
def main():
    if mode == 'id':
        print ("id mode...")
        for word in range(len(glove)):
            glove_dict[glove[word].split()[0]] = (word+1)

        for key in list(glove_dict.keys()):
            glove_dict_rev[glove_dict[key]] = key
        data_process(session,mode)

    elif mode == 'vec':
        print ("vec mode....")
        for word in range(len(glove)):
            glove_dict[glove[word].split()[0]] = glove[word].split()[1:]

        for key in list(glove_dict.keys()):
            glove_dict_rev[str(glove_dict[key])] = key
            #glove_dict_rev[glove_dict[key]] = key
        data_process(session,mode)

    else :
        print("Creating data alone...")
        data_process(session,mode)
    #data_process()


if __name__ == '__main__':
    main()
