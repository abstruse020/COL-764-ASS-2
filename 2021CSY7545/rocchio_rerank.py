#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: harsh pandey
"""

import mmap
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import sys
import pandas as pd
import string
import csv
import os
import subprocess
import json
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
ps = PorterStemmer()

#%% Initializing Paths

topic_path = sys.argv[1]
top_100_doc_path = sys.argv[2]
collection_dir = sys.argv[3]
output_path = sys.argv[4]
print('op path', output_path)

# topic_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/covid19-topics.xml'
# top_100_doc_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-top-100.txt'
# collection_dir = '/home/harsh/IITD/COL764-IR/Assignment2/2020-07-16'
# output_path = '/home/harsh/IITD/COL764-IR/Assignment2/output-file_vsm.txt'


temp_folder = '/home/harsh/IITD/COL764-IR/Assignment2/temp_data/'
if len(collection_dir) > 0 and collection_dir[-1] != '/':
    collection_dir+='/'

metadata_path = collection_dir + 'metadata.csv'

relevence_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-qrels.txt'


#%% Read whole metada + introduction from pdf_json_file
## open read metadata file

def read_metadata(metadata_path, collection_dir):
    t1 = datetime.datetime.now()
    cord_uid_to_text = defaultdict(list)
    with open(metadata_path) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            # access some metadata
            cord_uid = row['cord_uid']
            title = row['title']
            abstract = row['abstract']
            pdf_path = row['pdf_json_files']
            pmc_path = row['pmc_json_files']
            # save for later usage
            cord_uid_to_text[cord_uid].append({
                'title': title,
                'abstract': abstract,
                'pdf_path': pdf_path,
                'pmc_path': pmc_path
            })
    t2 = datetime.datetime.now()
    print('Time to read metadata file' , t2-t1)
    return cord_uid_to_text
print('reading metadata')
cord_uid_to_text = read_metadata(metadata_path, collection_dir)

#%% preprocessing + vocab

def preprocessing(string, stop_word = []):
    words = word_tokenize(string, "english")
    words = [ps.stem(word) for word in words if word not in stop_word]
    return words

# Reference of vocabulary code
# https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks-python.html
class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.vocab_length = 0
    
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.vocab_length
            self.word2count[word] = 1
            self.index2word[self.vocab_length] = word
            self.vocab_length += 1
        else:
            self.word2count[word] += 1
        return
    
    def to_word(self, index):
        return self.index2word[index]
    
    def to_index(self, word):
        return self.word2index[word]
    
    def get_count(self, word):
        return self.word2count[word]
    
vocab = Vocabulary()

#%% get whole docs as string

def contents_for_cord_id(cord_id, cord_id_to_text):
    str_list = []
    for docs in cord_id_to_text[cord_id]:
        str_list.append(docs['title'])
        str_list.append(docs['abstract'])
        other_text = []
        if docs['pdf_path']:
            for pdf_json in docs['pdf_path'].split('; '):
                with open(collection_dir + pdf_json) as f_json:
                        full_text_dict = json.load(f_json)
                        for paragraph_dict in full_text_dict['body_text']:
                            paragraph_text = paragraph_dict['text']
                            if paragraph_dict:
                                other_text.append(paragraph_text)
        if other_text == [] and docs['pmc_path']:
            for pmc_json in docs['pmc_path'].split('; '):
                with open(collection_dir + pmc_json) as f_json:
                        full_text_dict = json.load(f_json)
                        for paragraph_dict in full_text_dict['body_text']:
                            paragraph_text = paragraph_dict['text']
                            if paragraph_dict:
                                other_text.append(paragraph_text)
        str_list.extend(other_text)
    return ' '.join(str_list)

#%% Read Topics
def read_topics(path, read_from = 'query'):
    content = ''
    topics = {}
    with open(path, 'r') as file:
        content = file.read()
    content = BeautifulSoup(content, 'xml')
    for topic in content.find_all('topic'):
        # print(topic['number'])
        # print(topic.find_all(read_from))
        topics[topic['number']] = topic.find_all(read_from)[0].get_text()
    return topics
topics = read_topics(topic_path)
#%% Make vector

def process_ranked_docs(path, cord_id_to_text, vocab, topics, stop_words = []):
    ## First making vocabulary and storing the processed form of documents and making vector etc
    docs_by_topic = {}
    ranked_docs_info = {}
    processed_docs = {}
    processed_queries = {}
    idf = {}
    with open(path, 'r') as file:
        with mmap.mmap(file.fileno(),
                       length=0,
                       access=mmap.ACCESS_READ
                       ) as mmap_obj:
            i = 0
            for line in iter(mmap_obj.readline, b""):
                if line.decode().strip() == '':
                    continue
                topic, _, cord_id, rank, similarity, run_id = line.decode()[:-1].split(" ")
                query = topics[topic]
                #[BookKeeping] topic wise cord_id list
                if topic not in  docs_by_topic:
                    docs_by_topic[topic] = []
                docs_by_topic[topic].append(cord_id)
                #[BookKeeping] query file information
                ranked_docs_info[(topic,cord_id)] = {'oldrank': rank, 'oldsim': similarity, 'query': query, 'run_id': run_id}
                words = preprocessing(contents_for_cord_id(cord_id, cord_id_to_text), stop_words)
                processed_docs[cord_id] = words
                
                for word in words:
                    vocab.add_word(word)
                    if word not in idf:
                        idf[word] = {}
                    if cord_id not in idf[word]:
                        idf[word][cord_id] = 1
                
                q_words = preprocessing(query, stop_words)
                processed_queries[topic] = q_words
                for q_word in q_words:
                    vocab.add_word(q_word)
                i+=1
    docs = {}
    queries = {}
    N = len(processed_docs) 
    for cord_id in processed_docs:
        docs[cord_id] = np.zeros(vocab.vocab_length)
        #Mind that the word might not be there in vocab --- Not possible
        for word in processed_docs[cord_id]:
            idf_score = np.log(N/len(idf[word]))
            index = vocab.word2index[word]
            docs[cord_id][index] += 1*idf_score
            
    for topic in processed_queries:
        queries[topic] = np.zeros(vocab.vocab_length)
        for word in processed_queries[topic]:
            index = vocab.word2index[word]
            queries[topic][index] += 1
    
    return docs, queries, ranked_docs_info, vocab, docs_by_topic, idf

print('reading and making vocab for docs')
t1 = datetime.datetime.now()
vocab = Vocabulary()
docs, queries, ranked_docs_info, vocab, docs_by_topic, idf = process_ranked_docs(
    top_100_doc_path,
    cord_uid_to_text,
    vocab,
    topics,
    stopwords.words('english'))
t2 = datetime.datetime.now()
print('time to make vocabulary taken:', t2 - t1)

#%% Cals sum of relevent and non relevent docs
def calc_sum_dr(docs, docs_by_topic):
    sum_dr = {}
    for topic in docs_by_topic:
        sum_dr[topic] = np.zeros(len(docs[docs_by_topic[topic][0]]))
        #print(len(sum_dr[topic]))
        for cord_id in docs_by_topic[topic]:
            sum_dr[topic] += docs[cord_id]
        #sum_dr[topic] /= len(sum_dr[topic])
    return sum_dr

#Just sum of call docs later while calculating subtract sum for topic i
def calc_sum_d_all(docs):
    sum_d_all = np.zeros(vocab.vocab_length)
    for cord_id in docs:
        sum_d_all += docs[cord_id]
    #sum_d_all/=len(docs)
    return sum_d_all

#%% Query expansion
## Query expansion -->

def expand_query(query, sum_dr, sum_d_all, alpha=1, beta=0.8, gamma=0.1, m_r=100, m_nr=4000):
    new_query = alpha* query + beta*sum_dr/m_r + gamma* (sum_d_all - sum_dr)/m_nr
    return new_query

#%% Similarity
#Might not require
## Calculate similarity between vectors 
# query : n, documents = mXn
def sim_score(query, documents, cord_ids):
    norm_q = np.linalg.norm(query)
    cos_sim = []
    for cord_id in cord_ids:
        d = documents[cord_id]
        sim = 1.0*np.dot(d, query)/(np.linalg.norm(d)*norm_q)
        cos_sim.append((cord_id, sim))
    return cos_sim

#%% write to file
def write_reranked_results(path, topic, cord_id, rank, score, run_id):
    with open(path, 'a+')as file:
        content = " ".join([topic, 'Q1', cord_id, str(rank), str(score), run_id])
        file.write(content + '\n')
    return

def clear_file(path):
    if os.path.exists(path):
        open(path, 'w+').close()
#%% Reranking docs

def reranking(alpha, beta, gamma, write_op = False, op_path = output_path):
    sum_relev_d = calc_sum_dr(docs, docs_by_topic)
    sum_all_d = calc_sum_d_all(docs)
    if write_op:
        clear_file(op_path)
    for topic in queries:
        new_query = expand_query(
            queries[topic],
            sum_relev_d[topic],
            sum_all_d,
            alpha, beta, gamma
            )
        sim_of_doc = sim_score(new_query, docs, docs_by_topic[topic])
        rank = 1
        for (cord_id, sim) in sorted(sim_of_doc, key = lambda val: -1*val[1]):
            if write_op:
                write_reranked_results(op_path, topic, cord_id, rank, sim, 'Harsh_vsm')
            rank +=1
    return
#%% call rerank
reranking(1, 0.03, 0.01, True)


### Later parts were used to tune hyperparameters
# Initialise parth or qrels using variable to run this --> 
# relevence_path = 'path'

#%% Evaluating the predictions
def evaluate(op_path, relevence_path):
    map_val = 0
    trec_eval = 'trec_eval-9.0.7/trec_eval'
    op = subprocess.run([trec_eval,'-m','map', relevence_path, op_path],   capture_output = True)
    map_val = float(op.stdout.decode().split()[-1])
    print(op.stdout.decode())
    return map_val

# Uncomment to run
#evaluate(output_path, relevence_path)

#%% Hyper parameter Tuning
def train_hp(alphas, betas, gamma):
    mat = np.zeros((len(alphas), len(betas)))
    op_path = temp_folder + 'op.txt'
    for a in range(len(alphas)):
        for b in range(len(betas)):
            reranking(alphas[a], betas[b], gamma, True, op_path)
            mat[a,b] = evaluate(op_path, relevence_path)
    max_at = (0,0)
    max_val = mat[0,0]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if max_val < mat[i,j]:
                max_val = mat[i,j]
                max_at = (i,j)
            print(mat[i,j], end = ' ')
        print('')
    print('best alpha', alphas[max_at[0]])
    print('best beta', betas[max_at[1]])
    return alphas[max_at[0]], betas[max_at[1]]
alphas = [1, 0.7, 0.5]
betas = [0.03, 0.3, 0.5, 0.7]
gamma = 0.01

# Uncomment this to run
#train_hp(alphas, betas, gamma)



#%%