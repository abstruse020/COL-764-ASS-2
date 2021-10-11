#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:47:08 2021

@author: harsh
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
import json
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
ps = PorterStemmer()
# from random import seed
# from random import randint

#%%

# top_100_doc_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-top-100.txt'
top_100_doc_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-top-100.txt'
temp_folded = '/home/harsh/IITD/COL764-IR/Assignment2/temp_data/'
collection_dir = '/home/harsh/IITD/COL764-IR/Assignment2/2020-07-16'
if len(collection_dir) > 0 and collection_dir[-1] != '/':
    collection_dir+='/'
pdf_json_path = collection_dir + 'document_parses/pdf_json'
pmc_json_path = collection_dir + 'document_parses/pmc_json'
metadata_path = collection_dir + 'metadata.csv'
topic_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/covid19-topics.xml'

#%%
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
            authors = row['authors'].split('; ')
            # access the full text (if available) for Intro
            introduction = []
            if row['pdf_json_files']:
                for json_path in row['pdf_json_files'].split('; '):
                    with open(collection_dir + json_path) as f_json:
                        full_text_dict = json.load(f_json)
                        # grab introduction section from *some* version of the full text
                        for paragraph_dict in full_text_dict['body_text']:
                            paragraph_text = paragraph_dict['text']
                            section_name = paragraph_dict['section']
                            if 'intro' in section_name.lower():
                                introduction.append(paragraph_text)
                        # stop searching other copies of full text if already got introduction
                        if introduction:
                            break
            # save for later usage
            cord_uid_to_text[cord_uid].append({
                'title': title,
                'abstract': abstract,
                'introduction': introduction
            })
    t2 = datetime.datetime.now()
    print('Time to read metadata file' , t2-t1)
    return cord_uid_to_text

cord_uid_to_text = read_metadata(metadata_path, collection_dir)

#%%
## Show some data --
i = 0
for cord_uid in cord_uid_to_text:
    print('Cord id:', cord_uid)
    print('Total docs:',len(cord_uid_to_text[cord_uid]))
    print('Title:',cord_uid_to_text[cord_uid][0]['title'])
    print('abstract len:',len(cord_uid_to_text[cord_uid][0]['abstract']))
    print('intro len:',len(cord_uid_to_text[cord_uid][0]['introduction']))
    i+=1
    if i>=2:
        break

#%%
## preprocessing stage

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

#%%
## Using only the retrieved documents to calculate sum of relevent and non relevent vectors
def contents_for_cord_id(cord_id, cord_id_to_text):
    str_list = []
    for docs in cord_id_to_text[cord_id]:
        str_list.append(docs['title'])
        str_list.append(docs['abstract'])
        for intro in docs['introduction']:
            #print(intro[:10])
            str_list.append(intro)
    return ' '.join(str_list)
#Testing above code
contents_for_cord_id('a845az43', cord_uid_to_text)
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
# print(topics)
#%% Make vector

def process_ranked_docs(path, cord_id_to_text, vocab, topics, stop_words = []):
    ## First making vocabulary and storing the processed form of documents for further processing(making vector etc)
    docs_by_topic = {}
    ranked_docs_info = {}
    processed_docs = {}
    processed_queries = {}
    with open(path, 'r') as file:
        with mmap.mmap(file.fileno(),
                       length=0,
                       access=mmap.ACCESS_READ
                       ) as mmap_obj:
            i = 0
            for line in iter(mmap_obj.readline, b""):
                if line.decode().strip() == '':
                    continue
                topic, _, cord_id, rank, similarity, _ = line.decode()[:-1].split(" ")
                query = topics[topic]
                # print(topic, query)
                #[BookKeeping] topic wise cord_id list
                if topic not in  docs_by_topic:
                    docs_by_topic[topic] = []
                docs_by_topic[topic].append(cord_id)
                #[BookKeeping] query file information
                ranked_docs_info[(topic,cord_id)] = {'oldrank': rank, 'oldsim': similarity, 'query': query}
                words = preprocessing(contents_for_cord_id(cord_id, cord_id_to_text), stop_words)
                processed_docs[cord_id] = words
                for word in words:
                    vocab.add_word(word)
                
                q_words = preprocessing(query, stop_words)
                processed_queries[topic] = q_words
                for q_word in q_words:
                    vocab.add_word(q_word)
                i+=1
                # if i>=3:
                #     break
    docs = {}
    queries = {}
    for cord_id in processed_docs:
        docs[cord_id] = np.zeros(vocab.vocab_length)
        #Mind that the word might not be there in vocab --- Not possible
        for word in processed_docs[cord_id]:
            index = vocab.word2index[word]
            docs[cord_id][index] += 1
            
    for topic in processed_queries:
        queries[topic] = np.zeros(vocab.vocab_length)
        for word in processed_queries[topic]:
            index = vocab.word2index[word]
            queries[topic][index] += 1
            
        #print(docs[cord_id])
        #print(queries[cord_id])
    
    return docs, queries, ranked_docs_info, vocab, docs_by_topic

vocab = Vocabulary()
docs, queries, ranked_docs_info, vocab, docs_by_topic = process_ranked_docs(
    top_100_doc_path,
    cord_uid_to_text,
    vocab,
    topics)

#%% Print Vocab
## just printing vocab
print(vocab.vocab_length)
for i in range(vocab.vocab_length):
    print(vocab.index2word[i], end = " ")
    if i>=40:
        break
print(vocab.index2word[40])
# print(ranked_docs_info)
#%% Cals sum of relevent and non relevent docs
def calc_sum_dr(docs, docs_by_topic):
    sum_dr = {}
    for topic in docs_by_topic:
        sum_dr[topic] = np.zeros(len(docs[docs_by_topic[topic][0]]))
        #print(len(sum_dr[topic]))
        for cord_id in docs_by_topic[topic]:
            sum_dr[topic] += docs[cord_id]
    return sum_dr

#Just sum of call docs later while calculating subtract sum for topic i
def calc_sum_d_all(docs):
    sum_d_all = np.zeros(vocab.vocab_length)
    for cord_id in docs:
        sum_d_all += docs[cord_id]
    return sum_d_all

#%% Query expansion
## Query expansion -->

def expand_query(query, sum_dr, sum_d_all, alpha=1, beta=0.8, gamma=0.1, m_r=100, m_nr=3900):
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

#%% checking vars - 
# print(docs['h372mrr9'].shape)
# print(vocab.vocab_length)
# print(docs_by_topic['1'])
#%% Reranking docs

def reranking():
    sum_relev_d = calc_sum_dr(docs, docs_by_topic)
    sum_all_d = calc_sum_d_all(docs)
    # print('sum all docs', sum_all_d)
    alpha = 1
    beta = 0.8
    gamma = 0.1
    for topic in queries:
        print('for query:',topics[topic])
        print('For topic:', topic)
        # print('sum rel docs:', sum_relev_d[topic])
        
        new_query = expand_query(
            queries[topic],
            sum_relev_d[topic],
            sum_all_d,
            alpha, beta, gamma
            )
        sim_of_doc = sim_score(queries[topic], docs, docs_by_topic[topic])
        # print('similarity:', sim_of_doc)
        rank = 1
        for (cord_id, sim) in sorted(sim_of_doc, key = lambda val: -1*val[1]):
            query_local = ranked_docs_info[(topic, cord_id)]
            # print(topic, 'Q0', cord_id, rank, sim, query_local['query'][:])
            print('\t',topic, sim, query_local['oldsim'])
            print('\t',rank, query_local['oldrank'])
            rank +=1
            if rank>4:
                break
        if int(topic) > 5:
            break
    return
#%% call rerank
reranking()

        
        