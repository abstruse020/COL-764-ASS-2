#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: harsh pandey
"""
"""
To do --
1. can calculate kl_div for p(w|R) != 0 only, and not for the whole vocab

"""

import mmap
import numpy as np
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
# from random import seed
# from random import randint
#%% Initializing Paths

# top_100_doc_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-top-100.txt'
top_100_doc_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-top-100.txt'
temp_folder = '/home/harsh/IITD/COL764-IR/Assignment2/temp_data/'
collection_dir = '/home/harsh/IITD/COL764-IR/Assignment2/2020-07-16'
if len(collection_dir) > 0 and collection_dir[-1] != '/':
    collection_dir+='/'
pdf_json_path = collection_dir + 'document_parses/pdf_json'
pmc_json_path = collection_dir + 'document_parses/pmc_json'
metadata_path = collection_dir + 'metadata.csv'
topic_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/covid19-topics.xml'
output_path = '/home/harsh/IITD/COL764-IR/Assignment2/output-file.txt'
relevence_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-qrels.txt'

models_size = 20

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
#%% preprocessing + vocab

def preprocessing(string, stop_word = []):
    words = word_tokenize(string, "english")
    words = [ps.stem(word) for word in words if word not in stop_word]
    words = list(filter(lambda x: len(x)>1, words))
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

#%% get whole docs as string
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
# print(topics)

#%% Make vector

def process_ranked_docs(path, cord_id_to_text, vocab, topics, stop_words = []):
    ## First making vocabulary and storing the processed form of documents for further processing(making vector etc)
    docs_by_topic = {}
    ranked_docs_info = {}
    processed_docs = {}
    processed_queries = {}
    doc_models = {}
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
                rank = int(rank)
                query = topics[topic]
                # print(topic, query)
                #[BookKeeping] topic wise cord_id list
                if topic not in  docs_by_topic:
                    docs_by_topic[topic] = []
                if topic not in doc_models:
                    doc_models[topic] = []
                # Only docs for models s.t rank <= modelsize
                if rank <= models_size:
                    doc_models[topic].append(cord_id)
                docs_by_topic[topic].append(cord_id)
                #[BookKeeping] query file information
                ranked_docs_info[(topic,cord_id)] = {'oldrank': rank, 'oldsim': similarity, 'query': query, 'run_id': run_id}
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
    
    return docs, queries, ranked_docs_info, vocab, docs_by_topic, doc_models

vocab = Vocabulary()
topics = read_topics(topic_path)

docs, queries, ranked_docs_info, vocab, docs_by_topic, doc_models = process_ranked_docs(
    top_100_doc_path,
    cord_uid_to_text,
    vocab,
    topics,
    stopwords.words('english'))
#%% write to file
def write_reranked_results(path, topic, cord_id, rank, score, run_id):
    with open(path, 'a+')as file:
        content = " ".join([topic, 'Q1', cord_id, str(rank), str(score), run_id])
        file.write(content + '\n')
    return

def clear_file(path):
    if os.path.exists(path):
        open(path, 'w+').close()
#%% P(w/Mj)

def Pw_dj(mu, word, doc_j):
    index_w = vocab.to_index(word)
    pc_w = vocab.word2count[word]/vocab.vocab_length
    d_len = np.sum(doc_j)
    ans = (doc_j[index_w] + mu*pc_w)/(d_len + mu)
    return ans

#%% RM1 class
class rm1:
    def __init__(self, mu, vocab, documents, models):
        self.mu = mu
        self.vocab = vocab
        self.documents = documents
        self.models = models
        
    def P_w_q(self, word, query_str, doc_model_id):
        # for all x models i.e x docs retrieved for query
        pw_q = 0
        
        for cord_id in doc_model_id:
            pq_d = 1
            for q_w in preprocessing(query_str):
                index_w = self.vocab.to_index(q_w)
                pc_w = self.vocab.word2count[q_w]/self.vocab.vocab_length
                d_len = np.sum(self.documents[cord_id])
                pw_dj = (self.documents[cord_id][index_w] + self.mu*pc_w)/(d_len + self.mu)
                
                pq_d *= pw_dj
                
            pw_q += Pw_dj(self.mu, word, self.documents[cord_id])*pq_d
        pw_q = pw_q/len(doc_model_id)
        return pw_q
    
    def expansion_term(self, query, q_id, query_str, cord_ids):
        p_qs = 0
        top_x_terms = 10 # max limit 20
        v = self.vocab.vocab_length
        prob_list = np.zeros(v)
        
        print('For query', query_str)
        for i in range(v):
            prob_list[i] = self.P_w_q(self.vocab.to_word(i), query_str, self.models[q_id])
            # prob_list[i] /= p_qs 
            if i%1000 == 0:
                print("{0:1.2f}".format(i*100/v), end=' ')
            if i> 1000:
                break
        #Dividing by P(q1,q2,...qk)
        p_qs = np.sum(prob_list)
        prob_list /= p_qs
            
        # picking top x terms (20 at max)
        top_x_index = np.argsort(prob_list)[-1 * top_x_terms:]
        expanded_query = np.zeros(v)
        for i in top_x_index:
            expanded_query[i] = prob_list[i]
        
        # Calculating P(w|D) for all retrieved queries
        sim_score = []#np.zeros(len(cord_ids))
        t2 = datetime.datetime.now()
        for cord_id, i in zip(cord_ids,range(len(cord_ids))):
            p_w_ds = np.zeros(v)
            for j in range(v):
                p_w_ds[j] = Pw_dj(self.mu, self.vocab.to_word(j), self.documents[cord_id])
            
            sim_score.append((
                cord_id,
                self.kl_div(expanded_query[top_x_index], p_w_ds[top_x_index])
                ))
        t3 = datetime.datetime.now()
        print('Time of KL div fast:', t3 - t2)
        # print(preprocessing(query_str))
        print('\nExpanded query')
        for i in top_x_index:
            if expanded_query[i] != 0:
                print(self.vocab.to_word(i),'{0:1.3f}'.format(expanded_query[i]), end = " ")
        print()
        return expanded_query, sim_score
    
    def process_queries(self, queries,docs_by_topic, op_path = None):
        if op_path != None:
            clear_file(op_path)
        v = self.vocab.vocab_length
        expanded_queries = np.zeros((
                len(queries),v
            ))
        for topic, i in zip(queries, range(len(queries))):
            n = len(docs_by_topic[topic])
            expanded_queries[i], sim_score = self.expansion_term(
                queries[topic],
                topic,
                topics[topic],
                docs_by_topic[topic]
                )
            rank = 1
            if op_path != None:
                for (cord_id, sim) in sorted(sim_score, key = lambda val: -1*val[1]):
                    write_reranked_results(op_path,
                                           topic,
                                           cord_id,
                                           rank,
                                           sim,
                                           'Harsh'
                                           )
                    rank +=1
        
        return
    
    def kl_div(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(q), 0))
    def kl_div_fast(self, p, q, index):
        p_, q_ = p[index], q[index]
        return np.sum(np.where(p_ != 0, p_ * np.log(q_), 0))
#%% RM1 For 1 query

r = rm1(0.5, vocab, docs, doc_models)
r.expansion_term(queries['1'],'1', topics['1'], docs_by_topic['1'])

#%% RM1 For all queries
r = rm1(0.5, vocab, docs, doc_models)
r.process_queries(queries, docs_by_topic, 'temp_data/rm1.txt')

#%% Evaluating the predictions
def evaluate(op_path, relevence_path):
    trec_eval = 'trec_eval-9.0.7/trec_eval'
    op = subprocess.run([trec_eval,'-m', 'map', relevence_path, op_path],   capture_output = True)
    # print(op)
    map_val = float(op.stdout.decode().split()[-1])
    print(map_val)
    return map_val

evaluate('temp_data/rm1.txt', relevence_path)
#%% RM 2 Model
class rm2:
    def __init__(self, mu, vocab, documents):
        self.mu = mu
        self.vocab = vocab
        self.documents = documents
    
    def P_w_R(self, word, topic, cord_ids):
        v = self.vocab.vocab_length
        n = len(cord_ids)
        p_w = 0
        pw_q = 1
        for cord_id in cord_ids:
            p_w += Pw_dj(self.mu, word, self.documents[cord_id])
        for qi in preprocessing(topic):
            p_qi_w = 0
            for cord_id in cord_ids:
                p_m_w = Pw_dj(self.mu, word, self.documents[cord_id])*n/p_w
                p_qi_w += p_m_w * Pw_dj(self.mu, qi, self.documents[cord_id])
            pw_q *= p_qi_w
        pw_q *= p_w
        return pw_q
    
    def P_word_givn_R(self, word, topic, cord_ids):
        return
    
    def expansion_term(self, query, topic, cord_ids):
        v = self.vocab.vocab_length
        new_query = np.zeros(v)
        print('For query:', topic)
        for i in range(v):
            new_query[i] = self.P_w_R(self.vocab.to_word(i), topic, cord_ids)
            if i%100 == 0:
                print("{0:1.2f}".format(i*100/v), end=' ')
            if i> 1000:
                break
        top_20_index = np.argsort(new_query)[-20:]
        expanded_query = np.zeros(v)
        for i in top_20_index:
            expanded_query[i] = new_query[i]
        print("New Query")
        for i in top_20_index:
            if expanded_query[i] != 0:
                print(self.vocab.to_word(i), end = " ")
        return expanded_query
    
    def process_queries(self, queries,docs_by_topic, op_path = None):
        v = self.vocab.vocab_length
        expanded_queries = np.zeros((
                len(queries),v
            ))
        for topic, i in zip(queries, range(len(queries))):
            n = len(docs_by_topic[topic])
            expanded_queries[i] = self.expansion_term(
                queries[topic],
                topic,
                docs_by_topic[topic]
                )
            
            sim_score = []
            for cord_id, j in zip(docs_by_topic[topic], range(n)):
                sim_score.append((
                    cord_id,
                    self.kl_div(expanded_queries[i],
                                self.documents[cord_id])))
            rank = 1
            if op_path != None:
                for (cord_id, sim) in sorted(sim_score, key = lambda val: -1*val[1]):
                    write_reranked_results(op_path, topic, cord_id, rank, sim, 'Harsh')
                    rank +=1
        return
    
    def kl_div(self, p_w_r, doc):
        sim = 0
        for i in range(self.vocab.vocab_length):
            word = self.vocab.to_word(i)
            p_w_md = Pw_dj(self.mu, word, doc)
            sim += p_w_r * np.log(p_w_md)
        return sim
    # def kl_div_mod(self, query, doc):
    #     sim = 0
    #     for i in vocab.vocab_length:
    #         p_w_r = P_word_givn_R(vocab.to_word(i), query)
    #         p_w_md = p_word_given_m(vocab.to_word(i), doc)
    #         sim += p_w_r * np.log(p_w_md)
    #     return sim
#%% RM2 for single query
r2 = rm2(0.5, vocab, docs)
r2.expansion_term(queries['1'], topics['1'], docs_by_topic['1'])

#%% RM2 for all queries
r2_all = rm2(0.5, vocab, docs)
r2_all.process_queries(queries, docs_by_topic)







