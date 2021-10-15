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
# import math
import sys
# import pandas as pd
# import string
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
# top_100_doc_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-top-100.txt'
# temp_folder = '/home/harsh/IITD/COL764-IR/Assignment2/temp_data/'
# collection_dir = '/home/harsh/IITD/COL764-IR/Assignment2/2020-07-16'
# if len(collection_dir) > 0 and collection_dir[-1] != '/':
#     collection_dir+='/'
# pdf_json_path = collection_dir + 'document_parses/pdf_json'
# pmc_json_path = collection_dir + 'document_parses/pmc_json'
# metadata_path = collection_dir + 'metadata.csv'
# topic_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/covid19-topics.xml'

# output_path = '/home/harsh/IITD/COL764-IR/Assignment2/output-file.txt'
relevence_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-qrels.txt'

# models_size = 15

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
            # authors = row['authors'].split('; ')
            # access the full text (if available)
            
            
            # introduction = []
            # if row['pdf_json_files']:
            #     for json_path in row['pdf_json_files'].split('; '):
            #         with open(collection_dir + json_path) as f_json:
            #             full_text_dict = json.load(f_json)
            #             # grab introduction section from *some* version of the full text
            #             for paragraph_dict in full_text_dict['body_text']:
            #                 paragraph_text = paragraph_dict['text']
            #                 section_name = paragraph_dict['section']
            #                 if 'intro' in section_name.lower():
            #                     introduction.append(paragraph_text)
            #             # stop searching other copies of full text if already got introduction
            #             if introduction:
            #                 break
            
            
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

# cord_uid_to_text = read_metadata(metadata_path, collection_dir)
#%% preprocessing + vocab

def preprocessing(string, stop_word = stopwords.words('english')):
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
# def contents_for_cord_id(cord_id, cord_id_to_text):
#     str_list = []
#     for docs in cord_id_to_text[cord_id]:
#         str_list.append(docs['title'])
#         str_list.append(docs['abstract'])
#         for intro in docs['introduction']:
#             #print(intro[:10])
#             str_list.append(intro)
#     return ' '.join(str_list)
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
                            # section_name = paragraph_dict['section']
                            # if 'intro' in section_name.lower():
                            if paragraph_dict:
                                other_text.append(paragraph_text)
        if other_text == [] and docs['pmc_path']:
            for pmc_json in docs['pmc_path'].split('; '):
                with open(collection_dir + pmc_json) as f_json:
                        full_text_dict = json.load(f_json)
                        for paragraph_dict in full_text_dict['body_text']:
                            paragraph_text = paragraph_dict['text']
                            # section_name = paragraph_dict['section']
                            # if 'intro' in section_name.lower():
                            if paragraph_dict:
                                other_text.append(paragraph_text)
        str_list.extend(other_text)
    return ' '.join(str_list)

#Testing above code
# contents_for_cord_id('a845az43', cord_uid_to_text)

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
                doc_lengths[cord_id] = len(words)
                for word in words:
                    vocab.add_word(word)
                
                q_words = preprocessing(query, stop_words)
                processed_queries[topic] = q_words
                doc_lengths['$'+topic+'$'] = len(q_words)
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

# vocab = Vocabulary()
# topics = read_topics(topic_path)

# docs, queries, ranked_docs_info, vocab, docs_by_topic, doc_models = process_ranked_docs(
#     top_100_doc_path,
#     cord_uid_to_text,
#     vocab,
#     topics,
#     stopwords.words('english'))
#%% write to file
def write_reranked_results(path, topic, cord_id, rank, score, run_id):
    with open(path, 'a+') as file:
        content = " ".join([topic, 'Q1', cord_id, str(rank), str(score), run_id])
        file.write(content + '\n')
    return

def write_expansion_terms(topic, vector):
    contents = []
    for i in range(len(vector)):
        if vector[i] != 0:
            contents.append(vocab.to_word(i))
    contents = ", ".join(contents)
    with open(expansion_path, 'a+') as file:
        file.write(topic +' : ' + contents + '\n')
    

def clear_file(path):
    if os.path.exists(path):
        open(path, 'w+').close()
#%% P(w/Mj)

def Pw_dj(mu, word, doc_j, cord_id = None):
    index_w = vocab.to_index(word)
    pc_w = vocab.word2count[word]/vocab.vocab_length
    ans = 0
    if cord_id == None:
        d_len = np.sum(doc_j)
        ans = (doc_j[index_w] + mu*pc_w)/(d_len + mu)
    else:
        ans = (doc_j + mu*pc_w)/(doc_lengths[cord_id] + mu)
    return ans

#%% RM1 class
class rm1:
    def __init__(self, mu, vocab, documents, models):
        self.mu = mu
        self.vocab = vocab
        self.documents = documents
        self.models = models
        
    def P_w_q(self, word, query_str, doc_model_ids):
        # for all x models i.e x docs retrieved for query
        pw_q = 0
        for cord_id in doc_model_ids:
            pq_d = 1
            for q_w in preprocessing(query_str):
                indx = self.vocab.to_index(q_w)
                pw_dj = Pw_dj(self.mu, q_w, self.documents[cord_id][indx], cord_id)
                
                pq_d *= pw_dj
            indx = self.vocab.to_index(word)
            pw_q += Pw_dj(self.mu, word, self.documents[cord_id][indx], cord_id)*pq_d
        pw_q = pw_q/len(doc_model_ids)
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
            # if i> 10000:
            #     break
        print()
        #Dividing by P(q1,q2,...qk)
        p_qs = np.sum(prob_list)
        prob_list /= p_qs
            
        # picking top x terms (20 at max)
        top_x_index = np.argsort(prob_list)[-1 * top_x_terms:]
        expanded_query = np.zeros(v)
        for i in top_x_index:
            expanded_query[i] = prob_list[i]
        
        # Calculating P(w|D) for all retrieved queries
        sim_score = []
        t2 = datetime.datetime.now()
        kl_avg = 0
        for cord_id, i in zip(cord_ids,range(len(cord_ids))):
            p_w_ds = np.zeros(v)
            # Optimized by only passing top_x_index rather then the complete vector
            for j in top_x_index:
                p_w_ds[j] = Pw_dj(self.mu, self.vocab.to_word(j), self.documents[cord_id][j], cord_id)
            
            sim_score.append((
                cord_id,
                self.kl_div(expanded_query[top_x_index], p_w_ds[top_x_index])
                ))
            kl_avg += sim_score[-1][1]
        t3 = datetime.datetime.now()
        print('Time of KL div fast:', t3 - t2)
        print('sim avg val:', kl_avg)
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
            clear_file(expansion_path)
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
            write_expansion_terms(topic, expanded_queries[i])
            if op_path != None:
                for (cord_id, sim) in sorted(sim_score, key = lambda val: -1*val[1]):
                    write_reranked_results(op_path,
                                           topic,
                                           cord_id,
                                           rank,
                                           sim,
                                           'Harsh_rm1'
                                           )
                    rank +=1
        
        return
    
    def kl_div(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(q), 0))


#%% Evaluating the predictions
def evaluate(op_path, relevence_path):
    trec_eval = 'trec_eval-9.0.7/trec_eval'
    op = subprocess.run([trec_eval, relevence_path, op_path],   capture_output = True)
    # print(op)
    map_val = float(op.stdout.decode().split()[-1])
    print(op.stdout.decode())
    # print(map_val)
    return map_val

# evaluate('temp_data/rm1.txt', relevence_path)
# evaluate(top_100_doc_path, relevence_path)
#%% RM 2 Model
class rm2:
    def __init__(self, mu, vocab, documents, models):
        self.mu = mu
        self.vocab = vocab
        self.documents = documents
        self.models = models
    
    def P_word(self, word, doc_model_ids):
        p_w = 0
        for cord_id in doc_model_ids:
            indx = self.vocab.to_index(word)
            p_w+= Pw_dj(self.mu, word, self.documents[cord_id][indx], cord_id)
        return p_w/len(doc_model_ids)
    
    def P_qi_given_word(self, q_i, word, doc_model_ids):
        p_qi_w = 0
        indx_w = self.vocab.to_index(word)
        indx = self.vocab.to_index(q_i)
        p_w = self.P_word(word, doc_model_ids)
        for cord_id in doc_model_ids:
            p_qi_mj = Pw_dj(self.mu, q_i, self.documents[cord_id][indx], cord_id)
            p_mj_w = p_w * Pw_dj(self.mu, word, self.documents[cord_id][indx_w], cord_id)/len(doc_model_ids)
            p_qi_w += p_qi_mj * p_mj_w
        return p_qi_w
            
    def P_word_givn_R(self, word, query_str, doc_model_ids):
        p_qi_ws = 1
        p_w = self.P_word(word, doc_model_ids)
        for q_i in preprocessing(query_str):
            p_qi_ws *= self.P_qi_given_word(q_i, word, doc_model_ids)
        return p_w * p_qi_ws
    
    def expansion_term(self, query, q_id, query_str, cord_ids):
        p_qs = 0
        top_x_terms = 10 # max limit 20
        v = self.vocab.vocab_length
        prob_list = np.zeros(v)
        
        print('For query', query_str)
        for i in range(v):
            prob_list[i] = self.P_word_givn_R(self.vocab.to_word(i), query_str, self.models[q_id])
            # prob_list[i] /= p_qs 
            if i%100 == 0:
                print("{0:1.2f}".format(i*100/v), end=' ')
            # if i> 1000:
            #     break
        print()
        #Dividing by P(q1,q2,...qk)
        p_qs = np.sum(prob_list)
        prob_list /= p_qs
            
        # picking top x terms (20 at max)
        top_x_index = np.argsort(prob_list)[-1 * top_x_terms:]
        expanded_query = np.zeros(v)
        for i in top_x_index:
            expanded_query[i] = prob_list[i]
        
        # Calculating P(w|D) for all retrieved queries
        sim_score = []
        t2 = datetime.datetime.now()
        kl_avg = 0
        for cord_id, i in zip(cord_ids,range(len(cord_ids))):
            p_w_ds = np.zeros(v)
            # Optimized by only passing top_x_index rather then the complete vector
            for j in top_x_index:
                p_w_ds[j] = Pw_dj(self.mu, self.vocab.to_word(j), self.documents[cord_id][j], cord_id)
            
            sim_score.append((
                cord_id,
                self.kl_div(expanded_query[top_x_index], p_w_ds[top_x_index])
                ))
            kl_avg += sim_score[-1][1]
        t3 = datetime.datetime.now()
        print('Time of KL div fast:', t3 - t2)
        print('sim avg val:', kl_avg)
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
            clear_file(expansion_path)
            
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
            write_expansion_terms(topic, expanded_queries[i])
            if op_path != None:
                for (cord_id, sim) in sorted(sim_score, key = lambda val: -1*val[1]):
                    write_reranked_results(op_path,
                                           topic,
                                           cord_id,
                                           rank,
                                           sim,
                                           'Harsh_rm2'
                                           )
                    rank +=1
        
        return
    
    def kl_div(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(q), 0))


#%% Main()
# expansion_path = 'temp_data/exp1.txt'
rm_model = sys.argv[1]
print('model to use:', rm_model)
topic_path = sys.argv[2]
top_100_doc_path = sys.argv[3]
collection_dir = sys.argv[4]
if len(collection_dir) > 0 and collection_dir[-1] != '/':
    collection_dir+='/'
output_path = sys.argv[5]
expansion_path = sys.argv[6]
metadata_path = collection_dir + 'metadata.csv'
pdf_json_path = collection_dir + 'document_parses/pdf_json'
pmc_json_path = collection_dir + 'document_parses/pmc_json'
temp_folder = 'temp_data/'
    
models_size = 15

vocab = None
topics = None
docs =None
queries = None
ranked_docs_info= None
vocab = None
docs_by_topic=None
doc_models = None
doc_lengths = {}

def main():
    # rm_model = sys.argv[1]
    # print('model to use:', rm_model)
    # topic_path = sys.argv[2]
    # top_100_doc_path = sys.argv[3]
    # collection_dir = sys.argv[4]
    # if len(collection_dir) > 0 and collection_dir[-1] != '/':
    #     collection_dir+='/'
    # output_path = sys.argv[5]
    # expansion_path = sys.argv[6]
    # metadata_path = collection_dir + 'metadata.csv'
    # pdf_json_path = collection_dir + 'document_parses/pdf_json'
    # pmc_json_path = collection_dir + 'document_parses/pmc_json'
    # temp_folder = 'temp_data/'
    
    # models_size = 15
    global docs, queries, ranked_docs_info, vocab, docs_by_topic, doc_models, vocab, topics
    #Reading metadata csv
    print('reading metadata')
    cord_uid_to_text = read_metadata(metadata_path, collection_dir)
    print('building vocabulary')
    vocab = Vocabulary()
    topics = read_topics(topic_path)
    
    docs, queries, ranked_docs_info, vocab, docs_by_topic, doc_models = process_ranked_docs(
        top_100_doc_path,
        cord_uid_to_text,
        vocab,
        topics,
        stopwords.words('english'))
    
    if rm_model == 'rm1':
        r1 = rm1(0.5, vocab, docs, doc_models)
        r1.process_queries(queries, docs_by_topic, output_path)
        
    if rm_model == 'rm2':
        r2 = rm2(0.1, vocab, docs, doc_models)
        r2.process_queries(queries, docs_by_topic, output_path)
    
    evaluate(output_path, relevence_path)
    print('---------------------------END---------------------')
    return


if __name__ == '__main__':
    main()