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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import csv
import os
import json
from collections import defaultdict
ps = PorterStemmer()
# from random import seed
# from random import randint
#%%

# top_100_doc_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-top-100.txt'
top_100_doc_path = '/home/harsh/IITD/COL764-IR/Assignment2/GivenData/t40-top-100.txt'
temp_folded = '/home/harsh/IITD/COL764-IR/Assignment2/temp_data/'
collection_dir = '/home/harsh/IITD/COL764-IR/Assignment2/2020-07-16/'
if len(collection_dir) > 0 and collection_dir[-1] != '/':
    collection_dir+='/'
pdf_json_path = collection_dir + 'document_parses/pdf_json'
pmc_json_path = collection_dir + 'document_parses/pmc_json'
metadata_path = collection_dir + 'metadata.csv'

#%%
def read_files(path):
    file_content = ''
    with open(path, 'r') as file:
        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            file_content = mmap_obj.readline()
            print(file_content)
    return file_content

read_files(top_100_doc_path)


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
## Show somw data --
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

print(vocab.add_word('other'))
print(vocab.to_index('other'))
print(vocab.to_word(0))
print(vocab.get_count('other'))
print(vocab.vocab_length)

#%%
## Using only the retrieved documents to calculate sum of relevent and non relevent vectors

def string_for_cord_id(cord_id, cord_id_to_text):
    str_list = []
    for docs in cord_id_to_text[cord_id]:
        str_list.append(docs['title'])
        str_list.append(docs['abstract'])
        for intro in docs['introduction']:
            print(intro[:10])
            str_list.append(intro)
    return ' '.join(str_list)

#Testing above code
string_for_cord_id('a845az43', cord_uid_to_text)

#%%
    
    
def process_ranked_docs(path, cord_id_to_text, vocab, stop_words = []):
    ## First making vocabulary and storing the processed form of documents for further processing(making vector etc)
    processed_docs = {}
    processed_queries = {}
    with open(path, 'r') as file:
        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            i = 0
            for line in iter(mmap_obj.readline, b""):
                topic, _, cord_id, rank, similarity, query = line.decode().split(" ")
                words = preprocessing(string_for_cord_id(cord_id, cord_id_to_text), stop_words)
                processed_docs[cord_id] = words
                for word in words:
                    vocab.add_word(word)
                
                q_words = preprocessing(query, stop_words)
                processed_queries[cord_id] = q_words
                for q_word in q_words:
                    vocab.add_word(q_word)
                
                i+=1
                if i>=3:
                    break
    docs = {}
    queries = {}
    for cord_id in processed_docs:
        docs[cord_id] = np.zeros(vocab.vocab_length)
        queries[cord_id] = np.zeros(vocab.vocab_length)
        
        #Mind that word might not be there in vocab --- Not possible
        for word in processed_docs[cord_id]:
            index = vocab.word2index[word]
            docs[cord_id][index] += 1
        
        for word in processed_queries[cord_id]:
            index = vocab.word2index[word]
            queries[cord_id][index] += 1
            
        print(docs[cord_id])
        print(queries[cord_id])
    
    return docs, queries, vocab

docs, queries, vocab = process_ranked_docs(top_100_doc_path, cord_uid_to_text, vocab)

#%%
print(vocab.vocab_length)
for i in range(vocab.vocab_length):
    print(vocab.index2word[i], end = " ")
    if i>=40:
        break
print(vocab.index2word[40])
#%%
## Calculate similarity between vectors 
# query : n, documents = mXn
def sim_score(query, documents):
    return documents @ query