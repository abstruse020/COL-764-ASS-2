# -*- coding: utf-8 -*-
from sklearn.metrics import ndcg_score
import sys
import subprocess
import numpy as np

def evaluate(op_path, relevence_path):
    trec_eval = 'trec_eval-9.0.7/trec_eval'
    op = subprocess.run([trec_eval, relevence_path, op_path],   capture_output = True)
    # print(op)
    map_val = float(op.stdout.decode().split()[-1])
    print(op.stdout.decode())
    print(map_val)
    return map_val

def eval_dcg(op_path, relevence_path):
    relevence = {}
    lines = None
    true_rel = {}
    score = {}
    with open(relevence_path, 'r') as r_file:
        lines = r_file.read().split('\n')
    for line in lines:
        if len(line)<4:
            continue
        words = line.split(' ')
        query_id = words[0]
        cord_id = words[2]
        relev = words[3]
        if query_id not in relevence:
            relevence[query_id] = {}
        if cord_id not in relevence[query_id]:
            relevence[query_id][cord_id] = relev
    with open(op_path, 'r') as o_file:
        lines = o_file.read().split('\n')
    for line in lines:
        if len(line)<4:
            continue
        words = line.split(' ')
        query_id = words[0]
        cord_id = words[2]
        sc = float(words[4])
        if query_id not in true_rel:
            true_rel[query_id] = []
        if query_id not in score:
            score[query_id] = []
        rel = 0
        if cord_id in relevence[query_id]:
            rel = int(relevence[query_id][cord_id])
        true_rel[query_id].append(rel)
        score[query_id].append(sc)
        
    ndcg = 0
    k = 50
    for query_id in true_rel:
        ndcg += ndcg_score(np.array([true_rel[query_id]]),np.array([ score[query_id]]), k =k)
    print('nDGC @ %d : %f'%(k,ndcg))
    return ndcg

eval_dcg(sys.argv[1], sys.argv[2])
