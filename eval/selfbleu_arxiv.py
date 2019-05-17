import nltk
import re
import pdb

import sys
sys.path.append('/home/ml/lpagec/tensorflow/FM-GAN')
sys.path.append('/home/ml/lpagec/tensorflow/FM-GAN/pycocoevalcap')
sys.path.append('/home/ml/lpagec/tensorflow/FM-GAN/pycocoevalcap/bleu')
from pycocoevalcap.bleu.bleu import Bleu

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "s", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(5), ["Bleu_5"])
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

input_file = '../src/text_news/sents/Lucas_out.txt'
all_sents = []

with open(input_file, 'rb')as fin:
    for line in fin:
        #line.decode('utf-8')
        # line = re.sub(r",", "", line)
        # line = clean_str(line)
        # line = line.split()
        all_sents.append(line)

import numpy as np
ans = np.zeros(1)
for i in range(len(all_sents)):
    #print(i, ' / ', len(all_sents))
    tmp = all_sents[:]
    pop = tmp.pop(i)
    ref = {0: tmp}
    hop = {0: [pop]}

    import pdb; pdb.set_trace()
    ans[0] += score(ref, hop)['Bleu_5']
    # pdb.set_trace()
    print(ans[0] / i)

ans /= len(all_sents)
print('sink: ', ans)
