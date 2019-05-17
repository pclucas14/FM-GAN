import cPickle
import pdb
import os
import glob
import sys

# word, vocab = cPickle.load(open('./'+vocab_file))
word, vocab = cPickle.load(
            open('./data/NewsData/vocab_news.pkl', 'rb'))
# input_file = 'save/coco_451.txt'

input_file = sys.argv[1] # './text_news/syn_val_words.txt' 
output_file = sys.argv[2] # './text_news/sents/Lucas_out.txt' #Real_out.txt

with open(output_file, 'w') as fout:
    with open(input_file) as fin:
        for line in fin:
            #line.decode('utf-8')
            line = line.split()
            #line.pop()
            #line.pop()
            line = [int(x) for x in line]
            line = [word[x] for x in line if x != 0]
            # if 'OTHERPAD' not in line:
            line = ' '.join(line) + '\n'
            fout.write(line)
            
