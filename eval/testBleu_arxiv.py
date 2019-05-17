from utils.metrics.Bleu import Bleu
import pdb
import numpy as np

# import cPickle
# loadpath = "./data/arxiv.p"
# x = cPickle.load(open(loadpath, 'rb'))
# test = x[2]
# max_length = 41

# output_file = './test_arxiv.txt'
# with open(output_file, 'w')as fout:
#     for l in test:
#         if len(l) <= max_length:
#             f1_np = np.asarray(l)
#             f1_filled = np.lib.pad(f1_np, (0, max_length-f1_np.shape[0]),
#                                     'constant', constant_values=0)
#             line = [str(x) for x in f1_filled]
#             line = ' '.join(line) + '\n'
#             fout.write(line)

# pdb.set_trace()

#real_text = './arxiv_result/arxiv_test.txt'  # arxiv_1w

#test_text = './arxiv_result/ot.txt'  # syn_val_words.txt


real_text = '/home/ml/lpagec/tensorflow/FM-GAN/src/text_news/sents/Real_out.txt'

for std in [0.01, 0.1, 0.5, 1, 1.1]:
    test_text = '../src/OUT_SAMPLES_STD/text_' + str(std) + '.txt' #../src/text_news/sents/Lucas_out.txt'

    for i in [5]:#range(2, 6):
        get_Bleu = Bleu(test_text=test_text, real_text=real_text, gram=i)
        score = get_Bleu.get_bleu_parallel()
        print(std, score)
    
#test_text = './arxiv_result/vae.txt'  # syn_val_words.txt

#for i in range(2, 6):
##    get_Bleu = Bleu(test_text=test_text, real_text=real_text, gram=i)
#    score = get_Bleu.get_bleu_parallel()
#    print(score)
