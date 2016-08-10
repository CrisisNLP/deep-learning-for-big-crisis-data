from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter


CR_embfile="/alt/work/ndat/crisis_all_data_vecs.bin"
GG_embfile="/home/local/QCRI/sjoty/dialog-act/data/word2vec_GoogleNews/GoogleNews-vectors-negative300.bin"

""" load the word embeddings """

print "Loading pre-trained word2vec embeddings......"

if CR_embfile.endswith(".gz"):
    cr_f = gzip.open(CR_embfile, 'rb')
else:
    cr_f = open(CR_embfile, 'rb')


cr_vec_size_got = int ( cr_f.readline().strip().split()[1]) 
print (cr_vec_size_got)


if GG_embfile.endswith(".gz"):
    gg_f = gzip.open(GG_embfile, 'rb')
else:
    gg_f = open(GG_embfile, 'rb')


gg_vec_size_got = int ( gg_f.readline().strip().split()[1])
print (gg_vec_size_got)


for line in cr_f: # read from the emb file
    all_ent   = line.split()
    word, vec = all_ent[0].lower(), map (float, all_ent[1:])
    print (vec)    

cr_f.close()



def load_emb(embfile, vocab_idmap, index_from=3, start_char=1, oov_char=2, padding_char=0, vec_size=300):
    """ load the word embeddings """

    print "Loading pre-trained word2vec embeddings......"

    if embfile.endswith(".gz"):
        f = gzip.open(embfile, 'rb')
    else:
        f = open(embfile, 'rb')

    vec_size_got = int ( f.readline().strip().split()[1]) # read the header to get vec dim

    if vec_size_got != vec_size:
        print " vector size provided and found in the file don't match!!!"
        raw_input(' ')
        exit(1)

    # load Embedding matrix
    row_nb = index_from+len(vocab_idmap)
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (row_nb, vec_size) )

    wrd_found = {}

    for line in f: # read from the emb file
        all_ent   = line.split()
        word, vec = all_ent[0].lower(), map (float, all_ent[1:])

        if vocab_idmap.has_key(word):
            wrd_found[word] = 1
            wid    = vocab_idmap[word] + index_from
            E[wid] = np.array(vec)

    f.close()
    print " Number of words found in emb matrix: " + str (len (wrd_found)) + " of " + str (len(vocab_idmap))

    return E
